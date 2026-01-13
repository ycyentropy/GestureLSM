import numpy as np
import glob
import os
import pickle
import lmdb
import json
import fasttext
from loguru import logger
from scipy import linalg
from tqdm import tqdm


class Vocab:
    PAD_token = 0
    SOS_token = 1
    EOS_token = 2
    UNK_token = 3

    def __init__(self, name, insert_default_tokens=True):
        self.name = name
        self.trimmed = False
        self.word_embedding_weights = None
        self.reset_dictionary(insert_default_tokens)

    def reset_dictionary(self, insert_default_tokens=True):
        self.word2index = {}
        self.word2count = {}
        if insert_default_tokens:
            self.index2word = {self.PAD_token: "<PAD>", self.SOS_token: "<SOS>",
                               self.EOS_token: "<EOS>", self.UNK_token: "<UNK>"}
        else:
            self.index2word = {self.UNK_token: "<UNK>"}
        self.n_words = len(self.index2word)  # count default tokens

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def add_vocab(self, other_vocab):
        for word, _ in other_vocab.word2count.items():
            self.index_word(word)

    # remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('    word trimming, kept %s / %s = %.4f' % (
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # reinitialize dictionary
        self.reset_dictionary()
        for word in keep_words:
            self.index_word(word)

    def get_word_index(self, word):
        if word in self.word2index:
            return self.word2index[word]
        else:
            return self.UNK_token

    def load_word_vectors(self, pretrained_path, embedding_dim=300):
        print("  loading word vectors from '{}'...".format(pretrained_path))

        # initialize embeddings to random values for special words
        init_sd = 1 / np.sqrt(embedding_dim)
        weights = np.random.normal(0, scale=init_sd, size=[self.n_words, embedding_dim])
        weights = weights.astype(np.float32)

        # read word vectors
        word_model = fasttext.load_model(pretrained_path)
        for word, id in self.word2index.items():
            vec = word_model.get_word_vector(word)
            weights[id] = vec
        self.word_embedding_weights = weights


def index_words_from_seamless_json(lang_model, data_path):
    """
    从 Seamless 数据集的 JSON 文件中提取词汇

    Args:
        lang_model: Vocab 实例
        data_path: Seamless 数据集根目录（包含 train/, val/, test/ 子目录）
    """
    # 定义要处理的子集目录
    subsets = ['train', 'val', 'test']
    total_files = 0
    processed_files = 0

    print(f'    indexing words from Seamless dataset at {data_path}')

    # 验证数据集目录结构
    expected_subsets = ['train', 'val', 'test']
    found_subsets = []
    for subset in expected_subsets:
        subset_path = os.path.join(data_path, subset)
        if os.path.exists(subset_path):
            found_subsets.append(subset)
        else:
            print(f'    Warning: {subset_path} does not exist, skipping...')

    if not found_subsets:
        raise ValueError(f"No subset directories found in {data_path}. "
                        f"Expected structure: {data_path}/train/, {data_path}/val/, {data_path}/test/")

    print(f'    found subsets: {found_subsets}')

    for subset in found_subsets:  # 只处理存在的子集
        subset_path = os.path.join(data_path, subset)
        print(f'    processing {subset} subset...')

        # 遍历第一层目录 (session ID)
        session_dirs = [d for d in os.listdir(subset_path)
                       if os.path.isdir(os.path.join(subset_path, d)) and not d.startswith('.')]

        for session_dir in tqdm(session_dirs, desc=f'Processing {subset} sessions'):
            session_path = os.path.join(subset_path, session_dir)

            # 遍历第二层目录 (gesture ID)
            gesture_dirs = [d for d in os.listdir(session_path)
                          if os.path.isdir(os.path.join(session_path, d)) and not d.startswith('.')]

            for gesture_dir in gesture_dirs:
                gesture_path = os.path.join(session_path, gesture_dir)

                # 查找 JSON 文件
                json_files = glob.glob(os.path.join(gesture_path, '*.json'))

                for json_file in json_files:
                    total_files += 1
                    try:
                        # 读取 JSON 文件
                        with open(json_file, 'r', encoding='utf-8') as f:
                            text_data = json.load(f)

                        # 提取词汇
                        if 'metadata:transcript' in text_data:
                            for transcript_item in text_data['metadata:transcript']:
                                if 'words' in transcript_item:
                                    for word_info in transcript_item['words']:
                                        if 'word' in word_info:
                                            word = word_info['word']
                                            # 清理标点符号，与原脚本保持一致
                                            word = word.replace(",", " ")
                                            word = word.replace(".", " ")
                                            word = word.replace("?", " ")
                                            word = word.replace("!", " ")
                                            # 去除多余空格
                                            word = word.strip()
                                            if word:  # 确保不为空
                                                lang_model.index_word(word)

                        processed_files += 1

                    except Exception as e:
                        print(f'    Error processing {json_file}: {str(e)}')
                        continue

    print(f'    processed {processed_files}/{total_files} files')
    print(f'    indexed %d words' % lang_model.n_words)
    print('    vocabulary stats:', len(lang_model.word2index), 'unique words')


def build_vocab_seamless(name, data_path, cache_path, word_vec_path=None, feat_dim=None):
    """
    构建 Seamless 数据集的词汇表

    Args:
        name: 词汇表名称
        data_path: Seamless 数据集路径
        cache_path: 输出的词汇表缓存路径
        word_vec_path: FastText 模型路径
        feat_dim: 词向量维度
    """
    print('  building a language model from Seamless dataset...')

    # 检查缓存是否已存在
    if os.path.exists(cache_path):
        print(f'    loading existing vocabulary from {cache_path}')
        with open(cache_path, 'rb') as f:
            lang_model = pickle.load(f)

        # 如果需要加载词向量
        if word_vec_path is not None:
            if lang_model.word_embedding_weights is None:
                lang_model.load_word_vectors(word_vec_path, feat_dim)
            elif lang_model.word_embedding_weights.shape[0] != lang_model.n_words:
                logger.warning('    word embedding weights size mismatch, reloading...')
                lang_model.load_word_vectors(word_vec_path, feat_dim)

        return lang_model

    # 创建新的词汇表
    lang_model = Vocab(name)
    print(f'    indexing words from {data_path}')
    index_words_from_seamless_json(lang_model, data_path)

    # 加载词向量
    if word_vec_path is not None:
        lang_model.load_word_vectors(word_vec_path, feat_dim)

    # 保存词汇表
    print(f'    saving vocabulary to {cache_path}')
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(lang_model, f)

    return lang_model


if __name__ == "__main__":
    # 示例用法 - 请根据实际路径修改
    # data_path 应该指向包含 train/, val/, test/ 子目录的根目录
    # 例如："/datasets/seamless_interaction/improvised"
    # 而不是："/datasets/seamless_interaction/improvised/train"
    # data_path = "/path/to/seamless/dataset"  # 请替换为实际的 Seamless 数据集根目录
    data_path = "/home/embodied/yangchenyu/GestureLSM/datasets/seamless_interaction/improvised"
    cache_path = os.path.join(data_path, "weights/new_vocab.pkl")
    word_vec_path = "/home/embodied/yangchenyu/GestureLSM/ckpt/wiki.en.bin"

    print("=== Seamless Dataset Vocabulary Builder ===")
    print(f"Dataset path: {data_path}")
    print(f"Expected structure: {data_path}/train/, {data_path}/val/, {data_path}/test/")
    print()

    # 检查数据路径是否存在
    if not os.path.exists(data_path):
        print(f"Error: Dataset path {data_path} does not exist!")
        print("Please modify the data_path variable in this script to point to your Seamless dataset.")
        print()
        print("Example:")
        print('  data_path = "/datasets/seamless_interaction/improvised"')
        print('  # This should be the directory containing train/, val/, test/ subdirectories')
        exit(1)

    # 检查 FastText 模型是否存在
    if not os.path.exists(word_vec_path):
        print(f"Error: FastText model {word_vec_path} does not exist!")
        exit(1)

    # 构建词汇表
    lang_model = build_vocab_seamless(
        name="seamless",
        data_path=data_path,
        cache_path=cache_path,
        word_vec_path=word_vec_path,
        feat_dim=300
    )

    print(f"Vocabulary built successfully!")
    print(f"Total words: {lang_model.n_words}")
    print(f"Vocabulary saved to: {cache_path}")