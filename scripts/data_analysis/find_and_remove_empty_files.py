import os
import numpy as np
import argparse
import logging
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("empty_files.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def is_empty_npz_file(file_path):
    """
    检查NPZ文件是否为空（body_pose或global_orient帧数量为0）
    """
    try:
        # 加载NPZ文件
        data = np.load(file_path, allow_pickle=True)
        
        # 检查是否包含必要的键
        # 尝试两种可能的键格式: 嵌套结构和扁平化结构
        if 'smplh' in data:
            # 嵌套结构
            smplh_data = data['smplh'].item()
            if 'body_pose' not in smplh_data or 'global_orient' not in smplh_data:
                logger.warning(f"文件 {file_path} 的'smplh'中缺少必要键")
                return False
            
            body_pose = smplh_data['body_pose']
            global_orient = smplh_data['global_orient']
        elif 'smplh:body_pose' in data and 'smplh:global_orient' in data:
            # 扁平化结构
            body_pose = data['smplh:body_pose']
            global_orient = data['smplh:global_orient']
        else:
            logger.warning(f"文件 {file_path} 不包含必要的键")
            return False
        
        # 检查帧数量是否为0
        if (hasattr(body_pose, 'shape') and body_pose.shape[0] == 0) or \
           (hasattr(global_orient, 'shape') and global_orient.shape[0] == 0):
            logger.info(f"发现空文件: {file_path}, body_pose.shape={body_pose.shape}, global_orient.shape={global_orient.shape}")
            return True
            
        return False
        
    except Exception as e:
        logger.error(f"检查文件 {file_path} 时出错: {str(e)}")
        return False

def find_empty_npz_files(root_dir):
    """
    遍历目录找到所有空的NPZ文件
    """
    empty_files = []
    
    # 获取所有npz文件的总数
    total_files = 0
    for root, _, files in os.walk(root_dir):
        total_files += len([f for f in files if f.endswith('.npz')])
    
    logger.info(f"开始扫描 {total_files} 个NPZ文件...")
    
    # 遍历目录
    with tqdm(total=total_files, desc="扫描文件") as pbar:
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.npz'):
                    file_path = os.path.join(root, file)
                    if is_empty_npz_file(file_path):
                        empty_files.append(file_path)
                    pbar.update(1)
    
    logger.info(f"扫描完成，发现 {len(empty_files)} 个空NPZ文件")
    return empty_files

def find_related_files(base_file_path):
    """
    找到与指定文件同名的其他文件（不同扩展名）
    """
    related_files = []
    base_dir, base_filename = os.path.split(base_file_path)
    base_name_without_ext = os.path.splitext(base_filename)[0]
    
    # 获取目录中所有文件
    for file in os.listdir(base_dir):
        file_name_without_ext = os.path.splitext(file)[0]
        if file_name_without_ext == base_name_without_ext:
            related_file_path = os.path.join(base_dir, file)
            # 只添加非NPZ文件和原始NPZ文件
            if related_file_path != base_file_path:
                related_files.append(related_file_path)
    
    return related_files

def main():
    parser = argparse.ArgumentParser(description="查找并删除空的NPZ数据文件及其同名文件")
    parser.add_argument("--data_dir", type=str, default="./datasets", 
                        help="要扫描的数据集根目录")
    parser.add_argument("--dry_run", action="store_true", default=True,
                        help="仅显示要删除的文件，不执行实际删除操作")
    parser.add_argument("--no-dry-run", action="store_true", default=False,
                        help="执行实际删除操作，覆盖--dry_run参数")
    args = parser.parse_args()
    
    # 如果指定了--no-dry-run，则覆盖--dry_run设置
    if args.no_dry_run:
        args.dry_run = False
    
    # 查找空的NPZ文件
    empty_files = find_empty_npz_files(args.data_dir)
    
    if not empty_files:
        logger.info("没有发现空的NPZ文件，任务完成。")
        return
    
    # 收集所有要删除的文件
    all_files_to_delete = []
    for empty_file in empty_files:
        # 添加空的NPZ文件
        all_files_to_delete.append(empty_file)
        
        # 添加相关的同名文件
        related_files = find_related_files(empty_file)
        all_files_to_delete.extend(related_files)
    
    # 去重
    all_files_to_delete = list(set(all_files_to_delete))
    
    # 输出结果
    logger.info(f"总计找到 {len(all_files_to_delete)} 个文件需要删除:")
    for file in all_files_to_delete:
        logger.info(f"  {file}")
    
    # 确认删除
    if not args.dry_run:
        confirm = input(f"确认要删除这 {len(all_files_to_delete)} 个文件吗？(y/n): ")
        if confirm.lower() == 'y':
            deleted_count = 0
            for file in tqdm(all_files_to_delete, desc="删除文件"):
                try:
                    os.remove(file)
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"删除文件 {file} 失败: {str(e)}")
            
            logger.info(f"成功删除 {deleted_count}/{len(all_files_to_delete)} 个文件")
        else:
            logger.info("取消删除操作")
    else:
        logger.info("这是一次演练运行，没有实际删除任何文件。使用 --no-dry-run 参数执行实际删除")

if __name__ == "__main__":
    main()