import lmdb
import os

# LMDB数据库路径
lmdb_path = '/home/embodied/yangchenyu/GestureLSM/datasets/seamless_cache/seamless_smplh_24/train/smplh_cache'

def count_lmdb_entries(lmdb_path):
    # 打开LMDB数据库
    env = lmdb.open(lmdb_path, readonly=True)
    count = 0
    
    try:
        # 开始事务
        with env.begin() as txn:
            # 遍历所有键值对并计数
            with txn.cursor() as cursor:
                for _ in cursor:
                    count += 1
        return count
    finally:
        # 关闭数据库
        env.close()

if __name__ == '__main__':
    print(f"开始统计LMDB数据库中的条目数量: {lmdb_path}")
    total_entries = count_lmdb_entries(lmdb_path)
    print(f"LMDB数据库中共有 {total_entries} 条数据")
