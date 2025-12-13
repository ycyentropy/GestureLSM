import smplx
import torch

# 模型根目录（存放smplh文件夹的路径）
model_folder = "/home/embodied/yangchenyu/GestureLSM/datasets/hub/smplx_models"

# 加载SMPL-H模型（需指定性别，SMPL-H无中性模型）
model = smplx.create(
    model_folder,
    model_type="smplh",  # 指定模型类型为SMPL-H
    gender="neutral",    # 可选"male"或"female"
    num_betas=10,       # 形状参数数量（默认10）
    use_pca=False       # 不使用PCA降维（保留手部姿态完整参数）
)

# 输出模型基本信息
print(f"顶点数量: {model.get_num_verts()}")
print(f"关节数量: {model.NUM_JOINTS}")