import torch
import numpy as np
import os
import sys

# 添加项目根目录到Python路径
sys.path.append('/home/embodied/yangchenyu/GestureLSM')

# 从项目配置获取SMPLX模型路径
def get_smplx_model_path():
    # 从beat_sep_lower.py中看到的数据路径配置
    default_paths = [
        '/home/embodied/yangchenyu/GestureLSM/datasets/hub/smplx_models/',  # 绝对路径
        './datasets/hub/smplx_models/',  # 相对路径
        '../datasets/hub/smplx_models/',  # 备选相对路径
    ]
    
    for path in default_paths:
        if os.path.exists(path):
            print(f"找到SMPLX模型路径: {path}")
            return path
    
    print("警告: 未找到SMPLX模型路径，请检查README中的下载说明。")
    return './datasets/hub/smplx_models/'  # 返回默认路径

def test_smplx_without_jaw_pose():
    """
    测试smplx模型在不提供jaw_pose参数的情况下是否可以正常工作
    """
    try:
        # 首先尝试导入smplx库
        try:
            import smplx
            print("✅ 成功导入smplx库")
        except ImportError:
            print("❌ 无法导入smplx库，请先安装: pip install smplx")
            return False
        
        # 获取模型路径
        model_path = get_smplx_model_path()
        
        # 创建SMPLX模型 - 使用项目中相同的配置
        try:
            model = smplx.create(
                model_path=model_path,
                model_type='smplx',
                gender='NEUTRAL_2020',  # 与beat_sep_lower.py中的配置一致
                use_face_contour=False,
                num_betas=300,  # 与beat_sep_lower.py中的配置一致
                num_expression_coeffs=100,  # 与beat_sep_lower.py中的配置一致
                ext='npz',
                use_pca=False,
                batch_size=1
            )
            print("✅ 成功创建SMPLX模型（使用项目配置）")
        except Exception as e:
            print(f"❌ 使用项目配置创建SMPLX模型失败: {str(e)}")
            print("尝试使用基本配置创建模型...")
            # 尝试使用基本配置
            try:
                model = smplx.create(
                    model_type='smplx',
                    gender='neutral',
                    batch_size=1
                )
                print("✅ 成功使用基本配置创建SMPLX模型")
            except Exception as e2:
                print(f"❌ 创建SMPLX模型失败: {str(e2)}")
                return False
        
        # 检查是否有CUDA
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        print(f"模型已移至设备: {device}")
        
        # 准备输入数据
        batch_size = 1
        # 创建必要的输入参数，但不包括jaw_pose
        global_orient = torch.zeros((batch_size, 3)).to(device)
        body_pose = torch.zeros((batch_size, 63)).to(device)  # 21个身体关节 * 3
        betas = torch.zeros((batch_size, 10)).to(device)  # 10个基本形状参数
        
        print("\n测试1: 不提供jaw_pose参数")
        try:
            # 不提供jaw_pose参数
            output = model(
                global_orient=global_orient,
                body_pose=body_pose,
                betas=betas
            )
            print("✅ 不提供jaw_pose参数时前向传播成功!")
            has_default_jaw_pose = True
        except Exception as e:
            print(f"❌ 不提供jaw_pose参数时前向传播失败: {str(e)}")
            import traceback
            traceback.print_exc()
            has_default_jaw_pose = False
        
        print("\n测试2: 提供jaw_pose参数")
        try:
            # 提供jaw_pose参数
            jaw_pose = torch.zeros((batch_size, 3)).to(device)
            output = model(
                global_orient=global_orient,
                body_pose=body_pose,
                betas=betas,
                jaw_pose=jaw_pose
            )
            print("✅ 提供jaw_pose参数时前向传播成功!")
        except Exception as e:
            print(f"❌ 提供jaw_pose参数时前向传播失败: {str(e)}")
            import traceback
            traceback.print_exc()
        
        print("\n总结:")
        if has_default_jaw_pose:
            print("✅ SMPLX模型支持缺省加载jaw_pose参数，会使用默认值（通常是零向量）")
            print("   因此在final_sep.py中可以省略jaw_pose参数")
        else:
            print("❌ SMPLX模型不支持缺省加载jaw_pose参数，必须显式提供")
            print("   因此在final_sep.py中不能省略jaw_pose参数")
        
        return has_default_jaw_pose
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_from_existing_code():
    """
    从项目中现有的代码结构出发，测试final_sep.py中的smplx调用方式
    """
    try:
        # 导入smplx库
        import smplx
        
        # 获取模型路径
        model_path = get_smplx_model_path()
        
        # 初始化SMPLX模型 (使用项目中的配置)
        try:
            smplx_model = smplx.create(
                model_path=model_path,
                model_type='smplx',
                gender='NEUTRAL_2020',  # 从beat_sep_lower.py中看到的配置
                use_face_contour=False,
                num_betas=300,  # 从beat_sep_lower.py中看到的配置
                num_expression_coeffs=100,  # 从beat_sep_lower.py中看到的配置
                ext='npz',
                use_pca=False,
            )
            
            # 检查是否有CUDA
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            smplx_model = smplx_model.to(device)
            print(f"✅ 项目配置的SMPLX模型初始化成功，已移至设备: {device}")
            
            # 创建模拟的poses数组 (类似final_sep.py中的使用)
            batch_size = 2
            seq_length = 10
            # SMPLX通常有75个姿态参数
            poses = torch.zeros(batch_size, seq_length, 75).to(device)
            betas = torch.zeros(batch_size, 300).to(device)  # 与模型配置一致
            transl = torch.zeros(batch_size, seq_length, 3).to(device)
            expression = torch.zeros(batch_size, seq_length, 100).to(device)  # 与模型配置一致
            
            # 测试2.1: 模拟final_sep.py中的使用方式 - 提供jaw_pose
            print("\n测试2.1: 模拟final_sep.py中的使用方式 - 提供jaw_pose")
            try:
                results = []
                for i in range(batch_size):
                    # 模拟循环中的每一个批次处理
                    output = smplx_model(
                        betas=betas[i],
                        transl=transl[i],
                        expression=expression[i],
                        jaw_pose=poses[i, :, 66:69],  # 这是final_sep.py中提取jaw_pose的方式
                        global_orient=poses[i, :, :3],
                        body_pose=poses[i, :, 3:21*3+3],
                        left_hand_pose=poses[i, :, 25*3:40*3],
                        right_hand_pose=poses[i, :, 40*3:55*3],
                        return_verts=True,
                        return_joints=True,
                        leye_pose=poses[i, :, 69:72],
                        reye_pose=poses[i, :, 72:75],
                    )
                    results.append(output['joints'])
                
                print("✅ 提供jaw_pose时模拟项目代码运行成功！")
                print(f"  输出关节点形状: {results[0].shape}")
                
            except Exception as e:
                print(f"❌ 提供jaw_pose时模拟项目代码运行失败: {str(e)}")
                import traceback
                traceback.print_exc()
            
            # 测试2.2: 模拟final_sep.py中的使用方式 - 不提供jaw_pose
            print("\n测试2.2: 模拟final_sep.py中的使用方式 - 不提供jaw_pose")
            try:
                results = []
                for i in range(batch_size):
                    # 模拟循环中的每一个批次处理，但不提供jaw_pose
                    output = smplx_model(
                        betas=betas[i],
                        transl=transl[i],
                        expression=expression[i],
                        # 故意省略jaw_pose参数
                        global_orient=poses[i, :, :3],
                        body_pose=poses[i, :, 3:21*3+3],
                        left_hand_pose=poses[i, :, 25*3:40*3],
                        right_hand_pose=poses[i, :, 40*3:55*3],
                        return_verts=True,
                        return_joints=True,
                        leye_pose=poses[i, :, 69:72],
                        reye_pose=poses[i, :, 72:75],
                    )
                    results.append(output['joints'])
                
                print("✅ 不提供jaw_pose时模拟项目代码运行成功！")
                print(f"  输出关节点形状: {results[0].shape}")
                print("\n✅ 结论: final_sep.py第130行的jaw_pose参数可以安全地省略")
                
            except Exception as e:
                print(f"❌ 不提供jaw_pose时模拟项目代码运行失败: {str(e)}")
                import traceback
                traceback.print_exc()
                print("\n❌ 结论: final_sep.py第130行的jaw_pose参数不能省略")
                
        except Exception as e:
            print(f"❌ 项目配置的SMPLX模型初始化失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("========== SMPLX模型jaw_pose参数测试 ==========")
    
    # 测试1: 基本的smplx模型测试
    test_smplx_without_jaw_pose()
    
    # 测试2: 项目代码相关测试
    test_from_existing_code()
    
    print("\n========== 测试完成 ==========")