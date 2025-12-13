import numpy as np

# 测试不同的np.savez调用方式
output_file = '/home/embodied/yangchenyu/GestureLSM/test_different_save.npz'

# 创建测试数据
global_orient = np.random.randn(100, 3).astype(np.float32)
body_pose = np.random.randn(100, 21, 3).astype(np.float32)
left_hand_pose = np.random.randn(100, 15, 3).astype(np.float32)
right_hand_pose = np.random.randn(100, 15, 3).astype(np.float32)
translation = np.random.randn(100, 3).astype(np.float32)

print("=== 方法1: 使用 **kwargs ===")
try:
    output_data = {
        "smplh:global_orient": global_orient,
        "smplh:body_pose": body_pose,
        "smplh:left_hand_pose": left_hand_pose,
        "smplh:right_hand_pose": right_hand_pose,
        "smplh:translation": translation
    }
    np.savez(output_file.replace('.npz', '_method1.npz'), **output_data)
    print("✅ 方法1成功")

    # 验证
    with np.load(output_file.replace('.npz', '_method1.npz')) as data:
        print(f'保存的键: {list(data.keys())}')

except Exception as e:
    print(f'❌ 方法1失败: {e}')

print("\n=== 方法2: 直接传递参数 ===")
try:
    np.savez(output_file.replace('.npz', '_method2.npz'),
             smplh_global_orient=global_orient,
             smplh_body_pose=body_pose,
             smplh_left_hand_pose=left_hand_pose,
             smplh_right_hand_pose=right_hand_pose,
             smplh_translation=translation)
    print("✅ 方法2成功")

    # 验证
    with np.load(output_file.replace('.npz', '_method2.npz')) as data:
        print(f'保存的键: {list(data.keys())}')

except Exception as e:
    print(f'❌ 方法2失败: {e}')

print("\n=== 方法3: 检查键名是否合法 ===")
test_keys = ["smplh:global_orient", "test:key", "normal_key"]
for key in test_keys:
    print(f'键名 "{key}": 是否合法 {":" not in key or np.savez.__doc__ is not None}')