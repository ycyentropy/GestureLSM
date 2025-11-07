#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
下载OpenAI Whisper模型到本地并设置环境变量，以便demo.py可以从本地加载模型
"""

import os
import sys
from transformers import WhisperProcessor, WhisperForConditionalGeneration


def download_whisper_model(local_path="local_models/whisper-tiny.en", model_id="openai/whisper-tiny.en"):
    """下载Whisper模型到本地目录（如果尚不存在）"""
    if not os.path.exists(local_path):
        os.makedirs(local_path, exist_ok=True)
        print(f"正在下载Whisper模型 {model_id} 到 {local_path}...")
        
        # 下载模型和处理器
        try:
            model = WhisperForConditionalGeneration.from_pretrained(model_id)
            processor = WhisperProcessor.from_pretrained(model_id)
            
            # 保存到本地路径
            model.save_pretrained(local_path)
            processor.save_pretrained(local_path)
            print(f"Whisper模型已下载并保存到 {local_path}")
        except Exception as e:
            print(f"下载模型时出错: {e}")
            sys.exit(1)
    else:
        print(f"Whisper模型已存在于 {local_path}")
    
    return local_path


def create_env_script(local_model_path):
    """创建设置环境变量的脚本"""
    # 创建bash脚本
    with open("use_local_whisper.sh", "w") as f:
        f.write(f"#!/bin/bash\n")
        f.write(f"# 设置环境变量以使用本地Whisper模型\n")
        f.write(f"export WHISPER_LOCAL_PATH=\"{os.path.abspath(local_model_path)}\"\n")
        f.write(f"echo \"环境变量已设置，将使用本地Whisper模型: {os.path.abspath(local_model_path)}\"\n")
    
    # 创建Python包装脚本
    with open("use_local_whisper.py", "w") as f:
        f.write(f"#!/usr/bin/env python\n")
        f.write(f"# -*- coding: utf-8 -*-\n")
        f.write(f"import os\n")
        f.write(f"import subprocess\n")
        f.write(f"# 设置环境变量以使用本地Whisper模型\n")
        f.write(f"os.environ['WHISPER_LOCAL_PATH'] = '{os.path.abspath(local_model_path)}'\n")
        f.write(f"print('环境变量已设置，将使用本地Whisper模型')\n")
        f.write(f"# 调用demo.py\n")
        f.write(f"subprocess.run(['python', 'demo.py'])\n")
    
    # 设置脚本可执行权限
    os.chmod("use_local_whisper.sh", 0o755)
    os.chmod("use_local_whisper.py", 0o755)
    
    print("已创建环境变量设置脚本:")
    print("  - use_local_whisper.sh (bash脚本)")
    print("  - use_local_whisper.py (Python包装脚本)")


def create_model_patch(local_model_path):
    """创建一个简单的补丁脚本，用于临时修改demo.py以使用本地模型"""
    with open("patch_demo_whisper.py", "w") as f:
        f.write("#!/usr/bin/env python\n")
        f.write("# -*- coding: utf-8 -*-\n")
        f.write("\"\"\"临时修改demo.py以使用本地Whisper模型\"\"\"\n")
        f.write("import os\n")
        f.write("# 读取原始文件\n")
        f.write("with open('demo.py', 'r') as original_file:\n")
        f.write("    content = original_file.read()\n")
        f.write("\n")
        f.write("# 备份原始文件\n")
        f.write("with open('demo.py.bak', 'w') as backup_file:\n")
        f.write("    backup_file.write(content)\n")
        f.write("\n")
        f.write("# 获取本地模型路径\n")
        f.write("local_model_path = os.environ.get('WHISPER_LOCAL_PATH', 'local_models/whisper-tiny.en')\n")
        f.write("\n")
        f.write("# 替换模型路径\n")
        f.write("new_content = content.replace(\n")
        f.write("    'model=\"openai/whisper-tiny.en\",',\n")
        f.write("    f'model=\"{{local_model_path}}\",'\n")
        f.write(")\n")
        f.write("\n")
        f.write("# 写回文件\n")
        f.write("with open('demo.py', 'w') as modified_file:\n")
        f.write("    modified_file.write(new_content)\n")
        f.write("\n")
        f.write("print(f\"已临时修改demo.py使用本地Whisper模型: {local_model_path}\")\n")
        f.write("print(\"原始文件已备份为 demo.py.bak\")\n")
    
    # 创建恢复脚本
    with open("restore_demo.py", "w") as f:
        f.write("#!/usr/bin/env python\n")
        f.write("# -*- coding: utf-8 -*-\n")
        f.write("\"\"\"恢复demo.py到原始状态\"\"\"\n")
        f.write("import os\n")
        f.write("if os.path.exists('demo.py.bak'):\n")
        f.write("    with open('demo.py.bak', 'r') as backup_file:\n")
        f.write("        content = backup_file.read()\n")
        f.write("    with open('demo.py', 'w') as original_file:\n")
        f.write("        original_file.write(content)\n")
        f.write("    print(\"已恢复demo.py到原始状态\")\n")
        f.write("else:\n")
        f.write("    print(\"未找到备份文件 demo.py.bak\")\n")
    
    # 设置脚本可执行权限
    os.chmod("patch_demo_whisper.py", 0o755)
    os.chmod("restore_demo.py", 0o755)
    
    print("\n已创建补丁脚本:")
    print("  - patch_demo_whisper.py (临时修改demo.py)")
    print("  - restore_demo.py (恢复demo.py到原始状态)")
    print("\n使用方法:")
    print("  1. 设置环境变量: export WHISPER_LOCAL_PATH=path/to/local/model")
    print("  2. 运行补丁脚本: python patch_demo_whisper.py")
    print("  3. 运行demo.py: python demo.py")
    print("  4. 完成后恢复: python restore_demo.py")


def main():
    print("开始设置本地Whisper模型...")
    
    # 下载模型到本地
    local_model_path = download_whisper_model()
    
    # 创建环境变量脚本
    create_env_script(local_model_path)
    
    # 创建补丁脚本
    create_model_patch(local_model_path)
    
    print("\n设置完成！有两种使用本地Whisper模型的方法:")
    print("\n方法1: 使用包装脚本直接运行demo (推荐)")
    print("  - Python包装脚本: python use_local_whisper.py")
    print("  - Bash包装脚本: ./use_local_whisper.sh && python demo.py")
    print("\n方法2: 手动临时修改demo.py")
    print("  1. 运行补丁脚本: python patch_demo_whisper.py")
    print("  2. 运行demo.py: python demo.py")
    print("  3. 完成后恢复: python restore_demo.py")


if __name__ == "__main__":
    main()