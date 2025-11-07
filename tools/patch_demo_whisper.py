#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""临时修改demo.py以使用本地Whisper模型"""
import os
# 读取原始文件
with open('demo.py', 'r') as original_file:
    content = original_file.read()

# 备份原始文件
with open('demo.py.bak', 'w') as backup_file:
    backup_file.write(content)

# 获取本地模型路径
local_model_path = os.environ.get('WHISPER_LOCAL_PATH', 'local_models/whisper-tiny.en')

# 替换模型路径
new_content = content.replace(
    'model="openai/whisper-tiny.en",',
    f'model="{{local_model_path}}",'
)

# 写回文件
with open('demo.py', 'w') as modified_file:
    modified_file.write(new_content)

print(f"已临时修改demo.py使用本地Whisper模型: {local_model_path}")
print("原始文件已备份为 demo.py.bak")
