#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""恢复demo.py到原始状态"""
import os
if os.path.exists('demo.py.bak'):
    with open('demo.py.bak', 'r') as backup_file:
        content = backup_file.read()
    with open('demo.py', 'w') as original_file:
        original_file.write(content)
    print("已恢复demo.py到原始状态")
else:
    print("未找到备份文件 demo.py.bak")
