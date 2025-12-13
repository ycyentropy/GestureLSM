#!/bin/bash

# Seamless数据集RVQ-VAE训练脚本
# 训练不同人体部位的RVQ-VAE模型

# 默认参数设置
BATCH_SIZE=4096
WINDOW_SIZE=64
CODE_DIM=128
NB_CODE=1024
EXP_NAME=""  # 将在后面根据其他参数自动生成
GPU_ID=0
SLEEP_TIME=10
PARTS="lower upper hands"  # 默认训练的身体部位

# 帮助信息
show_help() {
    echo "用法: $0 [选项]"
    echo "选项:"
    echo "  -b, --batch-size      批量大小 (默认: $BATCH_SIZE)"
    echo "  -w, --window-size     窗口大小 (默认: $WINDOW_SIZE)"
    echo "  -c, --code-dim        编码维度 (默认: $CODE_DIM)"
    echo "  -n, --nb-code         码本数量 (默认: $NB_CODE)"
    echo "  -e, --exp-name        实验名称 (默认: 根据其他参数自动生成)"
    echo "  -g, --gpu-id          GPU ID (默认: $GPU_ID)"
    echo "  -s, --sleep-time      训练间隙等待时间(秒) (默认: $SLEEP_TIME)"
    echo "  -p, --parts           要训练的身体部位，用空格分隔 (默认: '$PARTS')"
    echo "  -h, --help            显示帮助信息"
    echo "注意: 实验名称(exp-name)通常不需要指定，将根据其他参数自动生成"
    exit 0
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -w|--window-size)
            WINDOW_SIZE="$2"
            shift 2
            ;;
        -c|--code-dim)
            CODE_DIM="$2"
            shift 2
            ;;
        -n|--nb-code)
            NB_CODE="$2"
            shift 2
            ;;
        -e|--exp-name)
            EXP_NAME="$2"
            shift 2
            ;;
        -g|--gpu-id)
            GPU_ID="$2"
            shift 2
            ;;
        -s|--sleep-time)
            SLEEP_TIME="$2"
            shift 2
            ;;
        -p|--parts)
            PARTS="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "未知选项: $1"
            show_help
            ;;
    esac
done

# 如果没有指定实验名称，则根据其他参数自动生成
if [ -z "$EXP_NAME" ]; then
    EXP_NAME="seamless_${WINDOW_SIZE}frame_${BATCH_SIZE}batch_${CODE_DIM}dim_${NB_CODE}code"
    echo "自动生成实验名称: $EXP_NAME"
fi

# 人体部位列表和对应的中文描述
declare -A BODY_PARTS
BODY_PARTS["lower"]="下半身"
BODY_PARTS["upper"]="上半身"
BODY_PARTS["hands"]="手部"

# 显示当前配置
echo "=================================="
echo "Seamless数据集RVQ-VAE训练配置"
echo "=================================="
echo "批量大小: $BATCH_SIZE"
echo "窗口大小: $WINDOW_SIZE"
echo "编码维度: $CODE_DIM"
echo "码本数量: $NB_CODE"
echo "实验名称: $EXP_NAME"
echo "GPU ID: $GPU_ID"
echo "训练部位: $PARTS"
echo "=================================="

# 循环训练不同身体部位的模型
for part in $PARTS; do
    echo "=== 训练${BODY_PARTS[$part]}模型 ==="
    python rvq_seamless_train.py \
        --batch-size $BATCH_SIZE \
        --window-size $WINDOW_SIZE \
        --body_part $part \
        --code-dim $CODE_DIM \
        --nb-code $NB_CODE \
        --exp-name $EXP_NAME \
        --gpu-id $GPU_ID \
    
    echo "${BODY_PARTS[$part]}训练完成，等待${SLEEP_TIME}秒..."
    sleep $SLEEP_TIME
done

echo "🎉 所有训练任务完成！"
echo "模型保存在: outputs/rvq_$EXP_NAME/"