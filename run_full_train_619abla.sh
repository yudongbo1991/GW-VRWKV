#!/bin/bash

# 1. 强制定位到项目根目录
cd "$(dirname "$0")"

# 2. 【核心修复】显式配置 CUDA 环境
# 这样即便在后台 nohup 运行，环境也是完全正确的
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 3. 打印 CUDA 版本信息到日志（用于双重检查）
echo "Using CUDA from: $CUDA_HOME"
nvcc --version

# 4. 执行顺序训练（增加 -u 确保日志实时刷新 HC+HH）
echo "Starting abla_1-only dual branch Task..."
python -u train_GW-VRWKV.py --config configs_619abla/sw_hanchuan_1.yaml

python -u train_GW-VRWKV.py --config configs_619abla/sw_honghu_1.yaml

echo "Starting abla_2-dual branch+fusion Task..."
python -u train_GW-VRWKV.py --config configs_619abla/sw_hanchuan_2.yaml

python -u train_GW-VRWKV.py --config configs_619abla/sw_honghu_2.yaml

echo "Starting abla_3-only ipm Task..."
python -u train_GW-VRWKV.py --config configs_619abla/sw_hanchuan_3.yaml

python -u train_GW-VRWKV.py --config configs_619abla/sw_honghu_3.yaml


echo "Starting abla_4-ipm+dual branch(no fusion) Task..."
python -u train_GW-VRWKV.py --config configs_619abla/sw_hanchuan_4.yaml

python -u train_GW-VRWKV.py --config configs_619abla/sw_honghu_4.yaml

echo "Starting full Task..."
python -u train_GW-VRWKV.py --config configs_619abla/sw_hanchuan_full.yaml

python -u train_GW-VRWKV.py --config configs_619abla/sw_honghu_full.yaml

echo "HC+HH训练任务完成！"









# 5. 执行顺序训练（增加 -u 确保日志实时刷新 IP+XZ）
echo "Starting abla_1-only dual branch Task..."
python -u train_GW-VRWKV.py --config configs_619abla/sw_indian_1.yaml

python -u train_GW-VRWKV.py --config configs_619abla/sw_xuzhou_1.yaml

echo "Starting abla_2-dual branch+fusion Task..."
python -u train_GW-VRWKV.py --config configs_619abla/sw_indian_2.yaml

python -u train_GW-VRWKV.py --config configs_619abla/sw_xuzhou_2.yaml

echo "Starting abla_3-only ipm Task..."
python -u train_GW-VRWKV.py --config configs_619abla/sw_indian_3.yaml

python -u train_GW-VRWKV.py --config configs_619abla/sw_xuzhou_3.yaml


echo "Starting abla_4-ipm+dual branch(no fusion) Task..."
python -u train_GW-VRWKV.py --config configs_619abla/sw_indian_4.yaml

python -u train_GW-VRWKV.py --config configs_619abla/sw_xuzhou_4.yaml

echo "Starting full Task..."
python -u train_GW-VRWKV.py --config configs_619abla/sw_indian_full.yaml

python -u train_GW-VRWKV.py --config configs_619abla/sw_xuzhou_full.yaml

echo "IP+XZ训练任务完成！"

echo "所有训练任务已成功执行完毕！"

# chmod +x run_full_train_619abla.sh

# source ~/.bashrc
# conda activate mambacp
# cd GW-RWKV
# nohup ./run_full_train_619abla.sh > 000619.txt 2>&1 &
