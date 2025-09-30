#!/bin/bash

echo "================================"
echo "开始安装QA问答系统"
echo "================================"

# 检查Python版本
echo -e "\n检查Python版本..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "当前Python版本: $python_version"

# 升级pip
echo -e "\n升级pip..."
pip3 install --upgrade pip

# 安装依赖
echo -e "\n安装Python依赖..."
pip3 install -r requirements.txt

# 检查是否有GPU
echo -e "\n检查GPU..."
if command -v nvidia-smi &> /dev/null
then
    echo "检测到NVIDIA GPU"
    nvidia-smi
    echo -e "\n安装GPU版本的PyTorch..."
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "未检测到GPU，将使用CPU模式"
fi

# 创建必要的目录
echo -e "\n创建必要的目录..."
mkdir -p faiss

echo -e "\n================================"
echo "安装完成！"
echo "================================"
echo -e "\n下一步："
echo "1. 准备知识库文件 knowledge.txt"
echo "2. 运行: python3 get_vector.py  (创建向量数据库)"
echo "3. 运行: python3 main.py  (启动问答系统)"
echo ""
