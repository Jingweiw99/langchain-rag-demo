# 基于知识库的问答系统

一个使用 Qwen 模型的智能问答系统，支持基于本地知识库的准确回答。

## 系统架构

- **LLM模型**: Qwen/Qwen2.5-3B-Instruct (通过ModelScope加载)
- **Embedding模型**: Qwen/Qwen3-Embedding-4B (通过ModelScope加载)
- **向量数据库**: FAISS
- **框架**: LangChain

## 功能特点

1. ✅ 使用ModelScope自动下载和管理模型
2. ✅ 支持本地知识库问答
3. ✅ 基于FAISS的高效向量检索
4. ✅ 支持交互式问答
5. ✅ 自动GPU/CPU设备适配
6. ✅ 可自定义知识库内容

## 安装步骤

### 1. 环境要求

- Python 3.8+
- CUDA 11.8+ (可选，用于GPU加速)
- 至少 16GB RAM
- 至少 20GB 硬盘空间（用于存储模型）

### 2. 安装依赖

```bash
# 安装Python依赖
pip install -r requirements.txt

# 如果使用GPU，需要安装对应版本的PyTorch
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. 配置ModelScope

首次运行时，ModelScope会自动从服务器下载模型。默认模型会保存在：
- Linux/Mac: `~/.cache/modelscope/hub/`
- Windows: `C:\Users\用户名\.cache\modelscope\hub\`

如需修改模型缓存路径，可设置环境变量：
```bash
export MODELSCOPE_CACHE=/your/custom/path
```

## 使用方法

### 第一步：准备知识库

将您的知识库内容保存为 `knowledge.txt` 文件（已提供示例文件）。

支持的文件格式：
- `.txt` - 纯文本文件
- `.md` - Markdown文件
- `.pdf` - PDF文档（需要额外安装 `pypdf`）
- `.docx` - Word文档（需要额外安装 `python-docx`）

### 第二步：创建向量数据库

运行以下命令，将知识库转换为向量数据库：

```bash
python get_vector.py
```

这个过程会：
1. 从ModelScope下载 Qwen3-Embedding-4B 模型
2. 加载并切分知识库文档
3. 生成向量并保存到 `./faiss/knowledge/` 目录

**注意**: 首次运行会下载约8GB的embedding模型，请耐心等待。

### 第三步：启动问答系统

```bash
python main.py
```

这个过程会：
1. 从ModelScope下载 Qwen2.5-3B-Instruct 模型（首次运行）
2. 加载向量数据库
3. 启动交互式问答

**注意**: 首次运行会下载约6GB的LLM模型，请耐心等待。

### 使用示例

```
进入交互式问答模式
输入 'quit' 或 'exit' 退出
输入 'clear' 清空对话历史
--------------------------------------------------

请输入您的问题: 我身高170，体重65斤，应该买什么尺码？

正在思考...

回答: 根据尺码对照表，您身高170cm，体重65kg，建议选择L码。
L码适合身高165-170cm，体重57.5-65kg的人群。
```

## 项目文件说明

```
.
├── README.md              # 说明文档
├── requirements.txt       # Python依赖
├── model.py              # Qwen对话模型封装
├── embedding_model.py    # Qwen嵌入模型封装
├── get_vector.py         # 向量数据库创建脚本
├── main.py               # 主程序（问答系统）
├── knowledge.txt         # 知识库文件（示例）
└── faiss/                # 向量数据库存储目录（自动创建）
    └── knowledge/
```

## 自定义配置

### 修改模型参数

在 `model.py` 中修改：

```python
self.max_token = 4096      # 最大生成长度
self.temperature = 0.8     # 温度（越高越随机）
self.top_p = 0.9          # 核采样概率
```

### 修改检索参数

在 `main.py` 的 `ask()` 方法中修改：

```python
def ask(self, question: str, top_k: int = 3)  # top_k: 检索的文档数量
```

### 修改文档切分参数

在 `get_vector.py` 中修改：

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # 每个文档块的大小
    chunk_overlap=50       # 文档块之间的重叠
)
```

## 性能优化建议

### GPU加速

如果有NVIDIA GPU，系统会自动使用GPU加速。可以通过以下方式查看：

```python
import torch
print(torch.cuda.is_available())  # 返回True表示可以使用GPU
```

### 内存优化

如果内存不足，可以：

1. 使用量化版本的模型（需要额外配置）
2. 减小 `chunk_size` 和 `top_k` 参数
3. 关闭其他占用内存的程序

### 提升回答质量

1. **优化知识库**: 知识库内容要准确、结构化
2. **调整检索数量**: 增加 `top_k` 可以检索更多相关文档
3. **优化提示词**: 修改 `main.py` 中的 `PROMPT_TEMPLATE`
4. **调整温度**: 降低 `temperature` 可以让回答更确定性

## 常见问题

### Q1: 模型下载失败？

**A**: 可能是网络问题。可以：
1. 配置代理
2. 使用ModelScope镜像站点
3. 手动下载模型并放到缓存目录

### Q2: CUDA out of memory 错误？

**A**: GPU显存不足。可以：
1. 使用CPU模式（自动降级）
2. 减小batch size
3. 使用更小的模型

### Q3: 回答不准确？

**A**: 可能原因：
1. 知识库内容不够全面
2. 文档切分不合理（调整 `chunk_size`）
3. 检索的文档数量不够（增加 `top_k`）

### Q4: 如何更新知识库？

**A**: 
1. 修改或替换 `knowledge.txt` 文件
2. 删除 `faiss/knowledge/` 目录
3. 重新运行 `python get_vector.py`

### Q5: 能否支持多个知识库？

**A**: 可以。创建不同的知识库文件，并指定不同的向量数据库路径：

```python
qa_system = QASystem(vector_db_path="./faiss/knowledge2")
```

## 技术支持

如有问题，请检查：
1. Python版本是否 >= 3.8
2. 依赖是否完整安装
3. 网络连接是否正常（首次需要下载模型）
4. 磁盘空间是否充足

## 许可证

本项目仅供学习和研究使用。

## 模型许可

- Qwen2.5-3B-Instruct: [Apache 2.0](https://github.com/QwenLM/Qwen2.5)
- Qwen3-Embedding-4B: 请查看ModelScope上的模型页面
