# 基于知识库的问答系统

一个使用 Qwen 模型的智能问答系统，支持基于本地知识库的准确回答。

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 下载模型
python download_models.py

# 3. 配置模型路径（编辑 config.py）
# EMBEDDING_MODEL_PATH = "/your/path/to/embedding/model"
# LLM_MODEL_PATH = "/your/path/to/llm/model"

# 4. 创建向量数据库
python get_vector.py

# 5. 启动问答系统
python main.py
```

## 系统架构

### 核心技术栈

**AI 框架**
- **LLM模型**: Qwen/Qwen3-1.7B 或 Qwen2.5-3B-Instruct (通过ModelScope加载)
- **Embedding模型**: Qwen/Qwen3-Embedding-0.6B 或 Qwen3-Embedding-4B (通过ModelScope加载)
- **深度学习**: PyTorch 2.3.1+cpu
- **Transformers**: 4.55.2
- **ModelScope**: 1.29.0

**检索与向量化**
- **向量数据库**: FAISS 1.12.0
- **Sentence Transformers**: 5.1.0
- **LangChain**: 0.3.27
- **LangChain Community**: 0.3.30

**数据处理**
- **NumPy**: 1.26.4
- **Pandas**: 2.2.3
- **Datasets**: 3.2.0

**Web 框架（可选）**
- **FastAPI**: 0.116.1
- **Gradio**: 5.42.0
- **Uvicorn**: 0.35.0

**评估工具**
- **EvalScope**: 0.17.1
- **Rouge-Score**: 0.1.2
- **SacreBLEU**: 2.5.1

**支持设备**: 自动适配 CUDA GPU / CPU

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

### 3. 下载模型

使用 ModelScope 下载所需的模型：

```python
from modelscope.hub.snapshot_download import snapshot_download

# 下载 Qwen3-Embedding-4B 嵌入模型
embedding_model_dir = snapshot_download('Qwen/Qwen3-Embedding-4B')
print(f"嵌入模型路径：{embedding_model_dir}")

# 下载 Qwen2.5-3B-Instruct 对话模型
llm_model_dir = snapshot_download('Qwen/Qwen2.5-3B-Instruct')
print(f"对话模型路径：{llm_model_dir}")
```

**或者直接运行下载脚本：**
```bash
python download_models.py
```

### 4. 配置模型路径

下载完成后，编辑 `config.py` 文件，设置模型的本地路径：

```python
# 修改为你的实际模型路径
EMBEDDING_MODEL_PATH = "/path/to/.cache/modelscope/models/Qwen/Qwen3-Embedding-4B"
LLM_MODEL_PATH = "/path/to/.cache/modelscope/models/Qwen/Qwen2___5-3B-Instruct"
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
1. 从本地加载 Qwen3-Embedding-4B 模型
2. 加载并切分知识库文档
3. 生成向量并保存到 `./faiss/knowledge/` 目录

### 第三步：启动问答系统

```bash
python main.py
```

这个过程会：
1. 从本地加载 Qwen2.5-3B-Instruct 模型
2. 加载向量数据库
3. 启动交互式问答

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
├── README.md                 # 说明文档
├── requirements.txt          # Python依赖
├── config.py                 # 配置文件（模型路径等）
├── model.py                  # Qwen对话模型封装（基础版，适合Qwen3-1.7B）
├── model_with_sampling.py    # Qwen对话模型封装（高级版，支持温度采样）
├── embedding_model.py        # Qwen嵌入模型封装
├── get_vector.py             # 向量数据库创建脚本
├── main.py                   # 主程序（问答系统）
├── download_models.py        # 模型下载脚本
├── knowledge.txt             # 知识库文件（包含技术栈和示例数据）
└── faiss/                    # 向量数据库存储目录（自动创建）
    └── knowledge/
```

### 模型文件选择说明

项目提供了两个版本的模型封装文件，根据你使用的模型选择：

#### 1. `model.py` (基础版) ⭐ 推荐
- **适用模型**: Qwen3-1.7B、Qwen3-0.5B 等较小模型
- **特点**: 
  - 只使用基础生成参数（max_new_tokens、pad_token_id、eos_token_id）
  - 无警告信息，运行清爽
  - 适合资源受限环境
- **优势**: 启动快，无警告，稳定性好
- **使用**: 默认使用此版本，无需修改

#### 2. `model_with_sampling.py` (高级版)
- **适用模型**: Qwen2.5-3B-Instruct、Qwen2.5-7B 等支持高级采样的模型
- **特点**:
  - 支持 temperature（温度）参数，控制生成随机性
  - 支持 top_p（核采样）参数，控制生成多样性
  - 支持 repetition_penalty（重复惩罚）参数
- **优势**: 生成质量更可控，适合需要精细调优的场景
- **使用**: 在 `main.py` 中修改导入语句

**如何切换到高级版：**
```python
# 在 main.py 第 5 行修改
# from model import ChatModel  # 基础版
from model_with_sampling import ChatModel  # 高级版
```

### 两种版本对比表

| 特性 | model.py (基础版) | model_with_sampling.py (高级版) |
|------|------------------|-------------------------------|
| **适用模型** | Qwen3-1.7B, Qwen3-0.5B | Qwen2.5-3B-Instruct, Qwen2.5-7B |
| **温度采样** | ❌ 不支持 | ✅ 支持 (temperature) |
| **核采样** | ❌ 不支持 | ✅ 支持 (top_p) |
| **重复惩罚** | ❌ 不支持 | ✅ 支持 (repetition_penalty) |
| **警告信息** | ✅ 无警告 | ⚠️ 可能有警告（取决于模型） |
| **生成速度** | 🚀 较快 | 🐢 稍慢（更多计算） |
| **资源占用** | 💚 较低 | 💛 中等 |
| **推荐场景** | 生产环境、资源受限 | 开发调试、精细调优 |
| **默认使用** | ✅ 是 | ❌ 否（需手动切换） |

## 自定义配置

### 配置文件 (config.py)

所有核心配置都在 `config.py` 文件中，建议根据实际情况调整：

```python
# 模型路径 - 必须配置
EMBEDDING_MODEL_PATH = "/path/to/Qwen3-Embedding-0.6B"  # 嵌入模型
LLM_MODEL_PATH = "/path/to/Qwen3-1.7B"                  # 对话模型

# 向量数据库路径
VECTOR_DB_PATH = "./faiss/knowledge"

# 知识库文件
KNOWLEDGE_FILE = "knowledge.txt"

# 文档切分参数
CHUNK_SIZE = 500        # 文档块大小（建议 300-1000）
CHUNK_OVERLAP = 50      # 块重叠大小（建议 CHUNK_SIZE 的 10%）

# 检索参数
TOP_K = 3              # 检索文档数量（建议 2-5）

# 生成参数（注意：某些模型不支持这些参数）
MAX_TOKEN = 4096       # 最大 token 数
TEMPERATURE = 0.8      # 温度（0.1-1.0，越高越随机）
TOP_P = 0.9           # 核采样（0.1-1.0）
```

### 模型选择建议

| 模型 | 大小 | 显存需求 | 支持采样参数 | 推荐场景 | 使用模型文件 |
|------|------|---------|-------------|---------|------------|
| Qwen3-1.7B | 较小 | ~4GB | ❌ | CPU 或小显存 GPU | model.py |
| Qwen3-0.5B | 极小 | ~2GB | ❌ | 极限资源环境 | model.py |
| Qwen2.5-3B | 中等 | ~8GB | ✅ | 平衡性能和速度 | model_with_sampling.py |
| Qwen2.5-7B | 较大 | ~16GB | ✅ | 高质量生成 | model_with_sampling.py |
| Qwen3-Embedding-0.6B | 较小 | ~2GB | - | 嵌入模型（小显存） | embedding_model.py |
| Qwen3-Embedding-4B | 较大 | ~8GB | - | 嵌入模型（高质量） | embedding_model.py |

**注意**：
- ✅ 表示模型支持 temperature、top_p 等采样参数
- ❌ 表示模型不支持这些参数，使用会产生警告
- 根据模型类型选择对应的模型文件，避免警告信息

### 参数调优建议

**1. 提升回答准确性：**
```python
TOP_K = 5              # 检索更多相关文档
CHUNK_SIZE = 300       # 使用更小的文档块，提高匹配精度
CHUNK_OVERLAP = 50     # 增加重叠，避免遗漏信息
```

**2. 提升响应速度：**
```python
TOP_K = 2              # 减少检索数量
CHUNK_SIZE = 800       # 使用更大的文档块
# 在 model.py 中：max_new_tokens = 64  # 减少生成长度
```

**3. 节省显存：**
- 使用较小的模型（Qwen3-1.7B）
- 减小 `max_new_tokens` 参数
- 使用 CPU 模式

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

### Q1: 出现 "generation flags are not valid" 警告？

**A**: 这是正常的提示信息，不影响使用。某些 Qwen 模型（如 Qwen3-1.7B）不支持 `temperature`、`top_p`、`top_k` 等高级采样参数，系统会自动使用基础的生成参数。

**示例警告信息：**
```
The following generation flags are not valid and may be ignored: ['temperature', 'top_p']. 
Set `TRANSFORMERS_VERBOSITY=info` for more details.
```

**解决方案：**
- 这是模型特性，不是错误
- 已在代码中优化，只使用模型支持的参数
- 不影响问答功能的正常使用

### Q2: 回答中出现 `<think></think>` 标签？

**A**: 某些 Qwen 模型会输出思考过程标签。系统已自动过滤这些标签，只显示最终答案。

### Q3: 模型路径错误？

**A**: 请检查 `config.py` 中的模型路径是否正确。可以通过以下方式查找模型：
```bash
# Linux/Mac
find ~/.cache/modelscope -name "Qwen*" -type d

# Windows (PowerShell)
Get-ChildItem -Path "$env:USERPROFILE\.cache\modelscope" -Filter "Qwen*" -Recurse -Directory
```

### Q4: CUDA out of memory 错误？

**A**: GPU显存不足。可以：
1. 使用 CPU 模式（系统会自动降级）
2. 使用更小的模型（如 Qwen3-1.7B 替代 Qwen2.5-3B）
3. 减小 `max_new_tokens` 参数（在 `model.py` 中）
4. 关闭其他占用显存的程序

### Q5: 回答不准确？

**A**: 可能原因：
1. **知识库内容不够全面** - 补充更多相关信息到 `knowledge.txt`
2. **文档切分不合理** - 调整 `config.py` 中的 `CHUNK_SIZE` 和 `CHUNK_OVERLAP`
3. **检索的文档数量不够** - 增加 `config.py` 中的 `TOP_K` 值
4. **提示词不够明确** - 优化 `main.py` 中的 `PROMPT_TEMPLATE`

### Q6: 如何更新知识库？

**A**: 
1. 修改或替换 `knowledge.txt` 文件
2. 删除 `faiss/knowledge/` 目录
3. 重新运行 `python get_vector.py`
4. 重新启动 `python main.py`

### Q7: 能否支持多个知识库？

**A**: 可以。创建不同的知识库文件，并指定不同的向量数据库路径：

```python
# 为不同知识库创建不同的向量数据库
qa_system1 = QASystem(vector_db_path="./faiss/knowledge1")
qa_system2 = QASystem(vector_db_path="./faiss/knowledge2")
```

### Q8: 模型加载很慢？

**A**: 
1. **首次加载** - 模型需要从磁盘读取，较慢是正常的
2. **网络问题** - 如果模型未下载，确保网络畅通
3. **CPU 模式** - CPU 加载比 GPU 慢，建议使用较小的模型
4. **磁盘 I/O** - 使用 SSD 可以显著提升加载速度

### Q9: 如何在云服务器上部署？

**A**: 
1. **上传代码** - 使用 git 或 scp 上传项目文件
2. **安装依赖** - 运行 `pip install -r requirements.txt`
3. **下载模型** - 运行 `python download_models.py`
4. **配置路径** - 修改 `config.py` 中的模型路径
5. **创建向量库** - 运行 `python get_vector.py`
6. **启动服务** - 运行 `python main.py`

**后台运行建议：**
```bash
# 使用 nohup 后台运行
nohup python main.py > output.log 2>&1 &

# 使用 screen 或 tmux
screen -S qa_system
python main.py
# 按 Ctrl+A+D 分离会话
```

## 技术支持

如有问题，请检查：
1. Python版本是否 >= 3.8
2. 依赖是否完整安装
3. 网络连接是否正常（首次需要下载模型）
4. 磁盘空间是否充足

## 项目亮点

✨ **易于部署**
- 一键下载模型脚本
- 自动适配 GPU/CPU
- 完整的配置文件

🚀 **性能优化**
- 自动清理模型输出的思考标签
- 只使用模型支持的生成参数，避免警告
- 基于 FAISS 的高效向量检索

🎯 **灵活可扩展**
- 支持多种文档格式
- 可自定义提示词模板
- 支持多知识库管理
- 参数可配置，易于调优

💡 **用户友好**
- 交互式问答界面
- 清晰的错误提示
- 详细的文档和示例

## 更新日志

### v1.2 (2025-09-30) - 最新版本 🎉
- ✅ **新增** 两个版本的模型文件：
  - `model.py` - 基础版，适合 Qwen3-1.7B，无警告信息
  - `model_with_sampling.py` - 高级版，支持温度采样等参数
- ✅ **更新** knowledge.txt 添加完整的技术栈信息
- ✅ **更新** requirements.txt 添加具体版本号
- ✅ **完善** 文档，添加技术栈详细版本信息
- ✅ **添加** 模型版本对比表，方便选择
- ✅ **优化** 项目结构说明

### v1.1 (2025-09-30)
- ✅ 优化模型生成参数，移除不支持的 `temperature`、`top_p` 等参数
- ✅ 自动过滤 `<think></think>` 思考标签
- ✅ 完善 README 文档，添加常见问题和部署指南
- ✅ 优化代码注释和错误提示

### v1.0 (初始版本)
- 基础问答系统实现
- 支持 FAISS 向量检索
- ModelScope 模型集成
- 交互式问答界面

## 许可证

本项目仅供学习和研究使用。

## 模型许可

- Qwen2.5-3B-Instruct: [Apache 2.0](https://github.com/QwenLM/Qwen2.5)
- Qwen3-1.7B: 请查看 [Qwen 官方仓库](https://github.com/QwenLM/Qwen)
- Qwen3-Embedding 系列: 请查看 ModelScope 上的模型页面

## 贡献

欢迎提交 Issue 和 Pull Request！

## 致谢

- [Qwen](https://github.com/QwenLM/Qwen) - 提供优秀的开源模型
- [LangChain](https://github.com/langchain-ai/langchain) - 提供强大的框架
- [FAISS](https://github.com/facebookresearch/faiss) - 提供高效的向量检索
