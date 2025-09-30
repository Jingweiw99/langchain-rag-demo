# 配置文件

# 模型路径配置
# 如果你的模型在其他位置，请修改这里的路径

# Qwen3-Embedding-4B 嵌入模型路径
EMBEDDING_MODEL_PATH = "/mnt/workspace/.cache/modelscope/models/Qwen/Qwen3-Embedding-4B"

# Qwen2.5-3B-Instruct 对话模型路径
LLM_MODEL_PATH = "/mnt/workspace/.cache/modelscope/models/Qwen/Qwen2___5-3B-Instruct"

# 向量数据库路径
VECTOR_DB_PATH = "./faiss/knowledge"

# 知识库文件路径
KNOWLEDGE_FILE = "knowledge.txt"

# 文档切分参数
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# 检索参数
TOP_K = 3  # 检索的相关文档数量

# 模型参数
MAX_TOKEN = 4096
TEMPERATURE = 0.8
TOP_P = 0.9
