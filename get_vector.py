from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from embedding_model import EmbeddingModel
from typing import List
import os

class CustomEmbeddings(Embeddings):
    """自定义Embeddings类，用于与LangChain集成"""
    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入文档列表"""
        embeddings = self.embedding_model.encode(texts)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """嵌入查询文本"""
        embedding = self.embedding_model.encode_single(text)
        return embedding.tolist()

def create_vector_db(knowledge_file: str, vector_db_path: str = "./faiss/knowledge"):
    """
    创建向量数据库
    :param knowledge_file: 知识库文件路径
    :param vector_db_path: 向量数据库保存路径
    """
    print(f"开始处理知识库文件: {knowledge_file}")
    
    # 第一步：加载文档
    loader = UnstructuredFileLoader(knowledge_file)
    data = loader.load()
    print(f'已加载文档数量: {len(data)}')

    # 第二步：切分文本
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    split_docs = text_splitter.split_documents(data)
    print(f'切分后的文档块数量: {len(split_docs)}')

    # 第三步：初始化嵌入模型
    print("正在加载嵌入模型...")
    embedding_model = EmbeddingModel(model_name="Qwen/Qwen3-Embedding-4B")
    embedding_model.load_model()
    
    embeddings = CustomEmbeddings(embedding_model)

    # 第四步：创建FAISS向量数据库
    print("正在创建向量数据库...")
    db = FAISS.from_documents(split_docs, embeddings)
    
    # 确保目录存在
    os.makedirs(os.path.dirname(vector_db_path), exist_ok=True)
    db.save_local(vector_db_path)
    print(f"向量数据库已保存到: {vector_db_path}")
    
    return split_docs

if __name__ == "__main__":
    # 示例：创建向量数据库
    knowledge_file = "knowledge.txt"
    if os.path.exists(knowledge_file):
        result = create_vector_db(knowledge_file)
        print(f"处理完成，共生成 {len(result)} 个文档块")
    else:
        print(f"知识库文件 {knowledge_file} 不存在，请先创建知识库文件")