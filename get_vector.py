from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS # 向量数据库

def main():
    # 定义向量模型路径
    EMBEDDING_MODEL = "modelscope.cn/Qwen/Qwen2.5-3B-Instruct-GGUF:latest"

    # 第一步：加载文档
    loader = UnstructuredFileLoader("衣服属性.txt")
    # 将文本转成 Document 对象
    data = loader.load()
    print(f'documents:{len(data)}')

    # 第二部：切分文本
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    # 切割加载的 document
    split_docs = text_splitter.split_documents(data)
    # print("split_docs size:",len(split_docs))
    # print(split_docs)


    # 第三步：初始化 hugginFace 的 embeddings 对象
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # 第四步：将 document通过embeddings对象计算得到向量信息并永久存入FAISS向量数据库，用于后续匹配查询
    db = FAISS.from_documents(split_docs, embeddings)
    db.save_local("./faiss/product")

    return split_docs

result = main()
print(result)