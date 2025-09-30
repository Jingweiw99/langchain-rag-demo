from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from get_vector import CustomEmbeddings
from embedding_model import EmbeddingModel
from model import ChatModel
import os
import config

class QASystem:
    def __init__(self, vector_db_path: str = "./faiss/knowledge", 
                 embedding_model_path: str = None,
                 llm_model_path: str = None):
        """
        初始化QA系统
        :param vector_db_path: 向量数据库路径
        :param embedding_model_path: 嵌入模型路径
        :param llm_model_path: LLM模型路径
        """
        self.vector_db_path = vector_db_path
        self.embedding_model_path = embedding_model_path
        self.llm_model_path = llm_model_path
        self.embeddings = None
        self.db = None
        self.llm = None
        
    def load_models(self):
        """加载所有模型"""
        print("=" * 50)
        print("正在初始化QA系统...")
        print("=" * 50)
        
        # 加载嵌入模型
        print("\n1. 加载嵌入模型...")
        embedding_model = EmbeddingModel(model_path=self.embedding_model_path)
        embedding_model.load_model()
        self.embeddings = CustomEmbeddings(embedding_model)
        
        # 加载向量数据库
        print("\n2. 加载向量数据库...")
        if not os.path.exists(self.vector_db_path):
            raise FileNotFoundError(
                f"向量数据库不存在: {self.vector_db_path}\n"
                "请先运行 get_vector.py 创建向量数据库"
            )
        self.db = FAISS.load_local(self.vector_db_path, self.embeddings, allow_dangerous_deserialization=True)
        print("向量数据库加载完成！")
        
        # 加载对话模型
        print("\n3. 加载对话模型...")
        self.llm = ChatModel(model_path=self.llm_model_path)
        self.llm.load_model()
        
        print("\n" + "=" * 50)
        print("QA系统初始化完成！")
        print("=" * 50 + "\n")
    
    def get_related_content(self, related_docs, max_docs: int = 3):
        """
        获取相关文档内容
        :param related_docs: 相关文档列表
        :param max_docs: 最多返回的文档数
        :return: 拼接后的文档内容
        """
        related_content = []
        for doc in related_docs[:max_docs]:
            content = doc.page_content.replace("\n\n", "\n")
            related_content.append(content)
        return "\n\n".join(related_content)
    
    def create_prompt(self, question: str, top_k: int = None):
        """
        创建提示词
        :param question: 用户问题
        :param top_k: 检索的文档数量（默认使用配置文件中的值）
        :return: 完整的提示词
        """
        if top_k is None:
            top_k = config.TOP_K
        
        # 从向量数据库中检索相关文档
        docs = self.db.similarity_search(question, k=top_k)
        related_content = self.get_related_content(docs, max_docs=top_k)
        
        # 定义提示词模板
        PROMPT_TEMPLATE = """你是一个专业的问答助手。请基于以下已知信息，直接给出答案，不要解释推理过程。

已知信息:
{context}

用户问题:
{question}

请注意：
1. 如果已知信息中包含答案，请直接基于这些信息回答
2. 如果已知信息不足以回答问题，请明确告知用户

你的回答:"""

        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=PROMPT_TEMPLATE
        )
        
        return prompt.format(context=related_content, question=question)
    def ask(self, question: str, top_k: int = None) -> str:
        """
        问答
        :param question: 用户问题
        :param top_k: 检索的文档数量（默认使用配置文件中的值）
        :return: 答案
        """
        if self.llm is None:
            raise ValueError("模型未加载，请先调用load_models()方法")
        
        if top_k is None:
            top_k = config.TOP_K
        
        # 创建提示词 , top_k 这个模型中top_k不可用
        prompt = self.create_prompt(question, top_k)
        
        # 获取答案
        answer = self.llm.chat(prompt, use_history=False)
        
        return answer
    
    def interactive_mode(self):
        """交互式问答模式"""
        print("\n进入交互式问答模式")
        print("输入 'quit' 或 'exit' 退出")
        print("输入 'clear' 清空对话历史")
        print("-" * 50)
        
        while True:
            question = input("\n请输入您的问题: ").strip()
            
            if question.lower() in ['quit', 'exit', '退出']:
                print("感谢使用，再见！")
                break
            
            if question.lower() in ['clear', '清空']:
                self.llm.clear_history()
                print("对话历史已清空")
                continue
            
            if not question:
                print("问题不能为空，请重新输入")
                continue
            
            print("\n正在思考...")
            try:
                answer = self.ask(question)
                print(f"\n回答: {answer}")
            except Exception as e:
                print(f"发生错误: {str(e)}")

def main():
    """主函数"""
    # 使用配置文件中的设置创建QA系统
    qa_system = QASystem(
        vector_db_path=config.VECTOR_DB_PATH,
        embedding_model_path=config.EMBEDDING_MODEL_PATH,
        llm_model_path=config.LLM_MODEL_PATH
    )
    
    # 加载模型
    qa_system.load_models()
    
    # 单次问答示例
    print("=" * 50)
    print("单次问答示例")
    question = "身高178，体重68，适合哪个尺码？"
    print(f"\n问题: {question}")
    answer = qa_system.ask(question)
    print(f"回答: {answer}")
    print("=" * 50)
    
    # 进入交互式模式
    qa_system.interactive_mode()

if __name__ == '__main__':
    main()