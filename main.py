# 导入必备的工具包ChatGLM2
from langchain import PromptTemplate
from get_vector import *
from model import ChatModel
# 加载FAISS向量库
EMBEDDING_MODEL = "moka-ai/m3e-base"
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
db = FAISS.load_local("faiss/product",embeddings)


def get_related_content(related_docs):
    related_content = []
    for doc in related_docs:
        related_content.append(doc.page_content.replace("\n\n", "\n"))
    return "\n".join(related_content)

def define_prompt():
    question = '我身高170，体重140斤,买多大尺码'
    docs = db.similarity_search(question, k=1)
    related_content = get_related_content(docs)

    PROMPT_TEMPLATE = """
        基于以下已知信息，简洁和专业的来回答用户的问题。不允许在答案中添加编造成分。
        已知内容:
        {context}
        问题:
        {question}"""

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=PROMPT_TEMPLATE,)

    my_pmt = prompt.format(context=related_content,
                        question=question)

    return my_pmt

def qa():
    llm = ChatModel()
    llm.load_model("/Users/**/PycharmProjects/llm/ChatGLM-6B/THUDM/chatglm-6b")
    my_pmt = define_prompt()
    result = llm(my_pmt)
    return result



if __name__ == '__main__':
    result = qa()
    print(result)