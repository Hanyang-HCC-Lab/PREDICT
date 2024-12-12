from langchain_community.vectorstores import FAISS
from langchain import hub
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
import threading

# 전역 변수로 FAISS 벡터스토어 인스턴스 초기화
vectorstore_instance = None

# model_name = 'jhgan/ko-sroberta-multitask'

def init_vectorstore(dataset_name):
    global vectorstore_instance
    vectorstore_instance = FAISS.load_local(f"./faiss/{dataset_name}_faiss_index_constitution", OpenAIEmbeddings())

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def rag_chain_invoke(sentence, retriever, prompt, llm):
    rag_chain = (
        {"context": retriever | format_docs, "text": RunnablePassthrough()}
        | prompt
        | llm
        | JsonOutputParser()
    )
    response = rag_chain.invoke(sentence)
    return response



def rag_chain_invoke_with_timeout(sentence, retriever, prompt, llm, timeout):
    result = [None]
    exception = [None]

    def target():
        try:
            result[0] = rag_chain_invoke(sentence, retriever, prompt, llm)
        except Exception as e:
            exception[0] = e

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise TimeoutError("Operation timed out")
    if exception[0]:
        raise exception[0]
    return result[0]


def RAG(sentence, agent_name):
    global vectorstore_instance
    if vectorstore_instance is None:
        raise Exception("Vectorstore not initialized. Call init_vectorstore() first.")
    
    # Step 4: Search
    retriever = vectorstore_instance.as_retriever()
    
    # Step 5: Create Prompt
    prompt = hub.pull(agent_name)
    
    # Create LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0)
    
    # 타임아웃 시간 (초)
    TIMEOUT_SECONDS = 30

    while True:
        try:
            response = rag_chain_invoke_with_timeout(sentence, retriever, prompt, llm, TIMEOUT_SECONDS)
            return response
        except TimeoutError:
            print("Operation timed out. Retrying...")
        except Exception as e:
            print(f"An error occurred: {e}")
            break


