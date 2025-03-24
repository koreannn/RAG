import os
from dotenv import load_dotenv
load_dotenv()
import yaml

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough

import yaml

def load_config(path):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config

class EmbeddingModel:
    def __init__(self):
        self.model_name = "jhgan/ko-sroberta-multitask"
        self.model_kwargs = {'device': 'mps'}
        self.encode_kwargs = {'normalize_embeddings': False}
        
    def embed_model(self):
        return HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs=self.model_kwargs,
            encode_kwargs=self.encode_kwargs
        )
    
    def embed_documents(self, documents):
        texts = [doc.page_content for doc in documents]
        return self.embed_model().embed_documents(texts)


class VectorStore:
    def __init__(self, documents, hf_embeddings):
        self.database = Chroma.from_documents(documents, hf_embeddings)
    
    def get_similarity_search(self, query, k=3) -> list[tuple[Document, float]]: 
        return self.database.similarity_search_with_score(query, k=k) 


        
class DocumentLoader:
    def __init__(self, file_path):
        self.loader = PyPDFLoader(file_path)
        self.pages = self.loader.load_and_split()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        self.splits = self.text_splitter.split_documents(self.pages)
        

class llm_model:
    def __init__(self, model_name="gpt-4o-mini", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY")):
        self.model = ChatOpenAI(model=model_name, temperature=temperature, openai_api_key=openai_api_key)

    def predict(self, prompt):
        return self.model.invoke(prompt)


if __name__ == "__main__":
    
    loader = DocumentLoader("https://snuac.snu.ac.kr/2015_snuac/wp-content/uploads/2015/07/asiabrief_3-26.pdf")
    hf_embeddings = EmbeddingModel()
    embeddings = hf_embeddings.embed_documents(loader.splits)
    config = load_config("config.yaml")
    
    vector_store = VectorStore(loader.splits, hf_embeddings.embed_model())
    print(vector_store.get_similarity_search("한국 저출산의 원인")) # LLM없이 검색
    
    template = config["custom_template"]
    rag_prompt_custom = PromptTemplate.from_template(template=template)
    
    
    # llm_model = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
    # retriever = vector_store.database.as_retriever(embeddings=OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY")))
    llm_model = llm_model(model_name="gpt-4o-mini", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
    retriever = vector_store.database.as_retriever(embeddings=OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY")))
    rag_chain = {"context" : retriever, "question" : RunnablePassthrough()} | rag_prompt_custom | llm_model
    print(rag_chain.invoke("한국 저출산의 원인")) # LLM 추가 검색

    # print(llm_model.predict("한국 저출산의 원인")) # RAG없이 모델 순수 답변
    
    