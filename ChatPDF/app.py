import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import tempfile
from main import DocumentLoader, EmbeddingModel, VectorStore
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough

from main import VectorStore
from main import DocumentLoader
from main import EmbeddingModel
from main import llm_model
from main import load_config

st.set_page_config(page_title="PDF 챗봇", layout="wide")

st.title("📚 PDF 챗봇")
st.write("PDF 파일을 업로드하고 질문해보세요!")

config = load_config("config.yaml")

# 세션 상태 초기화
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# PDF 파일 업로드
uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type=['pdf'])

if uploaded_file is not None:
    # 임시 파일로 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    # PDF 처리
    with st.spinner('PDF를 처리하는 중입니다...'):
        try:
            loader = DocumentLoader(tmp_file_path)
            hf_embeddings = EmbeddingModel()
            st.session_state.vector_store = VectorStore(loader.splits, hf_embeddings.embed_model())
            st.success('PDF 처리가 완료되었습니다!')
        except Exception as e:
            st.error(f'PDF 처리 중 오류가 발생했습니다: {str(e)}')
    
    # 임시 파일 삭제
    os.unlink(tmp_file_path)

# 챗봇 인터페이스
if st.session_state.vector_store is not None:
    # RAG 체인 설정
    template = config["custom_template"]
    rag_prompt_custom = PromptTemplate.from_template(template=template)
    
    llm_model = llm_model()
    retriever = st.session_state.vector_store.database.as_retriever(embeddings=OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY")))
    rag_chain = {"context": retriever, "question": RunnablePassthrough()} | rag_prompt_custom | llm_model

    # 채팅 히스토리 표시
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # 사용자 입력
    if prompt := st.chat_input("질문을 입력하세요"):
        # 사용자 메시지 추가
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # 챗봇 응답
        with st.chat_message("assistant"):
            with st.spinner('답변을 생성하는 중입니다...'):
                response = rag_chain.invoke(prompt)
                st.write(response.content)
                st.session_state.chat_history.append({"role": "assistant", "content": response.content})
else:
    st.info('PDF 파일을 업로드해주세요.')
