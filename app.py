import streamlit as st
import os
import json
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="Generador de Exámenes AI", layout="wide")
st.title("🎓 AI Quiz Generator")

# --- PROMPT ESPECÍFICO PARA EXÁMENES ---
QUIZ_SYSTEM_PROMPT = """
You are an expert educator. Your task is to create a multiple-choice quiz based ONLY on the provided context.
Format the output as a JSON list of objects. Each object must have:
- "question": The question text.
- "options": A list of 4 possible answers.
- "answer": The index (0-3) of the correct answer.
- "explanation": A brief explanation of why that answer is correct based on the text.

Generate exactly 3 questions. Output ONLY the JSON.
"""

# --- FUNCIONES DE PROCESAMIENTO ---
def get_vectorstore(file):
    temp_path = "temp_quiz.pdf"
    with open(temp_path, "wb") as f:
        f.write(file.getvalue())
    
    loader = PyMuPDFLoader(temp_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    return FAISS.from_documents(chunks, embeddings)

# --- UI ---
with st.sidebar:
    api_key = st.text_input("Groq API Key", type="password")
    uploaded_file = st.file_uploader("Sube el material de estudio (PDF)", type="pdf")
    generate_button = st.button("✨ Generar Examen")

if api_key and uploaded_file:
    os.environ["GROQ_API_KEY"] = api_key
    
    if generate_button:
        with st.spinner("Analizando material y creando preguntas..."):
            # 1. Procesar y recuperar contexto relevante
            vs = get_vectorstore(uploaded_file)
            # Recuperamos fragmentos aleatorios o variados para cubrir el tema
            context_docs = vs.similarity_search("key concepts and definitions", k=5)
            context_text = "\n".join([d.page_content for d in context_docs])
            
            # 2. Generar con Groq
            llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.5)
            prompt = ChatPromptTemplate.from_messages([
                ("system", QUIZ_SYSTEM_PROMPT),
                ("human", "Context: {context}")
            ])
            
            chain = prompt | llm
            response = chain.invoke({"context": context_text})
            
            # Limpiar respuesta por si el LLM añade texto extra
            raw_json = response.content.strip()
            if "```json" in raw_json:
                raw_json = raw_json.split("```json")[1].split("```")[0].strip()
            
            st.session_state.quiz = json.loads(raw_json)
            st.session_state.answers = [None] * len(st.session_state.quiz)

    # 3. Mostrar el examen
    if "quiz" in st.session_state:
        st.write("---")
        for i, q in enumerate(st.session_state.quiz):
            st.subheader(f"Pregunta {i+1}: {q['question']}")
            choice = st.radio(f"Selecciona una opción para la pregunta {i}", q['options'], key=f"q_{i}", index=None)
            
            if choice:
                correct_idx = q['answer']
                if q['options'].index(choice) == correct_idx:
                    st.success(f"¡Correcto! {q['explanation']}")
                else:
                    st.error(f"Incorrecto. La respuesta era: {q['options'][correct_idx]}")
                    st.info(f"Explicación: {q['explanation']}")
