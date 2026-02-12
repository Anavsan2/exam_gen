import streamlit as st
import os
import json
import tempfile
import pytesseract
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image

# LangChain & AI
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.docstore.document import Document

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Generador de Exámenes con IA (RAG + OCR)",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Generador de Exámenes Inteligente")
st.markdown("""
Sube un PDF (texto o escaneado) y la IA generará un examen tipo test basado en su contenido.
""")

# --- FUNCIONES AUXILIARES ---

def clean_json_string(json_str):
    """Limpia la respuesta del LLM para asegurar que es un JSON válido."""
    json_str = json_str.strip()
    if "```json" in json_str:
        json_str = json_str.split("```json")[1]
    if "```" in json_str:
        json_str = json_str.split("```")[0]
    return json_str

def perform_ocr(pdf_path):
    """Realiza OCR en un PDF escaneado."""
    st.info("⚠️ Detectando imágenes... Aplicando OCR (esto puede tardar unos segundos).")
    try:
        images = convert_from_path(pdf_path)
        text = ""
        progress_bar = st.progress(0)
        
        for i, image in enumerate(images):
            text += pytesseract.image_to_string(image) + "\n"
            progress_bar.progress((i + 1) / len(images))
        
        progress_bar.empty()
        return text
    except Exception as e:
        st.error(f"Error en OCR: {e}. Asegúrate de tener instalado 'poppler-utils' y 'tesseract-ocr'.")
        return ""

def process_document(uploaded_file):
    """Procesa el PDF, extrae texto (normal u OCR) y crea el VectorStore."""
    
    # Crear archivo temporal
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        # 1. Intentar extracción de texto nativo
        loader = PyMuPDFLoader(tmp_path)
        docs = loader.load()
        text_content = "\n".join([doc.page_content for doc in docs])

        # 2. Si hay muy poco texto, asumir que es escaneado y usar OCR
        if len(text_content.strip()) < 50:
            extracted_text = perform_ocr(tmp_path)
            if not extracted_text.strip():
                return None, "No se pudo extraer texto ni con OCR."
            docs = [Document(page_content=extracted_text, metadata={"source": uploaded_file.name})]

        # 3. Dividir texto en chunks (fragmentos)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(docs)

        if not chunks:
            return None, "El documento está vacío después del procesamiento."

        # 4. Crear Embeddings y Vector Store
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        
        return vectorstore, None

    except Exception as e:
        return None, str(e)
    finally:
        # Limpieza
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuración")
    api_key = st.text_input("Groq API Key", type="password")
    
    st.write("---")
    uploaded_file = st.file_uploader("Sube tu material (PDF)", type="pdf")
    num_questions = st.slider("Número de preguntas", 1, 10, 3)
    level = st.selectbox("Nivel de dificultad", ["Fácil", "Intermedio", "Difícil"])
    
    generate_btn = st.button("Generar Examen", type="primary")

# --- PROMPT DEL SISTEMA ---
QUIZ_SYSTEM_PROMPT = """
You are an expert teacher creating a quiz based ONLY on the provided context.
Difficulty Level: {level}
Number of Questions: {num_questions}

Output Requirements:
1. Return ONLY a JSON list of objects. Do not include markdown formatting or conversational text.
2. Each object must strictly follow this schema:
   {{
       "question": "Question text here",
       "options": ["Option A", "Option B", "Option C", "Option D"],
       "answer": 0,  // Index of the correct option (0-3)
       "explanation": "Brief explanation of why this is correct."
   }}
"""

# --- LÓGICA PRINCIPAL ---

if generate_btn:
    if not api_key:
        st.warning("⚠️ Por favor, introduce tu API Key de Groq.")
    elif not uploaded_file:
        st.warning("⚠️ Por favor, sube un archivo PDF.")
    else:
        os.environ["GROQ_API_KEY"] = api_key
        
        # 1. Procesamiento (solo si no se ha hecho ya para este archivo)
        if "vectorstore" not in st.session_state or st.session_state.get("current_file") != uploaded_file.name:
            with st.spinner("🔍 Analizando documento e indexando..."):
                vectorstore, error = process_document(uploaded_file)
                
                if error:
                    st.error(f"Error: {error}")
                else:
                    st.session_state.vectorstore = vectorstore
                    st.session_state.current_file = uploaded_file.name
                    st.success("Documento procesado correctamente.")

        # 2. Generación del Quiz
        if "vectorstore" in st.session_state:
            with st.spinner("🤖 La IA está redactando las preguntas..."):
                try:
                    # Recuperar contexto relevante (Retrieve)
                    # Buscamos conceptos generales para tener una visión amplia
                    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})
                    docs = retriever.invoke("summary of key concepts main topics definitions")
                    context_text = "\n\n".join([d.page_content for d in docs])

                    # Generar con LLM (Generate)
                    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.5)
                    
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", QUIZ_SYSTEM_PROMPT),
                        ("human", "Context: {context}")
                    ])
                    
                    chain = prompt | llm
                    response = chain.invoke({
                        "context": context_text,
                        "level": level,
                        "num_questions": num_questions
                    })
                    
                    # Limpieza y Parseo del JSON
                    json_content = clean_json_string(response.content)
                    quiz_data = json.loads(json_content)
                    
                    # Guardar en sesión
                    st.session_state.quiz_data = quiz_data
                    # Reiniciar respuestas del usuario
                    st.session_state.user_answers = [None] * len(quiz_data)
                    
                except Exception as e:
                    st.error(f"Ocurrió un error generando el examen: {e}")

# --- MOSTRAR EXAMEN ---
if "quiz_data" in st.session_state:
    st.write("---")
    st.subheader(f"📝 Examen: {uploaded_file.name}")
    
    score = 0
    all_answered = True
    
    # Crear formulario para evitar recargas constantes
    with st.form("quiz_form"):
        for i, q in enumerate(st.session_state.quiz_data):
            st.markdown(f"**{i+1}. {q['question']}**")
            
            # Radio buttons para las opciones
            # Usamos un key único para mantener el estado
            choice = st.radio(
                "Selecciona una opción:", 
                q['options'], 
                key=f"q_{i}", 
                index=None,
                label_visibility="collapsed"
            )
            
            st.write("") # Espacio
            
        submit = st.form_submit_button("Verificar Resultados")

    # Lógica de corrección (fuera del form para mostrar resultados)
    if submit:
        st.write("---")
        st.subheader("Resultados")
        
        correct_count = 0
        for i, q in enumerate(st.session_state.quiz_data):
            user_choice = st.session_state.get(f"q_{i}")
            correct_option = q['options'][q['answer']]
            
            if user_choice == correct_option:
                correct_count += 1
                st.success(f"✅ Pregunta {i+1}: Correcta")
            else:
                st.error(f"❌ Pregunta {i+1}: Incorrecta")
                st.markdown(f"**Tu respuesta:** {user_choice}")
                st.markdown(f"**Respuesta correcta:** {correct_option}")
                st.info(f"ℹ️ **Explicación:** {q['explanation']}")
            st.write("---")

        score_pct = (correct_count / len(st.session_state.quiz_data)) * 100
        if score_pct >= 70:
            st.balloons()
            st.success(f"🎉 ¡Felicidades! Puntuación final: {score_pct:.1f}% ({correct_count}/{len(st.session_state.quiz_data)})")
        else:
            st.warning(f"📚 A seguir estudiando. Puntuación final: {score_pct:.1f}% ({correct_count}/{len(st.session_state.quiz_data)})")
