import streamlit as st
import os
import json
import tempfile
import pytesseract
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image

# --- IMPORTACIONES CORREGIDAS ---
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
# ✅ ESTA ES LA LÍNEA QUE DABA ERROR (SOLUCIONADA):
from langchain_core.documents import Document 

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Generador de Exámenes con IA",
    page_icon="📝",
    layout="centered"
)

st.title("📝 Generador de Exámenes a Medida")
st.markdown("""
Sube tus apuntes o libros (PDF) y elige cuántas preguntas quieres para practicar.
""")

# --- FUNCIONES AUXILIARES (OCR y Limpieza) ---

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
        st.error(f"Error en OCR: {e}. Asegúrate de tener instalado 'poppler-utils' y 'tesseract-ocr' en packages.txt.")
        return ""

def process_document(uploaded_file):
    """Procesa el PDF, extrae texto (normal u OCR) y crea el VectorStore."""
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        loader = PyMuPDFLoader(tmp_path)
        docs = loader.load()
        text_content = "\n".join([doc.page_content for doc in docs])

        # Detectar si es escaneado (poco texto)
        if len(text_content.strip()) < 50:
            extracted_text = perform_ocr(tmp_path)
            if not extracted_text.strip():
                return None, "No se pudo extraer texto ni con OCR."
            docs = [Document(page_content=extracted_text, metadata={"source": uploaded_file.name})]

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(docs)

        if not chunks:
            return None, "El documento está vacío después del procesamiento."

        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        
        return vectorstore, None

    except Exception as e:
        return None, str(e)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# --- BARRA LATERAL (CONFIGURACIÓN) ---
with st.sidebar:
    st.header("⚙️ Configuración")
    api_key = st.text_input("Groq API Key", type="password")
    
    st.divider()
    
    uploaded_file = st.file_uploader("Sube tu material (PDF)", type="pdf")
    
    st.subheader("Parámetros del Examen")
    
    # Input numérico para elegir la cantidad exacta
    num_questions = st.number_input(
        "Número de preguntas:", 
        min_value=1, 
        max_value=20, 
        value=5, 
        step=1,
        help="Elige cuántas preguntas quieres generar (máx 20 por bloque)."
    )
    
    level = st.selectbox("Nivel de dificultad", ["Fácil", "Intermedio", "Difícil"])
    
    generate_btn = st.button("Generar Examen", type="primary")

# --- PROMPT REFORZADO ---
QUIZ_SYSTEM_PROMPT = """
You are an expert teacher creating a multiple-choice quiz based ONLY on the provided context.

STRICT REQUIREMENTS:
1. Difficulty Level: {level}
2. **Quantity: Generate EXACTLY {num_questions} questions.** Do not generate fewer or more.
3. Language: Spanish (Respond in Spanish).

OUTPUT FORMAT (JSON ONLY):
Return a raw JSON list of objects. No markdown, no explanations outside the JSON.
Schema:
[
   {{
       "question": "Texto de la pregunta",
       "options": ["Opción A", "Opción B", "Opción C", "Opción D"],
       "answer": 0,  // Índice de la respuesta correcta (0-3)
       "explanation": "Breve explicación de por qué es correcta según el texto."
   }}
]
"""

# --- LÓGICA PRINCIPAL ---

if generate_btn:
    if not api_key:
        st.warning("⚠️ Por favor, introduce tu API Key de Groq.")
    elif not uploaded_file:
        st.warning("⚠️ Por favor, sube un archivo PDF.")
    else:
        os.environ["GROQ_API_KEY"] = api_key
        
        # 1. Procesar documento (solo si es nuevo)
        if "vectorstore" not in st.session_state or st.session_state.get("current_file") != uploaded_file.name:
            with st.spinner("🔍 Procesando documento..."):
                vectorstore, error = process_document(uploaded_file)
                if error:
                    st.error(f"Error: {error}")
                else:
                    st.session_state.vectorstore = vectorstore
                    st.session_state.current_file = uploaded_file.name

        # 2. Generar Preguntas
        if "vectorstore" in st.session_state:
            with st.spinner(f"🤖 Generando {num_questions} preguntas de nivel {level}..."):
                try:
                    # Recuperamos más contexto si se piden muchas preguntas
                    # k = número de preguntas * 1.5 (para asegurar suficiente info)
                    k_retrieval = max(5, int(num_questions * 1.5))
                    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": k_retrieval})
                    
                    # Recuperar contexto
                    docs = retriever.invoke("conceptos clave definiciones importantes resumen del tema")
                    context_text = "\n\n".join([d.page_content for d in docs])

                    # Llamar al LLM
                    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.5)
                    
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", QUIZ_SYSTEM_PROMPT),
                        ("human", "Contexto: {context}")
                    ])
                    
                    chain = prompt | llm
                    response = chain.invoke({
                        "context": context_text,
                        "level": level,
                        "num_questions": num_questions
                    })
                    
                    # Procesar JSON
                    json_content = clean_json_string(response.content)
                    quiz_data = json.loads(json_content)
                    
                    # Validar cantidad (informativo)
                    if len(quiz_data) != num_questions:
                        st.warning(f"Nota: Se solicitaron {num_questions} preguntas, pero la IA generó {len(quiz_data)}.")
                    
                    # Guardar estado
                    st.session_state.quiz_data = quiz_data
                    # Reiniciar respuestas del usuario
                    st.session_state.user_answers = {} 
                    st.session_state.quiz_submitted = False
                    
                except Exception as e:
                    st.error(f"Error generando el examen: {e}")

# --- MOSTRAR EXAMEN ---
if "quiz_data" in st.session_state and st.session_state.quiz_data:
    st.write("---")
    st.subheader(f"📝 Examen ({len(st.session_state.quiz_data)} preguntas)")
    
    with st.form("quiz_form"):
        for i, q in enumerate(st.session_state.quiz_data):
            st.markdown(f"#### {i+1}. {q['question']}")
            
            # Recuperar respuesta previa si existe
            current_choice = st.session_state.user_answers.get(i, None)
            
            choice = st.radio(
                "Elige una opción:", 
                q['options'], 
                key=f"radio_{i}", 
                index=None if current_choice is None else q['options'].index(current_choice) if current_choice in q['options'] else None,
                label_visibility="collapsed"
            )
            st.write("") 
            
        submit = st.form_submit_button("Corregir Examen")
        
        if submit:
            st.session_state.quiz_submitted = True
            # Guardar las respuestas actuales del form en el estado
            for i in range(len(st.session_state.quiz_data)):
                st.session_state.user_answers[i] = st.session_state[f"radio_{i}"]

    # --- RESULTADOS ---
    if st.session_state.get("quiz_submitted", False):
        st.write("---")
        st.subheader("Resultados")
        
        correct_count = 0
        for i, q in enumerate(st.session_state.quiz_data):
            user_choice = st.session_state.user_answers.get(i)
            correct_option = q['options'][q['answer']]
            
            with st.expander(f"Pregunta {i+1}: {'✅ Correcta' if user_choice == correct_option else '❌ Incorrecta'}", expanded=True):
                if user_choice == correct_option:
                    correct_count += 1
                    st.success(f"¡Bien hecho! La respuesta es: {correct_option}")
                else:
                    st.error(f"Tu respuesta: {user_choice}")
                    st.success(f"Respuesta correcta: {correct_option}")
                
                st.info(f"💡 Explicación: {q['explanation']}")

        if len(st.session_state.quiz_data) > 0:
            score = (correct_count / len(st.session_state.quiz_data)) * 100
            st.metric(label="Calificación Final", value=f"{score:.0f}/100")
            
            if score == 100:
                st.balloons()
