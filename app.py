from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import mysql.connector
import logging
import traceback
import io
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app_debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

@app.route('/')
def serve_html():
    return send_from_directory('.', 'index.html')

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('static', filename)

# Database Configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Praveen@6360',
    'database': 'QPGen'
}

# Global variables
llm = None
prompt_template = None
embeddings = None
text_splitter = None
chain = None

def extract_pdf_text(pdf_blob):
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_blob))
        full_text = ""
        for page in pdf_reader.pages:
            full_text += page.extract_text() + "\n"
        return ' '.join(full_text.split())
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        logger.error(traceback.format_exc())
        return ""

def initialize_components():
    global llm, prompt_template, embeddings, text_splitter, chain
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            length_function=len
        )

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

        llm = CTransformers(
            model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
            model_type="llama",
            max_new_tokens=128,
            temperature=0.2,
            context_length=512,
            batch_size=8
        )

        prompt_template = PromptTemplate(
            input_variables=["context", "question_type", "difficulty"],
            template=(
                "Generate {question_type} questions at {difficulty} level based on:\n"
                "{context}\n\n"
                "Create more than 10 questions that test understanding of key concepts. "
                "Generate questions only, do not give the solution along with the question and exclude giving information in brackets."
            )
        )

        chain = prompt_template | llm

        logger.info("All components initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Initialization error: {e}")
        logger.error(traceback.format_exc())
        return False

def get_db_connection():
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except mysql.connector.Error as err:
        logger.error(f"Database connection error: {err}")
        raise

def process_context(context_text):
    words = context_text.split()
    return " ".join(words[:200]) if len(words) > 200 else context_text

@app.route('/generate_questions', methods=['POST'])
def generate_questions():
    try:
        if not components_loaded:
            return jsonify({"error": "Components not initialized"}), 500

        # Support JSON and multipart form
        if request.content_type.startswith('multipart/form-data'):
            data = request.form
            is_custom_subject = data.get("isCustomSubject") == 'true'
            file = request.files.get("pdfFile")
        else:
            data = request.get_json()
            is_custom_subject = data.get("isCustomSubject") == 'true'
            file = None

        required_fields = ['subjectCode', 'questionType', 'numModules', 'difficultyLevel']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        subject_code = data["subjectCode"]
        question_type = data["questionType"]
        difficulty_level = data["difficultyLevel"]
        num_modules = int(data["numModules"])

        all_chunks = []

        if is_custom_subject:
            if not file or not file.filename.endswith('.pdf'):
                return jsonify({"error": "A valid PDF file is required"}), 400
            module_text = extract_pdf_text(file.read())
            all_chunks.extend(text_splitter.split_text(module_text))
        else:
            conn = get_db_connection()
            cursor = conn.cursor(dictionary=True)
            try:
                cursor.execute(
                    "SELECT module_1, module_2, module_3, module_4, module_5 FROM SubjectPdfs WHERE subject_code = %s",
                    (subject_code,)
                )
                result = cursor.fetchone()
                if not result:
                    return jsonify({"error": "No syllabus found"}), 404

                for i in range(1, num_modules + 1):
                    module_pdf = result.get(f"module_{i}")
                    if module_pdf:
                        module_text = extract_pdf_text(module_pdf)
                        all_chunks.extend(text_splitter.split_text(module_text))
            finally:
                cursor.close()
                conn.close()

        if not all_chunks:
            return jsonify({"error": "No content extracted"}), 400

        vector_store = FAISS.from_texts(all_chunks, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 1})
        relevant_docs = retriever.invoke(f"Generate {question_type} questions")
        context_text = process_context(relevant_docs[0].page_content)

        response = chain.invoke({
            "context": context_text,
            "question_type": question_type,
            "difficulty": difficulty_level
        })

        response_text = response.get("text", "") if isinstance(response, dict) else str(response)
        questions = [
            q.strip()
            for q in response_text.split('\n')
            if q.strip() and not q.lower().startswith(('answer', 'solution'))
        ]

        return jsonify({"questions": questions})

    except Exception as e:
        logger.error(f"Error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    components_loaded = initialize_components()
    app.run(debug=True, port=5000)
    
