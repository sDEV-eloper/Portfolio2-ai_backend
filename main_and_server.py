from flask import Flask, request, jsonify
from flask_cors import CORS
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os

app = Flask(__name__)
CORS(app)

# Use environment variables for API keys
os.environ["OPENAI_API_KEY"] = "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"

# Make PDF file path configurable
PDF_FILE_PATH = './myresume.pdf'

@app.route('/api/home', methods=['POST'])
def receive_data():
    try:
        data = request.get_json()
        input_text = data.get('inputData')
        trained_text = modify_input(input_text)
        return jsonify({'inputData': trained_text})

    except Exception as e:
        return jsonify({'error': str(e)})

def parse_pdf(file_path):
    pdf_reader = PdfReader(file_path)
    raw_text = ''
    for i, page in enumerate(pdf_reader.pages):
        content = page.extract_text()
        if content:
            raw_text += content
    return raw_text

def split_text(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
    )
    return text_splitter.split_text(raw_text)

def create_embeddings(texts):
    embeddings = OpenAIEmbeddings()
    return embeddings, FAISS.from_texts(texts, embeddings)

def load_question_answering_chain():
    chain = load_qa_chain(OpenAI(), chain_type="stuff")
    return chain

def modify_input(input_text):
    query = input_text
    raw_text = parse_pdf(PDF_FILE_PATH)
    texts = split_text(raw_text)
    embeddings, document_search = create_embeddings(texts)
    chain = load_question_answering_chain()
    docs = document_search.similarity_search(query)
    return chain.run(input_documents=docs, question=query)

if __name__ == '__main__':
    app.run(debug=True, port=8080)
