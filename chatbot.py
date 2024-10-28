import fitz  # PyMuPDF para leitura de PDFs
from sentence_transformers import SentenceTransformer
import chromadb
import requests
import os
from groq import Groq
import json

# Carregar o modelo de embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Conectar ao ChromaDB e criar coleções para livros de metal
client = chromadb.Client()
metal_books_collection = client.get_or_create_collection("livros_metal")

# Função para extrair texto dos PDFs
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

# Função para processar e armazenar textos dos livros no ChromaDB
def process_and_store_book_data(pdf_path, title):
    text = extract_text_from_pdf(pdf_path)
    paragraphs = text.split("\n\n")  # Dividir em parágrafos para melhorar a precisão de recuperação
    
    for i, paragraph in enumerate(paragraphs):
        # Gerar embedding para o parágrafo
        embedding = embedding_model.encode(paragraph).tolist()

        # Armazenar no ChromaDB
        metal_books_collection.add(
            documents=[paragraph],
            metadatas=[{"title": title, "paragraph_id": i}],
            ids=[f"{title}_{i}"],
            embeddings=[embedding]
        )

# Processar todos os livros em PDF sobre metal
metal_pdf_files = {
    "Choosing Death": "livros/ChoosingDeath.pdf",
    "History of Heavy Metal": "livros/HistoryofHeavyMetal.pdf",
    "Louder Than Hell": "livros/LouderThanHell.pdf"
}

for title, path in metal_pdf_files.items():
    process_and_store_book_data(path, title)
    print(f"Processado: {title}")

def retrieve_documents_from_metal_books(query, top_k=1): 
    query_embedding = embedding_model.encode(query).tolist()
    results = metal_books_collection.query(query_embeddings=[query_embedding], n_results=top_k)

    # Extrair apenas os documentos dos resultados
    docs = [doc for sublist in results['documents'] for doc in sublist]
    return docs

working_dir = os.path.dirname(os.path.abspath(__file__))
config_data = json.load(open(f"{working_dir}/config.json"))

GROQ_API_KEY = config_data["GROQ_API_KEY"]

# Save the API key to environment variable
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

client = Groq()

def generate_response_with_context(query):
    # Recuperar documentos relevantes
    metal_docs = retrieve_documents_from_metal_books(query, top_k=1)

    # Juntar o contexto em um único bloco de texto, talvez resumindo ou pegando partes específicas
    context = " ".join(metal_docs)  # Pode-se usar apenas o primeiro documento ou um resumo

    # Limitar o tamanho do contexto (por exemplo, limitar a 512 tokens)
    context_tokens = context.split()[:512]  # Limitar para 512 tokens
    context = " ".join(context_tokens)

    # Exibir o contexto para diagnóstico
    print("Contexto combinado:", context)

    # Estrutura de mensagens para a API no formato esperado
    messages = [
        {"role": "system", "content": "Você é um assistente especializado em música metal."},
        {"role": "user", "content": f"Contexto:\n{context}\n\nPergunta: {query}"}
    ]

    # Criar uma solicitação de conclusão de chat usando o modelo Groq
    chat_completion = client.chat.completions.create(
        messages=messages,
        model="llama3-70b-8192",
    )

    # Recuperar e retornar o conteúdo gerado pela resposta
    return chat_completion.choices[0].message.content


