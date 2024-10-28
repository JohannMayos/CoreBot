import fitz  # PyMuPDF para leitura de PDFs
from sentence_transformers import SentenceTransformer
import chromadb
import requests
import gradio as gr

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
    "Choosing Death": "/livros/Choosing Death.pdf",
    "History of Heavy Metal": "/livros/History of Heavy Metal.pdf",
    "Louder Than Hell": "/livros/Louder Than Hell.pdf"
}

for title, path in metal_pdf_files.items():
    process_and_store_book_data(path, title)
    print(f"Processado: {title}")

def retrieve_documents_from_metal_books(query, top_k=3):
    query_embedding = embedding_model.encode(query).tolist()
    results = metal_books_collection.query(query_embeddings=[query_embedding], n_results=top_k)

    # Extrair apenas os documentos dos resultados
    docs = [doc for sublist in results['documents'] for doc in sublist]  # Garantindo que todos os documentos sejam incluídos
    return docs

# Função para gerar resposta final com LLAMA 3.1
def generate_response_with_context(query):
    # Recuperar documentos relevantes
    metal_docs = retrieve_documents_from_metal_books(query)

    # Juntar o contexto em um único bloco de texto
    context = "\n\n".join(metal_docs)

    # Exibir o contexto para diagnóstico
    print("Contexto combinado:", context)

    # Estrutura de mensagens para a API no formato esperado
    messages = [
        {"role": "system", "content": "Você é um assistente especializado em música metal."},
        {"role": "user", "content": f"Contexto:\n{context}\n\nPergunta: {query}"}
    ]

    # Enviar para LLAMA 3.1 via API (LM Studio ou outro endpoint)
    response = requests.post(
        "http://localhost:8000/v1/chat/completions",
        json={"messages": messages, "max_tokens": 300}
    )

    # Checar se a resposta foi recebida corretamente
    if response.status_code == 200:
        generated_text = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        print("Resposta gerada:", generated_text)  # Diagnóstico da resposta gerada
        return generated_text
    else:
        print("Erro na geração da resposta:", response.status_code, response.text)
        return "Erro na geração da resposta."

# Interface Gradio para consulta
def process_query(query):
    response = generate_response_with_context(query)
    return response

# Configuração da Interface Gradio
interface = gr.Interface(
    fn=process_query,
    inputs="text",
    outputs="text",
    title="Consultor de Música Metal",
    description="Consulte sobre a história do metal, subgêneros e bandas."
)

# Executar a interface
interface.launch()
