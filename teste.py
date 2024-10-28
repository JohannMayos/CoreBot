import os
import json
import random
import streamlit as st
from groq import Groq
from whoosh import index, scoring
from whoosh.fields import Schema, TEXT
from whoosh.qparser import QueryParser

# Função para indexar documentos
def create_index(documents):
    schema = Schema(title=TEXT(stored=True), path=TEXT(stored=True))
    if not os.path.exists("indexdir"):
        os.mkdir("indexdir")
    ix = index.create_in("indexdir", schema)
    writer = ix.writer()
    
    for doc in documents:
        writer.add_document(title=doc.split(": ")[0], path=doc.split(": ")[1])
    writer.commit()

# Função para recuperar documentos
def retrieve_documents(query, num_docs=3):
    ix = index.open_dir("indexdir")
    with ix.searcher(weighting=scoring.Frequency) as searcher:
        query_parser = QueryParser("title", ix.schema)
        query = query_parser.parse(query)
        results = searcher.search(query, limit=num_docs)
        return [hit['path'] for hit in results]

# Função para ler o conteúdo de um PDF
def read_pdf(file_path):
    import PyPDF2
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text.strip()

# Indexando os documentos inicialmente
documents = [
    "Choosing Death: livros/ChoosingDeath.pdf",
    "History of Heavy Metal: livros/HistoryofHeavyMetal.pdf",
    "Louder Than Hell: livros/LouderThanHell.pdf"
]
create_index(documents)

# Streamlit page configuration
st.set_page_config(
    page_title="LLAMA 3.1. Chat",
    page_icon="🦙",
    layout="centered"
)

working_dir = os.path.dirname(os.path.abspath(__file__))
config_data = json.load(open(f"{working_dir}/config.json"))

GROQ_API_KEY = config_data["GROQ_API_KEY"]

# Save the API key to environment variable
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

client = Groq()

# Initialize the chat history as Streamlit session state if not present already
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Streamlit page title
st.title("🦙 LLAMA 3.1. ChatBot")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input field for user's message
user_prompt = st.chat_input("Ask LLAMA...")

if user_prompt:
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    # Retrieve relevant documents based on user input
    retrieved_docs_paths = retrieve_documents(user_prompt)

    # Ler o conteúdo dos documentos recuperados
    retrieved_docs_content = [read_pdf(path) for path in retrieved_docs_paths]

    # Combine os conteúdos dos documentos em uma string
    combined_context = "\n".join(retrieved_docs_content)

    # Criar mensagens para o LLM, usando apenas o contexto dos documentos
    messages = [
        {"role": "system", "content": "You are a helpful assistant that only answers based on the provided documents."},
        {"role": "user", "content": user_prompt},
        {"role": "system", "content": f"Relevant information: {combined_context}"}
    ]

    # Enviar a mensagem do usuário ao LLM e obter uma resposta
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages
    )

    assistant_response = response.choices[0].message.content
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

    # Display the LLM's response
    with st.chat_message("assistant"):
        st.markdown(assistant_response)
