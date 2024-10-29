import streamlit as st
from chatbot import generate_response_with_context  # Importa a função de geração de respostas

# Inicializa o histórico de conversas no estado da sessão se ainda não estiver presente
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Título da página Streamlit
st.title("🎸 Consultor de Música Metal")

# Exibe o histórico de conversas
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Campo de entrada para a pergunta do usuário
user_prompt = st.chat_input("Digite sua pergunta sobre música metal...")

if user_prompt:
    # Adiciona a mensagem do usuário ao histórico e exibe
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    # Gera resposta com base na pergunta do usuário
    assistant_response = generate_response_with_context(user_prompt)
    
    # Adiciona a resposta do assistente ao histórico e exibe
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
    
    # Exibe a resposta do LLM
    with st.chat_message("assistant"):
        st.markdown(assistant_response)
