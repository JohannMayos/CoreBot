import gradio as gr
from chatbot import generate_response_with_context  # Importar a função correta

# Interface Gradio para consulta
def process_query(query):
    response = generate_response_with_context(query)  # Atualizado para usar a nova função
    return response

# Configuração da Interface Gradio
interface = gr.Interface(
    fn=process_query,
    inputs=gr.Textbox(lines=2, placeholder="Digite sua pergunta sobre música metal..."),
    outputs="text",
    title="Consultor de Música Metal",
    description="Consulte sobre a história do metal, subgêneros e bandas."
)

# Executar a interface
if __name__ == "__main__":
    interface.launch()
