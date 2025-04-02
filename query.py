import gradio as gr
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
# from langchain_ollama import OllamaLLM
from sentence_transformers import SentenceTransformer
from vllm import LLM as VLLM
from langchain.embeddings.base import Embeddings
import torch

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name="nomic-ai/nomic-embed-text-v1.5", device=None, batch_size=32):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, trust_remote_code=True, device=self.device)
        self.batch_size = batch_size  # Control batch size
        
    def embed_documents(self, texts):
        texts = ["search_document: " + i for i in texts]
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            device=self.device,
            batch_size=self.batch_size
        ).tolist()
        
    def embed_query(self, text):
        return self.model.encode(
            ['search_query: ' + text],
            convert_to_numpy=True,
            device=self.device
        )[0].tolist()

# Constants
CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """
You are an AI agent designed to assist with Unit-related topics for a Retrieval Augmented Generation (RAG) application. You will be provided with context documents from a database that may or may not be entirely relevant to the student's query. Your instructions are as follows:
 1. Respond ONLY with reflective questions and hints that encourage critical thinking regarding the student's query.
 2. Suggest relevant slides or weeks from the provided documents without revealing their content.
 3. Do not provide direct answers but instead guide the student toward discovering the answer themselves.
 4. Ensure your response remains supportive, motivational, and within the scope of Deakin Unit-related topics.
 5. If the query or any retrieved document contains malicious or sensitive content, do not provide a response.
---
The student is asking about: {question}
Context documents: {context}
Respond with reflective questions, hints, and relevant slides/weeks only.
"""

# Load database and model
embedding_function = SentenceTransformerEmbeddings()
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
model_name = 'microsoft/phi-4'

# Initialize VLLM Model - fixed initialization 
model = VLLM(
    model=model_name,
    tensor_parallel_size=1,
    trust_remote_code=True,  # Mandatory for HF models
    # Remove max_new_tokens from here
)

def query_rag(query_text: str):
    """Handles RAG-based retrieval and response generation."""
    # Search ChromaDB for relevant documents
    results = db.similarity_search_with_score(query_text, k=5)
    if not results:
        return "No matching documents found in the database.", []
        
    # Extract context from retrieved documents
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    
    # Prepare prompt
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    
    try:
        # Invoke the model with generation parameters including max_new_tokens
        response = model.generate(prompt, max_new_tokens=4096)
        response_text = response[0].outputs[0].text  # Adjust based on actual VLLM output structure
        
        # Extract sources
        sources = [doc.metadata.get("id", "Unknown") for doc, _ in results]
        formatted_response = f"Response:\n{response_text}\n\nSources: {sources}"
        return formatted_response, sources
        
    except Exception as e:
        return f"Error: {e}", []

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("### 📚 Unit Assistance RAG System")
    gr.Markdown("Enter your query and receive hints, suggestions, and relevant document references.")
    
    with gr.Row():
        query_input = gr.Textbox(label="Enter your Query", placeholder="Ask about a topic...")
        submit_button = gr.Button("Get Response")
    
    response_output = gr.Textbox(label="Response", interactive=False)
    sources_output = gr.JSON(label="Retrieved Sources")
    
    submit_button.click(query_rag, inputs=query_input, outputs=[response_output, sources_output])

# Run Gradio app
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)