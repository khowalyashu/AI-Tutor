import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM # Updated import
from get_embedding_func import get_embedding_function

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

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)

def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    
    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)
    
    # Check if we got any results
    if not results:
        print("No matching documents found in the database.")
        return "No matching documents found in the database."
    
    # Print number of results for debugging
    print(f"Found {len(results)} relevant documents")
    
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    
    # Using updated Ollama implementation
    model = OllamaLLM(model="mistral")
    
    try:
        # Invoke the model and get response
        response_text = model.invoke(prompt)
        print(f"Raw response from model: {response_text}")

        
        # Process sources
        sources = [doc.metadata.get("id", None) for doc, _ in results]
        formatted_response = f"Response: {response_text}\nSources: {sources}"
        print(formatted_response)
        return response_text
    except Exception as e:
        print(f"Error invoking the model: {e}")
        return f"Error: {e}"

if __name__ == "__main__":
    main()