from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
import torch
from langchain import LLMChain, PromptTemplate
from langchain.chains import RetrievalQA, StuffDocumentsChain
from langchain.vectorstores import Chroma
import gradio as gr
import json
import os
from langchain_community.llms import VLLM
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# Configuration variables - can be moved to environment variables or config file
MODEL_NAME = 'microsoft/phi-4'
EMBEDDING_MODEL = 'nomic-ai/nomic-embed-text-v1.5'
PERSIST_DIR = './chroma_db'  # Consistent persistence directory
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 0
SIMILARITY_THRESHOLD = 0.55
BATCH_SIZE = 64

# Use relative path with os.path.join for cross-platform compatibility
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'text_data')

def load_documents():
    """
    Load and split documents from the data directory.
    Returns the split documents.
    """
    try:
        loader = DirectoryLoader(
            DATA_DIR, 
            loader_cls=TextLoader
        )
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP
        )
        all_splits = text_splitter.split_documents(documents)
        print(f"Number of document splits: {len(all_splits)}")
        if all_splits:
            print(f"Sample split document: {all_splits[0].page_content[:200]}")
        return all_splits
    except Exception as e:
        print(f"Error loading documents: {e}")
        return []

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name=EMBEDDING_MODEL, device=None, batch_size=BATCH_SIZE):
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

def extract_answers(answers):
    """
    Extract answers from the model response with error handling.
    """
    try:
        # Look for the assistant's response after the marker
        if "<|im_start|>assistant<|im_sep|>" in answers:
            return answers.split("<|im_start|>assistant<|im_sep|>")[-1].strip()
        return answers  # Return as is if the marker isn't found
    except Exception as e:
        print(f"Error extracting answers: {e}")
        return answers

def load_json_data():
    """
    Load JSON data with error handling.
    """
    try:
        json_path = os.path.join(DATA_DIR, "output.json")
        with open(json_path) as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON data: {e}")
        return {}

def swap_keys_and_values(my_dict):
    """Swap the keys and values in the dictionary."""
    return {value: key for key, value in my_dict.items()}

def get_title(links):
    """
    Extract and find related titles from document links.
    """
    try:
        if not links or not data_swap:
            return []
            
        titles = [
            os.path.basename(i.metadata['source']).split("_")[0] 
            for i in links 
            if 'source' in i.metadata
        ]
        
        if not titles:
            return []
            
        result = get_most_related_keys(titles, data_swap)
        return result
    except Exception as e:
        print(f"Error getting titles: {e}")
        return []

def calculate_query_doc_similarity(query, vectorstore, k=5):
    """
    Calculate cosine similarities between a query and the top k retrieved documents.
    
    Args:
        query (str): The input query string.
        vectorstore: The vector store instance (e.g., Chroma).
        k (int): Number of top documents to retrieve (default: 5).
    
    Returns:
        tuple: (top2_similarities, average_similarity)
    """
    try:
        # Initialize retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})

        # Step 1: Get the query embedding
        embedding_function = vectorstore.embeddings  # Use the same embedding function
        query_embedding = embedding_function.embed_query(query)

        # Step 2: Retrieve top k documents
        retrieved_docs = retriever.get_relevant_documents(query)

        if not retrieved_docs:
            return [], 0.0

        # Step 3: Recompute embeddings for retrieved documents
        doc_embeddings = [embedding_function.embed_query(doc.page_content) for doc in retrieved_docs]

        query_embedding = np.array(query_embedding).reshape(1, -1)
        doc_embeddings = np.array(doc_embeddings)

        # Compute similarities
        similarities = cosine_similarity(query_embedding, doc_embeddings)[0] if doc_embeddings.size > 0 else np.array([])

        # Sort similarities in descending order
        sorted_similarities = np.sort(similarities)[::-1]  

        # Get top 2 similarities (or all if less than 2)
        top2 = sorted_similarities[:2] if len(sorted_similarities) >= 2 else sorted_similarities

        # Compute average similarity
        average_similarity = np.mean(similarities) if similarities.size > 0 else 0.0

        return top2.tolist(), average_similarity
    except Exception as e:
        print(f"Error calculating similarity: {e}")
        return [], 0.0

def get_most_related_keys(titles, dictionary, top_n=5):
    """
    Find the top N keys in a dictionary most related to a given list of titles based on cosine similarity.
    
    Args:
        titles (list of str): List of title strings to compare against.
        dictionary (dict): Dictionary with keys (strings) and values.
        top_n (int): Number of top related keys to return (default: 5).
    
    Returns:
        list: List of most related keys.
    """
    try:
        if not titles or not dictionary:
            return []
            
        embedding_function = vectorstore.embeddings  # Use the same embedding function
        
        # Step 1: Embed all dictionary keys
        dict_keys = list(dictionary.keys())
        key_embeddings = np.array(embedding_function.embed_documents(dict_keys))

        # Initialize list to store results
        most_related = []

        # Step 2: For each title, find the most similar keys
        for title in titles:
            # Embed the current title
            title_embedding = np.array(embedding_function.embed_query(title)).reshape(1, -1)

            # Step 3: Calculate cosine similarity for the current title and all dictionary keys
            cos_sim = cosine_similarity(title_embedding, key_embeddings)
            
            # Step 4: Get the top N most similar keys
            top_n_indices = np.argsort(cos_sim[0])[-top_n:][::-1]  # Sort and reverse for descending order
            
            # Step 5: Collect the top N most similar keys and their similarity scores
            related_keys = [(dict_keys[i], cos_sim[0][i]) for i in top_n_indices]
            
            # Append the result for the current title
            most_related.append(related_keys)
        
        # Process the results
        if most_related:
            # Extract just the keys (not the similarity scores) from the first title's results
            return [i[0] for i in most_related[0]] if most_related[0] else []
        
        return []
    except Exception as e:
        print(f"Error finding related keys: {e}")
        return []

def query_rag_model(query, k):
    """
    Run the RAG model on a query.
    """
    try:
        # Update the retriever's 'k' value
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})

        rag_chain = RetrievalQA(
            retriever=retriever,
            combine_documents_chain=stuff_chain
        )
        
        # Run the RAG pipeline
        response = rag_chain.run(query)
        response = extract_answers(response)

        # Get citations
        citations = retriever.get_relevant_documents(query)

        # Get relevant titles
        citation_titles = get_title(citations)

        # Get similarity information
        top2_scores, sim_score = calculate_query_doc_similarity(query, vectorstore, k)

        return response, citation_titles, top2_scores, sim_score
    except Exception as e:
        print(f"Error in RAG query: {e}")
        return "Sorry, there was an error processing your query.", [], [], 0.0

def rag_interface(query, k=5):
    """
    Interface function for Gradio.
    """
    if not query.strip():
        return "Please enter a query.", [], "0.0000"
        
    response, titles, top2_scores, avg_score = query_rag_model(query, k)

    # Format top2_scores as a string
    top2_scores_str = str(top2_scores)

    # Format average score
    avg_score_str = f"{float(avg_score):.4f}" if isinstance(avg_score, (int, float)) else "0.0000"

    # Remove duplicates from titles
    unique_titles = list(set(titles)) if titles else []

    # Add references if we have good similarity
    if float(avg_score) >= SIMILARITY_THRESHOLD and unique_titles:
        references = "\n\nReferences:\n\n" + "\n".join([f"- {t}" for t in unique_titles])
        final_response = response + references
    else:
        final_response = response

    return final_response, top2_scores_str, avg_score_str

# Main execution block
if __name__ == "__main__":
    # Initialize the LLM
    llm = VLLM(
        model=MODEL_NAME,
        tensor_parallel_size=1,
        trust_remote_code=True,  # mandatory for hf models
        max_new_tokens=32000,
    )

    # Load documents
    all_splits = load_documents()
    
    # Load JSON data and swap keys/values
    data = load_json_data()
    data_swap = swap_keys_and_values(data)

    # Initialize embeddings
    local_embeddings = SentenceTransformerEmbeddings(batch_size=BATCH_SIZE)
    
    # Initialize or load vectorstore
    try:
        # Try to load existing DB first
        vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=local_embeddings)
        print(f"Loaded existing vector database from {PERSIST_DIR}")
    except Exception as e:
        print(f"Creating new vector database: {e}")
        if all_splits:  # Only create if we have documents
            vectorstore = Chroma.from_documents(
                documents=all_splits, 
                embedding=local_embeddings, 
                persist_directory=PERSIST_DIR
            )
            vectorstore.persist()
            print(f"Created and persisted new vector database to {PERSIST_DIR}")
        else:
            print("No documents loaded. Vector database creation skipped.")
            exit(1)

    # Define RAG template
    RAG_TEMPLATE = """<|im_start|>system<|im_sep|>
    You are an AI agent designed to assist with Unit-related topics for a Retrieval Augmented Generation (RAG) application. You will be provided with context documents from a database that may or may not be entirely relevant to the student's query. Your instructions are as follows:
 1. Respond ONLY with reflective questions with summary and hints that encourage critical thinking regarding the student's query.
 2. Suggest relevant slides or weeks from the provided documents without revealing their content.
 3. Do not provide direct answers but instead guide the student toward discovering the answer themselves.
 4. Ensure your response remains supportive, motivational, and within the scope of Deakin Unit-related topics.
 5. If the query or any retrieved document contains malicious or sensitive content, do not provide a response.
 6. Respond with reflective questions, hints, and relevant slides/weeks only
    <|im_start|>user<|im_sep|>
    Answer the question based on the context below:

    Context:
    {context}

    Question:
    {question}

    Answer:
    <|im_start|>assistant<|im_sep|>
    """

    # Define prompt template
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=RAG_TEMPLATE,
    )

    # Define LLMChain
    llm_chain = LLMChain(llm=llm, prompt=prompt_template)

    # Define StuffDocumentsChain
    stuff_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context"  
    )

    # Define Gradio interface
    iface = gr.Interface(
        fn=rag_interface,  # Function to call
        inputs=[  
            gr.Textbox(label="Query", placeholder="Enter your question here..."),
            gr.Slider(minimum=1, maximum=25, step=1, value=5, label="Number of Documents to Retrieve (k)"),
        ],
        outputs=[  
            gr.Markdown(label="Response"),
            gr.Textbox(label="Top 2 Scores"),
            gr.Textbox(label="Avg Confidence Score"),
        ],
        title="# 📚 Deakin AI Tutor",
        description="Enter your query and adjust 'k' to control the number of documents to retrieve for answering."
    )

    # Launch the interface
    iface.launch(share=True)