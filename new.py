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
import difflib
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader


model_name='microsoft/phi-4'
# OUTPUT_DIR = 'text_data'
# loader = DirectoryLoader(OUTPUT_DIR, glob="*.md")
# documents = loader.load()
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=0)
# all_splits = text_splitter.split_documents(documents)


crawled_pages = '/Users/yashkhowal/Desktop/AI-Tutor/text_data/'

loader = DirectoryLoader(crawled_pages, 
    loader_cls=TextLoader
)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=0)
all_splits = text_splitter.split_documents(documents)

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name="nomic-ai/nomic-embed-text-v1.5", device=None, batch_size=32):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, trust_remote_code=True, device=self.device)
        self.batch_size = batch_size  # Control batch size

    def embed_documents(self, texts):
        texts = ["search_document: "+i for i in texts]
        return self.model.encode(
            texts, 
            convert_to_numpy=True, 
            device=self.device, 
            batch_size=self.batch_size
        ).tolist()

    def embed_query(self, text):
        return self.model.encode(
            ['search_query: '+text], 
            convert_to_numpy=True, 
            device=self.device
        )[0].tolist()
    
local_embeddings = SentenceTransformerEmbeddings(batch_size=64)
vectorstore = Chroma.from_documents(documents=all_splits, embedding=local_embeddings, persist_directory='./chroma_rl')
print(f"Number of document splits: {len(all_splits)}")
print(f"Sample split document: {all_splits[0].page_content[:200]}")

def extract_answers(answers):
    answers = answers.split("<|im_start|>assistant<|im_sep|>")[-1]
    return answers


def get_title(links):
    titles = [i.metadata['source'].split(crawled_pages)[-1].split("_")[0] for i in links]
    result = get_most_related_keys(titles, data_swap)
    return result
    
with open(crawled_pages+ "output.json") as f:
    data = json.load(f)
def swap_keys_and_values(my_dict):
    """Swap the keys and values in the dictionary."""
    return {value: key for key, value in my_dict.items()}
def get_values_by_keys(my_dict, keys):
    """Return the values from the dictionary based on a list of keys."""
    return [my_dict[key] for key in keys if key in my_dict]


data_swap = swap_keys_and_values(data)

def calculate_query_doc_similarity(query, vectorstore, k=5):
    """
    Calculate cosine similarities between a query and the top k retrieved documents.
    
    Args:
        query (str): The input query string.
        vectorstore: The vector store instance (e.g., Chroma).
        k (int): Number of top documents to retrieve (default: 5).
    
    Returns:
        dict: Contains doc_similarity_pairs and average_similarity.
    """
    # Initialize retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # Step 1: Get the query embedding
    embedding_function = vectorstore.embeddings  # Use the same embedding function
    query_embedding = embedding_function.embed_query(query)

    # Step 2: Retrieve top k documents
    retrieved_docs = retriever.get_relevant_documents(query)

    # Step 3: Recompute embeddings for retrieved documents
    doc_embeddings = [embedding_function.embed_query(doc.page_content) for doc in retrieved_docs]

    query_embedding = np.array(query_embedding).reshape(1, -1)
    doc_embeddings = np.array(doc_embeddings)

    # Compute similarities
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0] if doc_embeddings.size > 0 else np.array([])

    # Sort similarities in descending order
    sorted_similarities = np.sort(similarities)[::-1]  

    # Compute average similarity
    average_similarity = np.mean(similarities) if similarities.size > 0 else 0.0

    return sorted_similarities[:2], average_similarity



def get_most_related_keys(titles, dictionary, top_n=5):
    """
    Find the top N keys in a dictionary most related to a given list of titles based on cosine similarity.
    
    Args:
        titles (list of str): List of title strings to compare against.
        dictionary (dict): Dictionary with keys (strings) and values.
        top_n (int): Number of top related keys to return (default: 5).
    
    Returns:
        list: List of tuples (title, [(key, similarity_score)]) sorted by similarity in descending order.
    """
    embedding_function = vectorstore.embeddings  # Use the same embedding function
    
    # Step 1: Embed all dictionary keys
    key_embeddings = np.array(embedding_function.embed_documents(list(dictionary.keys())))

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
        related_keys = [(list(dictionary.keys())[i], cos_sim[0][i]) for i in top_n_indices]
        
        # Append the result for the current title
        most_related.append(related_keys)
    
    # Return the results
    if len(most_related):

        most_related = [i[0] for i in most_related[0]]

    return most_related
def query_rag_model(query, k):
    # Update the retriever's 'k' value
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    rag_chain = RetrievalQA(
        retriever=retriever,
        combine_documents_chain=stuff_chain
    )
    
    # Run the RAG pipeline
    response = rag_chain.run(query)
    response = extract_answers(response)

    citations = rag_chain.retriever.get_relevant_documents(query)

    # Get relevant titles
    citation_titles = get_title(citations)

    # Similarity info
    top2_Score, sim_score = calculate_query_doc_similarity(query, vectorstore, k)

    return response, citation_titles, top2_Score, f"{sim_score:.4f}"

def rag_interface(query, k=5):
    response, titles, top2_Score, avg_score = query_rag_model(query, k)

    # Remove duplicates and format as text
    unique_titles = list(set(titles))

    if (len(unique_titles) == 0 and float(avg_score) < 0.55) or float(avg_score) < 0.55:
        final_response = response
    else:
        references = "\n\nReferences:\n\n" + "\n".join([f"- {t}" for t in unique_titles])
        final_response = response + references

    return final_response, top2_Score, avg_score


if __name__ == "__main__":


    llm = VLLM(
        model=model_name,
        tensor_parallel_size=1,
        trust_remote_code=True,  # mandatory for hf models
        max_new_tokens=32000,
    )

    local_embeddings = SentenceTransformerEmbeddings(batch_size=64)
    vectorstore = Chroma(persist_directory="./chroma_rl", embedding_function=local_embeddings)

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

    # template="""<|im_start|>system<|im_sep|><|im_end|>

# <|im_start|>user<|im_sep|>Answer the question based on the context below:\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:<|im_start|>assistant<|im_sep|>""",



    # Define your prompt template

    prompt_template = PromptTemplate(

        input_variables=["context", "question"],

         template=RAG_TEMPLATE,
    )

    # Define your LLMChain
    
    llm_chain = LLMChain(llm=llm, prompt=prompt_template)  # Replace with your LLM


    stuff_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context"  
    )


    iface = gr.Interface(

        fn=rag_interface,  # Function to call
        
        inputs=[  # List of inputs: query and k (slider)
            gr.Textbox(label="Query", placeholder="Enter your question here..."),
            gr.Slider(minimum=1, maximum=25, step=1, value=5, label="Number of Documents to Retrieve (k)"),
        ]
        ,
        
        outputs=[  # Two outputs: response (markdown) and score (text)
            
            gr.Markdown(label="Response"),

            gr.Textbox(label="Top 2 Scores"),

            gr.Textbox(label="Avg Confidence Score"),

        ],

        
        title="# 📚 Deakin AI Tutor",
        
        description="Enter your query and adjust 'k' to control the number of documents to retrieve for answering."

    )



    iface.launch(share=True)