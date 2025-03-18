# AI-Tutor
# PDF-RAGify

## Prerequisites
- Python3
- ollama
- chroma db

## Setup
-- Move your data (pdfs) to the `data` directory
-- Run the below command
```
Create a virtual env and then run
$ pip install -r requirements.txt
```
-- Pull the required Embeddings Model and LLM using below commands:
```
$ ollama pull nomic-embed-text
$ ollama pull mistral
```
-- If you want to use different models then, pull them and makes changes in the code as well

## Generating Embeddings
```
$ python generate_db.py  
```
-- This will automatically read your pdf files and convert them into text and then divide them into chunks and generate embeddings for them and store them in the chroma DB.

## Querying
```
python query.py "{Your prompt related to the info in the PDFs}"
```
\---

Happy Coding :v: