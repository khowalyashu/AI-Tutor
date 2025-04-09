import argparse
import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_chroma import Chroma
from get_embedding_func import get_embedding_function

CHROMA_PATH = "chroma"
DATA_PATH = "data"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()

    if args.reset:
        print("🧹 Resetting the database...")
        clear_database()

    try:
        documents = load_documents()
        if not documents:
            print("⚠️ No documents found in the data path.")
            return
    except Exception as e:
        print(f"❌ Failed to load documents: {e}")
        return

    print(f"📄 Loaded {len(documents)} documents")

    chunks = split_documents(documents)
    print(f"✂️ Split into {len(chunks)} chunks")

    chunks = calculate_chunk_ids(chunks)
    add_to_chroma(chunks)

def load_documents():
    loader = PyPDFDirectoryLoader(DATA_PATH)
    return loader.load()

def split_documents(documents: list[Document]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return splitter.split_documents(documents)

def calculate_chunk_ids(chunks: list[Document]) -> list[Document]:
    chunk_counters = {}

    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", 0)
        key = f"{source}:{page}"

        chunk_index = chunk_counters.get(key, 0)
        chunk.metadata["id"] = f"{key}:{chunk_index}"
        chunk_counters[key] = chunk_index + 1

    return chunks

def add_to_chroma(chunks: list[Document]):
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function(),
    )

    existing_items = db.get(include=[])
    existing_ids = set(existing_items.get("ids", []))
    print(f"📚 Existing documents in DB: {len(existing_ids)}")

    new_chunks = [
        chunk for chunk in chunks
        if chunk.metadata["id"] not in existing_ids
    ]

    if new_chunks:
        print(f"➕ Adding {len(new_chunks)} new documents")
        new_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_ids)
    else:
        print("✅ No new documents to add")

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("✅ Database directory cleared")

if __name__ == "__main__":
    main()
