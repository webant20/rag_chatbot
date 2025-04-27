import os
import time
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import shutil

# ─────────────────────────────────────────────────────────────────────────────
start_time = time.time()

# ─────────────────────────────────────────────────────────────────────────────
# Prepare
docs_dir = "docs"
ingested_docs_dir = "docs_ingested"  # Directory for moved documents
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)

# Make sure the doc_ingested directory exists
os.makedirs(ingested_docs_dir, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Load and process each document separately
for filename in os.listdir(docs_dir):
    filepath = os.path.join(docs_dir, filename)
    if not (filename.lower().endswith(".txt") or filename.lower().endswith(".pdf")):
        continue

    print(f"📄 Processing file: {filename}")

    # Load file
    if filename.lower().endswith(".txt"):
        loader = TextLoader(filepath)
    else:
        loader = PyPDFLoader(filepath)

    documents = loader.load()

    # Split into chunks
    chunks = splitter.split_documents(documents)

    # Create FAISS vectorstore for this document
    db = FAISS.from_documents(chunks, embedding)

    # Save with a folder name based on filename
    vectorstore_dir = os.path.join("vectorstore", filename.replace(".", "_"))
    os.makedirs(vectorstore_dir, exist_ok=True)
    db.save_local(vectorstore_dir)

    print(f"✅ Saved vectors for {filename} to {vectorstore_dir}")

    # Move the processed file to 'doc_ingested' directory
    ingested_filepath = os.path.join(ingested_docs_dir, filename)
    shutil.move(filepath, ingested_filepath)
    print(f"✅ Moved {filename} to {ingested_docs_dir}")

# ─────────────────────────────────────────────────────────────────────────────
end_time = time.time()
print(f"\n🕒 Total ingestion time: {end_time - start_time:.2f} seconds")
