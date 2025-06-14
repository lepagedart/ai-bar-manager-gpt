# rag_retriever.py

import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# Folder where your knowledge base files are stored
KB_FOLDER = "knowledge_base"

def build_vectorstore():
    print("🔍 Building vectorstore from knowledge_base/\n")

    # Load supported files
    documents = []
    for root, _, files in os.walk(KB_FOLDER):
        for file in files:
            path = os.path.join(root, file)
            if file.endswith(".pdf"):
                loader = PyPDFLoader(path)
            elif file.endswith(".txt"):
                loader = TextLoader(path)
            else:
                print(f"❌ Skipping unsupported file type: {file}")
                continue

            try:
                docs = loader.load()
                documents.extend(docs)
                print(f"✅ Loaded {len(docs)} docs from: {file}")
            except Exception as e:
                print(f"❌ Error loading {file}: {e}")

    if not documents:
        raise ValueError("❌ No valid documents found to embed. Vectorstore not created.")

    # Chunk the documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)
    print(f"\n🧩 Total chunks created: {len(chunks)}")

    # Create vectorstore
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("vectorstore")
    print("✅ Vectorstore built and saved to ./vectorstore\n")

if __name__ == "__main__":
    build_vectorstore()