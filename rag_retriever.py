import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Folder where your knowledge base files are stored
KB_FOLDER = "knowledge_base"
VECTORSTORE_FILE = "vectorstore"

def build_vectorstore():
    print("\n🔎 Building vectorstore from knowledge_base/\n")

    documents = []
    total_files = 0
    total_loaded = 0
    skipped_files = []

    # Walk through all folders and files inside knowledge_base/
    for root, dirs, files in os.walk(KB_FOLDER):
        for file in files:
            total_files += 1
            path = os.path.join(root, file)

            # Handle supported file types
            if file.endswith(".pdf"):
                loader = PyPDFLoader(path)
            elif file.endswith(".txt"):
                loader = TextLoader(path)
            else:
                print(f"⚠️ Skipping unsupported file type: {file}")
                skipped_files.append(file)
                continue

            try:
                docs = loader.load()
                if len(docs) == 0:
                    print(f"⚠️ File loaded but returned no documents: {file}")
                    skipped_files.append(file)
                else:
                    documents.extend(docs)
                    print(f"✅ Loaded {len(docs)} docs from: {file}")
                    total_loaded += 1
            except Exception as e:
                print(f"❌ Error loading file {file}: {e}")
                skipped_files.append(file)

    print(f"\n📊 Summary: {total_loaded} files successfully loaded out of {total_files}")
    if skipped_files:
        print(f"⚠️ Skipped files: {skipped_files}")

    if not documents:
        raise ValueError("❌ No valid documents found to embed. Vectorstore not created.")

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)
    print(f"\n🧩 Total chunks created: {len(chunks)}")

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create vectorstore and save to disk
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTORSTORE_FILE)
    print("\n✅ Vectorstore built and saved successfully.\n")

# This function is called by app.py automatically on startup
def check_and_update_vectorstore(knowledge_folder):
    print("\n🔄 Checking knowledge base and updating vectorstore if needed...\n")
    build_vectorstore()

# If run manually:
if __name__ == "__main__":
    build_vectorstore()
    # Retrieval function called by app.py
def retrieve_codex_context(prompt: str) -> str:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(VECTORSTORE_FILE, embeddings, allow_dangerous_deserialization=True)
    results = db.similarity_search(prompt, k=3)
    return "\n\n".join([doc.page_content for doc in results])