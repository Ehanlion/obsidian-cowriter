import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# --- CONFIGURATION ---
# Load environment variables from .env file
load_dotenv()

# Get the vault path from the environment variables
VAULT_PATH = os.getenv("OBSIDIAN_VAULT_PATH")
if not VAULT_PATH:
    raise ValueError("OBSIDIAN_VAULT_PATH environment variable not set.")

# Define the path for the persistent vector database
CHROMA_PATH = "chroma_db"
# Define the embedding model to use
# Make sure this model is available in your Ollama instance (e.g., run `ollama list`)
# Using 'gemma' as an example, but you can change it to 'magistral', 'qwen', etc.
EMBEDDING_MODEL = "gemma" 

def main():
    """
    Main function to ingest the Obsidian vault documents.
    1. Loads all markdown documents from the vault.
    2. Splits the documents into smaller, manageable chunks.
    3. Generates embeddings for each chunk using a local Ollama model.
    4. Stores the chunks and their embeddings in a persistent ChromaDB vector store.
    """
    print("--- Starting Document Ingestion ---")

    # Validate that the vault path exists
    if not os.path.exists(VAULT_PATH):
        print(f"Error: Vault path does not exist: {VAULT_PATH}")
        return

    print(f"Loading documents from: {VAULT_PATH}")
    # Use DirectoryLoader to load all markdown files recursively
    # UnstructuredMarkdownLoader is used for its effectiveness with .md files
    loader = DirectoryLoader(
        VAULT_PATH, 
        glob="**/*.md", 
        loader_cls=UnstructuredMarkdownLoader,
        show_progress=True,
        use_multithreading=True,
        silent_errors=True 
    )
    documents = loader.load()

    if not documents:
        print("No markdown documents found in the vault. Exiting.")
        return
    print(f"Successfully loaded {len(documents)} documents.")

    print("Splitting documents into chunks...")
    # Use a text splitter to break down large documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} chunks.")

    print(f"Initializing embedding model: {EMBEDDING_MODEL}")
    # Initialize the Ollama model for creating embeddings.
    # The 'show_progress' argument is no longer supported in recent versions.
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    print(f"Creating and persisting vector store at: {CHROMA_PATH}")
    # Create the Chroma vector store from the document chunks and embeddings
    # This process can take some time as it generates embeddings for all chunks.
    Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=CHROMA_PATH
    )

    print("--- Ingestion Complete ---")
    print(f"Vector store created at '{CHROMA_PATH}'. You can now query your vault.")

if __name__ == "__main__":
    main()
