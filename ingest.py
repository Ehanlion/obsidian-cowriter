import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# Load the .env file here
load_dotenv()

# Config vars from the .env files
VAULT_PATH = os.getenv("OBSIDIAN_VAULT_PATH")               # path to osidian vault
CHROMA_PATH = os.getenv("CHROMA_DB_PATH")                   # path to the chromadb directory
EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL")       # name of model to embed (ollama)

# Raise errors if .env fails to contain files
if not VAULT_PATH:
    raise ValueError("Error. OBSIDIAN_VAULT_PATH variable expected in .env file. Not found.")
if not CHROMA_PATH:
    raise ValueError("Error. CHROMA_DB_PATH variable expected in .env file. Not found.")
if not EMBEDDING_MODEL:
    raise ValueError("Error. OLLAMA_EMBEDDING_MODEL variable expected in .env file. Not found.")

# function to print out contents of a file directory
def printDirectory(path : str, printOnlyMd : bool = False):
    """
    Helper function to print contents of a directory

    Args:
        path (str): directory to search
        printOnlyMd (bool): if True, print only *.md file, else print all files
    """

    print(f"\n--- Printing Directory ---")
    print(f"\tSearching directory: {path}")
    print(f"Mode: {'Only *.md Files' if printOnlyMd else 'All Files'}")


    totalFiles = 0
    mdFiles = 0
    files = []

    for dirpath, _, filenames in os.walk(path):
        for filename in filenames:
            totalFiles += 1
            filePath = os.path.join(dirpath, filename)
            if(filename.endswith('.md')):
                mdFiles += 1
                if printOnlyMd:
                    files.append(filePath)
            if not printOnlyMd:
                files.append(filePath)
    
    for f in files:
        print(f"\t  -> Found: {f}")

    print(f"\nSummary:")
    print(f"\tTotal Markdown Files found: {mdFiles}")
    print(f"\tTotal Overall Files found {totalFiles}")
    print(f"--- Finished Printing Directory ---\n")
    return

"""
Main method, used to injest the contents of a directory.
"""
def main():
    # Print loaded environment data from .env
    print("\n=== Environment Data ===")
    print(f"\tVault path: {VAULT_PATH}")
    print(f"\tChroma DB Path: {CHROMA_PATH}")
    print(f"\tEmbedding Model ID: {EMBEDDING_MODEL}")
    print("=== End Environment Data ===\n")

    # Validate path to the obsidian vault
    if not os.path.exists(VAULT_PATH):
        print(f"Error: Vault path is invalid. Loaded path: {VAULT_PATH}")
        return
    else:
        print(f"Validated vault path {VAULT_PATH}, proceeding to load all md files")

    # Use a directory loaded to load all markdown files
    loader = DirectoryLoader(
        path=VAULT_PATH,
        glob="**/*.md",
        loader_cls=UnstructuredMarkdownLoader,
        show_progress=True,
        use_multithreading=False,
        silent_errors=False
    )
    documents = loader.load() # load documents into documents variable
    if not documents:
        print(f"Failed to load documents, no markdown files in vault")
        return
    else:
        print(f"Succesfully loaded {len(documents)} documents")

if __name__ == "__main__":
    main() # run main