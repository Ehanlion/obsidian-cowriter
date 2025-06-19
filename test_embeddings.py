from langchain_ollama import OllamaEmbeddings
import numpy as np

# --- CONFIGURATION ---
# Make sure this model is running in your Ollama instance
EMBEDDING_MODEL = "qwen3:30b" 

def main():
    """
    A simple script to test if the OllamaEmbeddings object is working correctly.
    """
    print(f"--- Testing Ollama Embeddings with model: '{EMBEDDING_MODEL}' ---")

    try:
        # 1. Initialize the embeddings object
        print("1. Initializing the OllamaEmbeddings object...")
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        print("   - Success: Object initialized.")

        # 2. Define a simple test sentence
        test_text = "The silver dragon sleeps on a hoard of gold."
        print(f"\n2. Preparing to generate embedding for the text: '{test_text}'")

        # 3. Generate the embedding
        # This is the line that actually communicates with Ollama
        print("3. Calling the embed_query() method... (This may take a moment)")
        vector = embeddings.embed_query(test_text)
        print("   - Success: Communication with Ollama was successful!")

        # 4. Visualize the output
        print("\n4. Analyzing the received embedding vector...")
        
        # Check if the output is a list of numbers (floats)
        is_list_of_floats = all(isinstance(item, float) for item in vector)
        
        print(f"   - Type of output: {type(vector)}")
        print(f"   - Is it a list of numbers? {'Yes' if is_list_of_floats else 'No'}")
        print(f"   - Number of dimensions (vector length): {len(vector)}")
        
        # Print a small sample of the vector
        print(f"   - First 5 dimensions: {np.round(vector[:5], 3)}")
        print(f"   - Last 5 dimensions: {np.round(vector[-5:], 3)}")

        print("\n--- Test Complete: The embeddings object is alive and working correctly! ---")

    except Exception as e:
        print("\n--- TEST FAILED ---")
        print("An error occurred during the test. This likely means there is an issue with:")
        print("  - Your Ollama server (is it running?)")
        print("  - The connection between this script and the server.")
        print("  - The specified model name (does it exist in Ollama?).")
        print(f"\nError details: {e}")

if __name__ == "__main__":
    main()