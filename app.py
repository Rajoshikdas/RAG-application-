from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch

# Example usage
if __name__ == "__main__":
    
    docs = load_all_documents("data")
    store = FaissVectorStore("faiss_store")
    
    # Build index if it doesn't exist
    import os
    if not os.path.exists("faiss_store/faiss.index"):
        store.build_from_documents(docs)
    else:
        store.load()
    
    # Query and display results
    query = "What is attention mechanism?"
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}\n")
    
    results = store.query(query, top_k=3)
    
    print(f"Found {len(results)} relevant documents:\n")
    for i, result in enumerate(results, 1):
        text = result.get('metadata', {}).get('text', 'N/A')
        distance = result.get('distance', 'N/A')
        print(f"Document {i} (Distance: {distance:.4f}):")
        print(f"{text[:500]}...")  # Print first 500 chars
        print("-" * 60)
    
    print("\n✅ RAG Pipeline executed successfully!")