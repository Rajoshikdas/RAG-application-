import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from src.vectorstore import FaissVectorStore

load_dotenv()

class RAGSearch:
    """RAG (Retrieval Augmented Generation) search pipeline"""
    
    def __init__(self, vector_store: FaissVectorStore = None, groq_api_key: str = None):
        """
        Initialize RAG search
        
        Args:
            vector_store: FAISS vector store instance
            groq_api_key: Groq API key (uses env variable if not provided)
        """
        self.vector_store = vector_store or FaissVectorStore()
        
        # Initialize Groq LLM
        api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.llm = ChatGroq(
            model="gemma-7b-it",
            temperature=0.7,
            groq_api_key=api_key
        )
        
        # RAG prompt template
        self.rag_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a helpful assistant that answers questions based on the provided context.

Context:
{context}

Question:
{question}

Based on the context above, provide a clear and concise answer. If the context doesn't contain relevant information, say so."""
        )
    
    def search_and_summarize(self, query: str, top_k: int = 3) -> str:
        """
        Search for relevant documents and generate a summary using LLM
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            
        Returns:
            LLM-generated answer based on retrieved context
        """
        print(f"Searching for: '{query}'")
        
        # Retrieve relevant documents
        results = self.vector_store.query(query, top_k=top_k)
        
        if not results:
            return "No relevant documents found."
        
        # Build context from retrieved documents
        context_parts = []
        for i, result in enumerate(results):
            # Handle different result formats
            text = result.get('metadata', {}).get('text') or result.get('content', '')
            if text:
                context_parts.append(f"Document {i+1}:\n{text}")
        
        context = "\n\n".join(context_parts)
        
        if not context:
            return "No relevant documents found."
        
        print(f"Retrieved {len(results)} documents")
        
        # Format prompt
        prompt_text = self.rag_prompt.format(context=context, question=query)
        
        # Generate response using Groq
        print("Generating response...")
        response = self.llm.invoke([HumanMessage(content=prompt_text)])
        
        return response.content
    
    def batch_search(self, queries: list, top_k: int = 3) -> dict:
        """
        Process multiple queries
        
        Args:
            queries: List of query strings
            top_k: Number of documents to retrieve per query
            
        Returns:
            Dictionary mapping queries to answers
        """
        results = {}
        for query in queries:
            results[query] = self.search_and_summarize(query, top_k)
        return results
