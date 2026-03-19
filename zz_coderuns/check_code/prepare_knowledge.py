from fed_rag import RAGSystem, RAGConfig
from fed_rag.knowledge_stores import InMemoryKnowledgeStore
from fed_rag.retrievers import HFSentenceTransformerRetriever
from fed_rag.generators import HFPretrainedModelGenerator
from fed_rag.data_structures.knowledge_node import KnowledgeNode, NodeType
from datasets import load_dataset

# Small knowledge base - simple facts
KNOWLEDGE_CHUNKS = [
    "The Eiffel Tower is located in Paris, France.",
    "The capital of Japan is Tokyo.",
    "Water boils at 100 degrees Celsius at sea level.",
    "The Earth orbits around the Sun.",
    "Python is a popular programming language.",
    "The Great Wall of China is visible from space.",
    "Mount Everest is the highest mountain on Earth.",
    "The Pacific Ocean is the largest ocean.",
    "London is the capital of the United Kingdom.",
    "The speed of light is approximately 299,792 kilometers per second.",
    "Gold is a chemical element with symbol Au.",
    "The human body has 206 bones.",
    "Mars is known as the Red Planet.",
    "The Amazon River is the longest river in South America.",
    "Shakespeare wrote Romeo and Juliet.",
    "The Great Pyramid of Giza is in Egypt.",
    "Carbon dioxide is a greenhouse gas.",
    "The Pacific Ring of Fire is an area of frequent earthquakes.",
    "DNA stands for deoxyribonucleic acid.",
    "The Sahara is the largest hot desert in the world."
]

def setup_knowledge_store():
    """Create and populate in-memory knowledge store"""
    
    # Initialize retriever (for encoding)
    retriever = HFSentenceTransformerRetriever(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Create knowledge nodes
    nodes = []
    for idx, text in enumerate(KNOWLEDGE_CHUNKS):
        # Encode the text
        embedding = retriever.encode_context(text)[0].tolist()
        
        # Create node
        node = KnowledgeNode(
            node_type=NodeType.TEXT,
            embedding=embedding,
            text_content=text,
            metadata={"source": "facts", "id": idx}
        )
        nodes.append(node)
    
    # Create and populate store
    knowledge_store = InMemoryKnowledgeStore()
    knowledge_store.load_nodes(nodes)
    
    print(f"✅ Loaded {knowledge_store.count} nodes into knowledge store")
    return knowledge_store, retriever

if __name__ == "__main__":
    knowledge_store, retriever = setup_knowledge_store()
    
    # Test retrieval
    query_emb = retriever.encode_query("What is the capital of France?")[0].tolist()
    results = knowledge_store.retrieve(query_emb, top_k=3)
    
    print("\n🔍 Test retrieval for 'What is the capital of France?':")
    for score, node in results:
        print(f"  Score: {score:.4f} | Text: {node.text_content}")