import random
import torch
from datasets import load_dataset, Dataset
from fed_rag.knowledge_stores import InMemoryKnowledgeStore
from fed_rag.retrievers import HFSentenceTransformerRetriever
from fed_rag.data_structures import KnowledgeNode, NodeType


RETRIEVER_MODEL = "sentence-transformers/all-MiniLM-L6-v2" # Strong retriever used for robustness-style runs.
MECHANISM_RETRIEVER_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2" # Slightly weaker retriever with more headroom for the mechanism benchmark.
MAX_CORPUS_DOCS = 8000 # The maximum number of documents to load into the knowledge store.
MAX_TRAIN_PAIRS = 4000 # The maximum number of training pairs to use.
MAX_SERVER_VAL_PAIRS = 400 # Validation pairs used to pick the best global round.
MAX_FINAL_TEST_PAIRS = 500 # Final held-out test pairs used for the final report.
MAX_SHARED_QUALITY_PAIRS = 400 # Shared clean comparison pairs used for quality-aware weighting.
TOP_K = 10 # The number of top results to retrieve.
SEED = 42 # The seed for reproducibility.
MAX_RESPONSE_CHARS = 500 # The maximum number of characters to use for the response.


def load_beir_dataset(dataset_name: str = "nfcorpus"):
    """
    Loads a BEIR dataset from Hugging Face including the corpus, queries, and relevance judgments.

    Args:
        dataset_name (str): The name of the BEIR dataset to load (e.g., "nfcorpus").

    Returns:
        tuple: (doc_lookup, query_lookup, qrels_ds)
            - doc_lookup (dict): Mapping from doc_id to document text.
            - query_lookup (dict): Mapping from query_id to query text.
            - qrels_ds (Dataset): The relevance judgments dataset.
    """
    print(f"📥 Loading BEIR/{dataset_name}...")

    # corpus is the Knowledge Base or the collection of documents. It contains all the "answers" or evidence.
    try:
        corpus_ds = load_dataset(f"BeIR/{dataset_name}", "corpus", split="corpus")
    except Exception:
        corpus_ds = load_dataset(f"BeIR/{dataset_name}", split="train")

    # queries is a list of Questions or search terms users might type. Each query has a unique ID and text. These are the inputs you use to train and evaluate your RAG system.
    try:
        queries_ds = load_dataset(f"BeIR/{dataset_name}", "queries", split="queries")
    except Exception:
        queries_ds = None

    # qrels is Short for "Query Relevance judgments." This is the "Ground Truth" or the Answer Key. It maps which Documents (corpus-id) are actually relevant to which Queries (query-id).
    # A typical row in qrels looks like: {"query-id": "Q1", "corpus-id": "D5", "score": 1}. This tells the system that Document D5 is a correct answer for Query Q1.
    try:
        qrels_ds = load_dataset(f"BeIR/{dataset_name}-qrels", split="test")
    except Exception:
        """eg: [
            {"query-id": "101", "corpus-id": "doc_882", "score": 1},
            {"query-id": "101", "corpus-id": "doc_450", "score": 1},
            {"query-id": "102", "corpus-id": "doc_12",  "score": 1},
            {"query-id": "103", "corpus-id": "doc_99",  "score": 0},
            ...]
        """
        qrels_ds = load_dataset(f"BeIR/{dataset_name}-qrels", split="validation")

    """
    eg:
    {
    "MED-10": "Dietary fiber and whole grains in relation to risk of heart disease...",
    "MED-245": "Recent studies show that Vitamin D supplementation may improve...",
    "MED-882": "The effects of green tea catechins on weight loss and weight maintenance...",
    "doc_450": "A randomized controlled trial investigating the impact of Mediterranean diet..."
    }
    """
    doc_lookup = {row["_id"]: row["text"] for row in corpus_ds}

    if queries_ds is not None:
        """
        {
            "101": "how does dietary fiber affect heart health?",
            "102": "benefits of vitamin d for elderly",
            "103": "green tea for weight loss research",
            "query_99": "is the mediterranean diet good for heart disease?"
        }
        """
        query_lookup = {row["_id"]: row["text"] for row in queries_ds}
    else:
        query_lookup = {}
        for row in corpus_ds:
            title = row.get("title", "")
            text = row.get("text", "")
            q_text = title if title else text[:200]
            query_lookup[row["_id"]] = q_text
            try:
                query_lookup[int(row["_id"])] = q_text
            except (ValueError, TypeError):
                pass

    print(f"  Corpus: {len(doc_lookup)} docs | Queries: {len(query_lookup)} | QRels: {len(qrels_ds)}")
    return doc_lookup, query_lookup, qrels_ds


def build_train_eval_pairs(
    doc_lookup,
    query_lookup,
    qrels_ds,
    max_train=MAX_TRAIN_PAIRS,
    max_eval=MAX_SERVER_VAL_PAIRS + MAX_FINAL_TEST_PAIRS + MAX_SHARED_QUALITY_PAIRS,
    seed=SEED,
):
    """
    Constructs query-response pairs from the dataset for training and evaluation.

    Args:
        doc_lookup (dict): Mapping from doc_id to document text.
        query_lookup (dict): Mapping from query_id to query text.
        qrels_ds (iterable): Relevance judgments containing query-doc mappings.
        max_train (int): Maximum number of training pairs to generate.
        max_eval (int): Maximum number of held-out pairs to generate.
        seed (int): Random seed for reproducibility during shuffling.

    Returns:
        tuple: (train_pairs, heldout_pairs) - Lists of dictionaries containing query and response text.
    """
    pairs = []
    seen = set()
    for row in qrels_ds:
        qid = row["query-id"]
        did = row["corpus-id"]
        score = row["score"]
        if score > 0 and (qid, did) not in seen:
            q_text = query_lookup.get(qid) or query_lookup.get(str(qid))
            d_text = doc_lookup.get(did) or doc_lookup.get(str(did))
            if q_text and d_text:
                d_text_truncated = d_text[:MAX_RESPONSE_CHARS]
                doc_id_str = str(did)
                pairs.append({"query": q_text, "response": d_text_truncated, "query_id": qid, "doc_id": doc_id_str})
                seen.add((qid, did))
    """
    eg pairs:
    [
        {
            "query": "how does dietary fiber affect heart health?",
            "response": "Dietary fiber and whole grains in relation to risk of heart disease...",
            "query_id": "101",
            "doc_id": "MED-10"
        },
        {
            "query": "benefits of vitamin d for elderly",
            "response": "Recent studies show that Vitamin D supplementation may improve...",
            "query_id": "102",
            "doc_id": "MED-245"
        },
        {
            "query": "green tea for weight loss research",
            "response": "The effects of green tea catechins on weight loss and weight maintenance...",
            "query_id": "103",
            "doc_id": "MED-882"
        },
        {
            "query": "is the mediterranean diet good for heart disease?",
            "response": "A randomized controlled trial investigating the impact of Mediterranean diet...",
            "query_id": "query_99",
            "doc_id": "doc_450"
        }
    ]
    """
    rng = random.Random(seed)
    rng.shuffle(pairs)

    total = min(len(pairs), max_train + max_eval)
    heldout_pairs = pairs[:max_eval]
    train_pairs = pairs[max_eval:total]

    print(f"  Built {len(train_pairs)} train pairs + {len(heldout_pairs)} held-out pairs")
    return train_pairs, heldout_pairs


def build_knowledge_store(doc_lookup, retriever, max_docs=MAX_CORPUS_DOCS):
    """
    Initializes an in-memory knowledge store and populates it with document embeddings.

    Args:
        doc_lookup (dict): Mapping from doc_id to document text.
        retriever (BaseRetriever): The retriever model used to generate embeddings.
        max_docs (int): Maximum number of documents to load and embed.

    Returns:
        InMemoryKnowledgeStore: The populated knowledge store.
    """
    print(f"📚 Building knowledge store (max {max_docs} docs)...")

    store = InMemoryKnowledgeStore()
    nodes = []

    docs = list(doc_lookup.items())[:max_docs]
    for idx, (doc_id, text) in enumerate(docs):
        # truncate to fit the context window of the retriever model
        text_truncated = text[:512]
        # encode_context returns a numpy array of shape (1, embedding_dim)
        embedding = retriever.encode_context(text_truncated)[0].tolist()
        node = KnowledgeNode(
            node_type=NodeType.TEXT,
            embedding=embedding,
            text_content=text_truncated,
            metadata={"doc_id": doc_id, "idx": idx},
        )
        nodes.append(node)

    store.load_nodes(nodes)
    print(f"  ✅ Loaded {store.count} nodes into knowledge store")
    return store


def create_retriever(model_name: str | None = None):
    """
    Instantiates the default SentenceTransformer retriever for the project.

    Returns:
        HFSentenceTransformerRetriever: An initialized retriever instance.
    """
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Use separate query/context encoders so federated training updates only the
    # query tower while the knowledge-store document embeddings remain valid.
    return HFSentenceTransformerRetriever(
        query_model_name=model_name or RETRIEVER_MODEL,
        context_model_name=model_name or RETRIEVER_MODEL,
        load_model_kwargs={"device": device},
    )


def evaluate_retriever(retriever, knowledge_store, eval_pairs, top_k=TOP_K):
    """
    Computes IR metrics (MRR, Recall, NDCG) for the retriever against a knowledge store.

    Args:
        retriever (BaseRetriever): The retriever to evaluate.
        knowledge_store (BaseKnowledgeStore): The store contains the document candidates.
        eval_pairs (list): List of (query, relevant_doc) pairs for testing.
        top_k (int): Number of top results to consider for metric calculation.

    Returns:
        dict: A dictionary containing 'mrr', 'recall_at_k', and 'ndcg_at_k'.
    """
    if not eval_pairs:
        return {"mrr": 0.0, "recall_at_k": 0.0, "ndcg_at_k": 0.0}

    import math

    mrr_total = 0.0
    recall_total = 0.0
    ndcg_total = 0.0

    for pair in eval_pairs:
        query_text = pair["query"]
        relevant_doc_id = pair["doc_id"]

        query_emb = retriever.encode_query(query_text)[0].tolist()
        results = knowledge_store.retrieve(query_emb, top_k=top_k)

        retrieved_ids = [node.metadata.get("doc_id", "") for _score, node in results]

        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id == relevant_doc_id:
                mrr_total += 1.0 / rank
                break

        if relevant_doc_id in retrieved_ids:
            recall_total += 1.0

        dcg = 0.0
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id == relevant_doc_id:
                dcg += 1.0 / math.log2(rank + 1)
        idcg = 1.0 / math.log2(2)
        ndcg_total += dcg / idcg if idcg > 0 else 0.0

    n = len(eval_pairs)
    return {
        "mrr": mrr_total / n,
        "recall_at_k": recall_total / n,
        "ndcg_at_k": ndcg_total / n,
    }


def filter_eval_pairs_by_store(eval_pairs, knowledge_store):
    """
    Filters evaluation pairs to ensure that the ground-truth documents actually exist in the knowledge store.
    This prevents recall from being unfairly penalized if the store is a subset of the full corpus.

    Args:
        eval_pairs (list): Original list of evaluation pairs.
        knowledge_store (BaseKnowledgeStore): The store to check against.

    Returns:
        list: Filtered list of evaluation pairs.
    """
    store_doc_ids = set()
    for node_id, node in knowledge_store._data.items():
        doc_id = node.metadata.get("doc_id", "")
        if doc_id:
            store_doc_ids.add(doc_id)

    filtered = [p for p in eval_pairs if p["doc_id"] in store_doc_ids]
    print(f"  🔍 Filtered eval pairs: {len(eval_pairs)} → {len(filtered)} (only docs in knowledge store)")
    return filtered


def setup_dataset(
    dataset_name="nfcorpus",
    max_train=MAX_TRAIN_PAIRS,
    max_server_val=MAX_SERVER_VAL_PAIRS,
    max_final_test=MAX_FINAL_TEST_PAIRS,
    max_shared_quality=MAX_SHARED_QUALITY_PAIRS,
    max_docs=MAX_CORPUS_DOCS,
    seed=SEED,
    retriever_model: str | None = None,
):
    """
    Orchestrates the entire data preparation pipeline: loading, pairing, embedding, and filtering.

    Args:
        dataset_name (str): Name of the BEIR dataset to use.
        max_train (int): Maximum number of training samples.
        max_server_val (int): Maximum number of validation samples used for best-round selection.
        max_final_test (int): Maximum number of held-out test samples used for final reporting.
        max_shared_quality (int): Maximum shared clean comparison pairs reserved for weighting.
        max_docs (int): Maximum documents in the knowledge store.
        seed (int): Seed for random operations.

    Returns:
        dict: A dictionary containing the retriever, knowledge_store, training dataset,
              and the underlying pairs for reference.
    """
    doc_lookup, query_lookup, qrels_ds = load_beir_dataset(dataset_name)
    total_holdout = max_shared_quality + max_server_val + max_final_test
    train_pairs, heldout_pairs = build_train_eval_pairs(
        doc_lookup,
        query_lookup,
        qrels_ds,
        max_train,
        total_holdout,
        seed=seed,
    )
    retriever = create_retriever(retriever_model)
    knowledge_store = build_knowledge_store(doc_lookup, retriever, max_docs)

    heldout_pairs = filter_eval_pairs_by_store(heldout_pairs, knowledge_store)
    shared_quality_pairs = heldout_pairs[:max_shared_quality]
    offset = max_shared_quality
    server_val_pairs = heldout_pairs[offset:offset + max_server_val]
    offset += max_server_val
    final_test_pairs = heldout_pairs[offset:offset + max_final_test]

    print(
        "  🧪 Shared quality pairs: "
        f"{len(shared_quality_pairs)} | "
        f"Server val pairs: {len(server_val_pairs)} | "
        f"Final test pairs: {len(final_test_pairs)}"
    )

    train_dataset = Dataset.from_dict({
        "query": [p["query"] for p in train_pairs],
        "response": [p["response"] for p in train_pairs],
    })

    return {
        "retriever": retriever,
        "knowledge_store": knowledge_store,
        "train_dataset": train_dataset,
        "train_pairs": train_pairs,
        "shared_quality_pairs": shared_quality_pairs,
        "server_val_pairs": server_val_pairs,
        "final_test_pairs": final_test_pairs,
        "doc_lookup": doc_lookup,
        "retriever_model": retriever_model or RETRIEVER_MODEL,
    }


if __name__ == "__main__":
    data = setup_dataset(
        "nfcorpus",
        max_train=20,
        max_shared_quality=5,
        max_server_val=5,
        max_final_test=5,
        max_docs=50,
    )
    print(f"\nTrain dataset size: {len(data['train_dataset'])}")
    print(f"Server val pairs: {len(data['server_val_pairs'])}")
    print(f"Final test pairs: {len(data['final_test_pairs'])}")

    metrics = evaluate_retriever(
        data["retriever"],
        data["knowledge_store"],
        data["final_test_pairs"],
    )
    print(f"\nBaseline metrics (before training):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
