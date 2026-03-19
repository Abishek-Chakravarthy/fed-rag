import random
from datasets import load_dataset, Dataset
from fed_rag.knowledge_stores import InMemoryKnowledgeStore
from fed_rag.retrievers import HFSentenceTransformerRetriever
from fed_rag.data_structures import KnowledgeNode, NodeType


RETRIEVER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_CORPUS_DOCS = 1000
MAX_TRAIN_PAIRS = 500
MAX_EVAL_PAIRS = 100
TOP_K = 10
SEED = 42
MAX_RESPONSE_CHARS = 500

#Load the beir dataset
def load_beir_dataset(dataset_name: str = "nfcorpus"):
    print(f"📥 Loading BEIR/{dataset_name}...")

    try:
        corpus_ds = load_dataset(f"BeIR/{dataset_name}", "corpus", split="corpus")
    except Exception:
        corpus_ds = load_dataset(f"BeIR/{dataset_name}", split="train")

    try:
        queries_ds = load_dataset(f"BeIR/{dataset_name}", "queries", split="queries")
    except Exception:
        queries_ds = None
    try:
        qrels_ds = load_dataset(f"BeIR/{dataset_name}-qrels", split="test")
    except Exception:
        qrels_ds = load_dataset(f"BeIR/{dataset_name}-qrels", split="validation")

    doc_lookup = {row["_id"]: row["text"] for row in corpus_ds}

    if queries_ds is not None:
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

def build_train_eval_pairs(doc_lookup, query_lookup, qrels_ds, max_train=MAX_TRAIN_PAIRS, max_eval=MAX_EVAL_PAIRS):
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

    random.seed(SEED)
    random.shuffle(pairs)

    total = min(len(pairs), max_train + max_eval)
    eval_pairs = pairs[:max_eval]
    train_pairs = pairs[max_eval:total]

    print(f"  Built {len(train_pairs)} train pairs + {len(eval_pairs)} eval pairs")
    return train_pairs, eval_pairs

def build_knowledge_store(doc_lookup, retriever, max_docs=MAX_CORPUS_DOCS):
    print(f"📚 Building knowledge store (max {max_docs} docs)...")

    store = InMemoryKnowledgeStore()
    nodes = []

    docs = list(doc_lookup.items())[:max_docs]
    for idx, (doc_id, text) in enumerate(docs):
        text_truncated = text[:512]
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


def create_retriever():
    return HFSentenceTransformerRetriever(
        model_name=RETRIEVER_MODEL,
    )

def evaluate_retriever(retriever, knowledge_store, eval_pairs, top_k=TOP_K):
    if not eval_pairs:
        return {"mrr": 0.0, "recall_at_k": 0.0, "ndcg_at_k": 0.0}

    import math

    mrr_total = 0.0
    recall_total = 0.0
    ndcg_total = 0.0

    for pair in eval_pairs:
        query_text = pair["query"]
        relevant_doc_id = pair["doc_id"]

        # Encode query and retrieve
        query_emb = retriever.encode_query(query_text)[0].tolist()
        results = knowledge_store.retrieve(query_emb, top_k=top_k)

        # Get retrieved doc IDs
        retrieved_ids = [node.metadata.get("doc_id", "") for _score, node in results]

        # MRR
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id == relevant_doc_id:
                mrr_total += 1.0 / rank
                break

        # Recall@k
        if relevant_doc_id in retrieved_ids:
            recall_total += 1.0

        # NDCG@k (binary relevance)
        dcg = 0.0
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id == relevant_doc_id:
                dcg += 1.0 / math.log2(rank + 1)
        idcg = 1.0 / math.log2(2)  # perfect: relevant doc at rank 1
        ndcg_total += dcg / idcg if idcg > 0 else 0.0

    n = len(eval_pairs)
    return {
        "mrr": mrr_total / n,
        "recall_at_k": recall_total / n,
        "ndcg_at_k": ndcg_total / n,
    }

def filter_eval_pairs_by_store(eval_pairs, knowledge_store):
    store_doc_ids = set()
    for node_id, node in knowledge_store._data.items():
        doc_id = node.metadata.get("doc_id", "")
        if doc_id:
            store_doc_ids.add(doc_id)

    filtered = [p for p in eval_pairs if p["doc_id"] in store_doc_ids]
    print(f"  🔍 Filtered eval pairs: {len(eval_pairs)} → {len(filtered)} (only docs in knowledge store)")
    return filtered


def setup_dataset(dataset_name="nfcorpus", max_train=MAX_TRAIN_PAIRS, max_eval=MAX_EVAL_PAIRS, max_docs=MAX_CORPUS_DOCS):
    doc_lookup, query_lookup, qrels_ds = load_beir_dataset(dataset_name)
    train_pairs, eval_pairs = build_train_eval_pairs(doc_lookup, query_lookup, qrels_ds, max_train, max_eval * 5)
    retriever = create_retriever()
    knowledge_store = build_knowledge_store(doc_lookup, retriever, max_docs)

    eval_pairs = filter_eval_pairs_by_store(eval_pairs, knowledge_store)
    eval_pairs = eval_pairs[:max_eval]

    train_dataset = Dataset.from_dict({
        "query": [p["query"] for p in train_pairs],
        "response": [p["response"] for p in train_pairs],
    })

    return {
        "retriever": retriever,
        "knowledge_store": knowledge_store,
        "train_dataset": train_dataset,
        "train_pairs": train_pairs,
        "eval_pairs": eval_pairs,
        "doc_lookup": doc_lookup,
    }


if __name__ == "__main__":
    data = setup_dataset("nfcorpus", max_train=20, max_eval=5, max_docs=50)
    print(f"\nTrain dataset size: {len(data['train_dataset'])}")
    print(f"Eval pairs: {len(data['eval_pairs'])}")

    metrics = evaluate_retriever(data["retriever"], data["knowledge_store"], data["eval_pairs"])
    print(f"\nBaseline metrics (before training):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
