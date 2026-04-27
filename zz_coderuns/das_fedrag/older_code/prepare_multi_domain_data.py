import hashlib
import json
import os
import random
import numpy as np
import torch
from datasets import load_dataset
from fed_rag.knowledge_stores import InMemoryKnowledgeStore
from fed_rag.retrievers import HFSentenceTransformerRetriever
from fed_rag.data_structures import KnowledgeNode, NodeType


RETRIEVER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_CORPUS_DOCS = 8000
MAX_TRAIN_PAIRS = 4000
MAX_SERVER_VAL_PAIRS = 400
MAX_FINAL_TEST_PAIRS = 500
TOP_K = 10
SEED = 42
MAX_RESPONSE_CHARS = 500
MAX_QUERY_CHARS = 300
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".centroid_cache")

DEFAULT_CLIENT_CONFIGS = [
    {"dataset_name": "nfcorpus", "label": "medical"},
    {"dataset_name": "scifact", "label": "science"},
    {"dataset_name": "fiqa", "label": "finance"},
    {"dataset_name": "arguana", "label": "argument"},
    {"dataset_name": "trec-covid", "label": "biomedical"},
]


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def create_retriever():
    return HFSentenceTransformerRetriever(
        query_model_name=RETRIEVER_MODEL,
        context_model_name=RETRIEVER_MODEL,
        load_model_kwargs={"device": get_device()},
    )


def load_beir_dataset(dataset_name: str):
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


def build_positive_pairs(
    doc_lookup,
    query_lookup,
    qrels_ds,
    seed=SEED,
):
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
                q_text_truncated = q_text[:MAX_QUERY_CHARS]
                d_text_truncated = d_text[:MAX_RESPONSE_CHARS]
                doc_id_str = str(did)
                pairs.append({
                    "query": q_text_truncated,
                    "response": d_text_truncated,
                    "query_id": qid,
                    "doc_id": doc_id_str,
                })
                seen.add((qid, did))

    rng = random.Random(seed)
    rng.shuffle(pairs)
    print(f"  Built {len(pairs)} positive pairs before splitting")
    return pairs


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
    print(f"  ✅ Loaded {store.count} nodes")
    return store


def filter_eval_pairs_by_store(eval_pairs, knowledge_store):
    store_doc_ids = set()
    for node_id, node in knowledge_store._data.items():
        doc_id = node.metadata.get("doc_id", "")
        if doc_id:
            store_doc_ids.add(doc_id)

    filtered = [p for p in eval_pairs if p["doc_id"] in store_doc_ids]
    print(f"  🔍 Filtered eval: {len(eval_pairs)} → {len(filtered)}")
    return filtered


def split_pairs_with_caps(
    pairs,
    *,
    max_train=MAX_TRAIN_PAIRS,
    max_server_val=MAX_SERVER_VAL_PAIRS,
    max_final_test=MAX_FINAL_TEST_PAIRS,
):
    total = len(pairs)
    if total == 0:
        return [], [], []

    train_target = min(max_train, max(1, int(total * 0.7)))
    remaining = max(0, total - train_target)

    server_val_target = min(max_server_val, max(1 if remaining > 1 else 0, int(total * 0.15)))
    server_val_target = min(server_val_target, remaining)
    remaining -= server_val_target

    final_test_target = min(
        max_final_test,
        max(1 if remaining > 0 else 0, int(total * 0.15)),
    )
    final_test_target = min(final_test_target, remaining)
    remaining -= final_test_target

    train_count = min(train_target, total - server_val_target - final_test_target)
    if train_count <= 0:
        train_count = max(1, total - max(1 if total > 1 else 0, final_test_target))
        remaining_after_train = total - train_count
        server_val_target = min(server_val_target, remaining_after_train)
        final_test_target = min(final_test_target, remaining_after_train - server_val_target)

    train_pairs = pairs[:train_count]
    server_val_pairs = pairs[train_count:train_count + server_val_target]
    final_test_pairs = pairs[
        train_count + server_val_target:
        train_count + server_val_target + final_test_target
    ]

    return train_pairs, server_val_pairs, final_test_pairs


def compute_domain_centroid(knowledge_store) -> np.ndarray:
    embeddings = []
    for _, node in knowledge_store._data.items():
        embeddings.append(np.array(node.embedding, dtype=np.float32))
    centroid = np.mean(embeddings, axis=0)
    return centroid / np.linalg.norm(centroid)


def compute_target_centroid(knowledge_store) -> np.ndarray:
    # Use the same document-space centroid definition for both clients and the
    # target domain so cosine similarity is a same-modality comparison.
    return compute_domain_centroid(knowledge_store)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def build_cache_key(dataset_name: str, max_docs: int) -> str:
    payload = {
        "dataset_name": dataset_name,
        "retriever_model": RETRIEVER_MODEL,
        "max_docs": max_docs,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.md5(encoded).hexdigest()


def load_cached_centroid(dataset_name: str, max_docs: int) -> np.ndarray | None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{build_cache_key(dataset_name, max_docs)}.npy")
    if not os.path.exists(cache_path):
        return None
    centroid = np.load(cache_path)
    return np.array(centroid, dtype=np.float32)


def save_cached_centroid(dataset_name: str, max_docs: int, centroid: np.ndarray) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{build_cache_key(dataset_name, max_docs)}.npy")
    np.save(cache_path, centroid.astype(np.float32))


def evaluate_retriever(retriever, knowledge_store, eval_pairs, top_k=TOP_K):
    if not eval_pairs:
        return {"mrr": 0.0, "recall_at_k": 0.0, "ndcg_at_k": 0.0}

    import math

    mrr_total = 0.0
    recall_total = 0.0
    ndcg_total = 0.0

    for pair in eval_pairs:
        query_emb = retriever.encode_query(pair["query"])[0].tolist()
        results = knowledge_store.retrieve(query_emb, top_k=top_k)
        retrieved_ids = [node.metadata.get("doc_id", "") for _score, node in results]

        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id == pair["doc_id"]:
                mrr_total += 1.0 / rank
                break

        if pair["doc_id"] in retrieved_ids:
            recall_total += 1.0

        dcg = 0.0
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id == pair["doc_id"]:
                dcg += 1.0 / math.log2(rank + 1)
        idcg = 1.0 / math.log2(2)
        ndcg_total += dcg / idcg if idcg > 0 else 0.0

    n = len(eval_pairs)
    return {
        "mrr": mrr_total / n,
        "recall_at_k": recall_total / n,
        "ndcg_at_k": ndcg_total / n,
    }


def load_client_data(
    dataset_name: str,
    label: str,
    retriever,
    max_train=MAX_TRAIN_PAIRS,
    max_server_val=MAX_SERVER_VAL_PAIRS,
    max_final_test=MAX_FINAL_TEST_PAIRS,
    max_docs=MAX_CORPUS_DOCS,
    seed=SEED,
):
    print(f"\n{'─' * 50}")
    print(f"Loading client data: {dataset_name} ({label})")
    print(f"{'─' * 50}")

    doc_lookup, query_lookup, qrels_ds = load_beir_dataset(dataset_name)
    knowledge_store = build_knowledge_store(doc_lookup, retriever, max_docs)
    all_pairs = build_positive_pairs(
        doc_lookup, query_lookup, qrels_ds, seed=seed
    )
    eligible_pairs = filter_eval_pairs_by_store(all_pairs, knowledge_store)
    train_pairs, server_val_pairs, final_test_pairs = split_pairs_with_caps(
        eligible_pairs,
        max_train=max_train,
        max_server_val=max_server_val,
        max_final_test=max_final_test,
    )

    domain_centroid = load_cached_centroid(dataset_name, max_docs)
    if domain_centroid is None:
        domain_centroid = compute_domain_centroid(knowledge_store)
        save_cached_centroid(dataset_name, max_docs, domain_centroid)

    print(f"  Domain centroid norm: {np.linalg.norm(domain_centroid):.4f}")
    print(
        f"  Split sizes | train={len(train_pairs)} | server_val={len(server_val_pairs)} | "
        f"final_test={len(final_test_pairs)}"
    )

    return {
        "dataset_name": dataset_name,
        "label": label,
        "doc_lookup": doc_lookup,
        "train_pairs": train_pairs,
        "server_val_pairs": server_val_pairs,
        "final_test_pairs": final_test_pairs,
        "knowledge_store": knowledge_store,
        "domain_centroid": domain_centroid,
    }


def setup_multi_domain_experiment(
    client_configs=None,
    target_dataset="nfcorpus",
    max_train=MAX_TRAIN_PAIRS,
    max_server_val=MAX_SERVER_VAL_PAIRS,
    max_final_test=MAX_FINAL_TEST_PAIRS,
    max_docs=MAX_CORPUS_DOCS,
    seed=SEED,
):
    if client_configs is None:
        client_configs = DEFAULT_CLIENT_CONFIGS

    retriever = create_retriever()

    client_data = {}
    for idx, config in enumerate(client_configs):
        cid = str(idx)
        data = load_client_data(
            dataset_name=config["dataset_name"],
            label=config["label"],
            retriever=retriever,
            max_train=max_train,
            max_server_val=max_server_val,
            max_final_test=max_final_test,
            max_docs=max_docs,
            seed=seed,
        )
        client_data[cid] = data

    target_cid = None
    for cid, data in client_data.items():
        if data["dataset_name"] == target_dataset:
            target_cid = cid
            break

    if target_cid is None:
        raise ValueError(
            f"Target dataset '{target_dataset}' not found in client configs. "
            f"Available: {[c['dataset_name'] for c in client_configs]}"
        )

    target_data = client_data[target_cid]
    target_centroid = compute_target_centroid(target_data["knowledge_store"])

    relevance_scores = {}
    for cid, data in sorted(client_data.items()):
        score = cosine_similarity(data["domain_centroid"], target_centroid)
        relevance_scores[cid] = score

    print(f"\n{'=' * 60}")
    print(f"Domain Relevance Scores (target: {target_dataset})")
    print(f"{'=' * 60}")
    for cid in sorted(relevance_scores, key=lambda c: relevance_scores[c], reverse=True):
        data = client_data[cid]
        score = relevance_scores[cid]
        marker = " ← TARGET" if cid == target_cid else ""
        print(f"  Client {cid} ({data['label']:>10}): d = {score:.4f}{marker}")
    print(f"{'=' * 60}")

    return {
        "retriever": retriever,
        "client_data": client_data,
        "target_cid": target_cid,
        "target_centroid": target_centroid,
        "relevance_scores": relevance_scores,
    }


if __name__ == "__main__":
    result = setup_multi_domain_experiment(
        max_train=20,
        max_server_val=5,
        max_final_test=5,
        max_docs=50,
    )
    print("\nRelevance scores:", result["relevance_scores"])
    print("Target client:", result["target_cid"])
