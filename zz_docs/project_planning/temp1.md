
# OVERVIEW

## 1. LLM (Large Language Model)

```text
┌─────────────────────────────────────────────────────────────┐
│                     LARGE LANGUAGE MODEL                    │
│                                                             │
│  ┌──────────────┐         ┌──────────────┐                  │
│  │   Training   │         │  Inference   │                  │
│  │     Data     │  ────>  │   (Query)    │                  │
│  │ (Wikipedia,  │         │              │                  │
│  │  Books, Web) │         │   "What is   │                  │
│  └──────────────┘         │   the cure   │                  │
│                           │   for flu?"  │                  │
│                           └──────┬───────┘                  │
│                                  │                          │
│                                  ▼                          │
│                           ┌──────────────┐                  │
│                           │   Response   │                  │
│                           │  Generated   │                  │
│                           │     from     │                  │
│                           │  PARAMETRIC  │                  │
│                           │    MEMORY    │                  │
│                           └──────────────┘                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
Key Point: LLM has knowledge frozen at training time, stored in weights.

```

## 2. Why This is Not Enough (Intro to RAG)

**PROBLEMS WITH LLM-ONLY APPROACH**

**Problem 1: OUTDATED KNOWLEDGE**

```text
┌────────────────────────────────────────┐
│ LLM trained in 2023                    │
│                                        │
│ User (2026): "Who won 2025 election?"  │
│ LLM: "I don't know, my knowledge       │
│       cutoff is April 2023"            │
└────────────────────────────────────────┘

```

**Problem 2: HALLUCINATION**

```text
┌────────────────────────────────────────┐
│ User: "What's Drug X dosage for kids?" │
│                                        │
│ LLM: "50mg twice daily"                │
│      ↑                                 │
│      MADE UP! (dangerous)              │
│                                        │
│ Actual dosage: 10mg once daily         │
└────────────────────────────────────────┘

```

**Problem 3: DOMAIN-SPECIFIC KNOWLEDGE**

```text
┌────────────────────────────────────────┐
│ User: "Explain our company's Q3        │
│        financial policy"               │
│                                        │
│ LLM: "I don't have access to your      │
│       internal documents"              │
└────────────────────────────────────────┘

```

**SOLUTION: RAG (Retrieval-Augmented Generation)**

```text
┌──────────────────────────────────────────────────────────┐
│                        RAG SYSTEM                        │
│                                                          │
│  User Query                                              │
│      │                                                   │
│      ▼                                                   │
│  ┌─────────────┐                                         │
│  │  RETRIEVER  │──────> Search in                        │
│  │  (finds     │        EXTERNAL                         │
│  │  relevant   │        KNOWLEDGE                        │
│  │  docs)      │        ┌──────────────┐                 │
│  └─────────────┘        │ 2025 News    │                 │
│      │                  │ Drug Database│                 │
│      │                  │ Company Docs │                 │
│      │                  └──────────────┘                 │
│      │ Retrieved                                         │
│      │ Context                                           │
│      ▼                                                   │
│  ┌─────────────┐                                         │
│  │  GENERATOR  │                                         │
│  │  (LLM with  │                                         │
│  │  retrieved  │                                         │
│  │  context)   │                                         │
│  └─────────────┘                                         │
│      │                                                   │
│      ▼                                                   │
│  Response (grounded in retrieved facts)                  │
└──────────────────────────────────────────────────────────┘

Now LLM has access to:
✓ Current information
✓ Verified facts from knowledge base
✓ Domain-specific documents

```

## 3. RAG Fine-Tuning

**BEFORE FINE-TUNING: Components don't work well together**

```text
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  Query: "Treatment for diabetes?"                       │
│                                                         │
│  ┌──────────────┐          ┌─────────────────┐          │
│  │  RETRIEVER   │          │  Retrieved Docs │          │
│  │  (Generic)   │   ─────> │                 │          │
│  │              │          │    "History of  │          │
│  │  Not trained │          │    insulin"     │          │
│  │  for medical │          │    "Diabetes    │          │
│  │  domain      │          │    statistics"  │          │
│  └──────────────┘          └─────────────────┘          │
│                                    │                    │
│                                    ▼                    │
│  ┌──────────────────────────────────────────┐           │
│  │         GENERATOR (Generic LLM)          │           │
│  │                                          │           │
│  │  Receives context but...                 │           │
│  │     Ignores retrieved docs               │           │
│  │     Generates from memory instead        │           │
│  │                                          │           │
│  │  Response: "Diabetes is managed with     │           │
│  │           diet and exercise"             │           │
│  │           (ignoring retrieved treatment  │           │
│  │            protocols!)                   │           │
│  └──────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────┘

```

**AFTER FINE-TUNING: Components work as a team**

```text
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  Query: "Treatment for diabetes?"                       │
│                                                         │
│  ┌──────────────┐          ┌─────────────────┐          │
│  │  RETRIEVER   │          │  Retrieved Docs │          │
│  │ (Fine-tuned  │   ─────> │                 │          │
│  │  with LSR)   │          │    "Metformin   │          │
│  │              │          │    protocol"    │          │
│  │  Learns what │          │    "Insulin     │          │
│  │  helps LLM   │          │    dosing"      │          │
│  └──────────────┘          └─────────────────┘          │
│                                    │                    │
│                                    ▼                    │
│  ┌──────────────────────────────────────────┐           │
│  │       GENERATOR (Fine-tuned with RALT)   │           │
│  │                                          │           │
│  │  Trained to use retrieved context:       │           │
│  │     Cites retrieved documents            │           │
│  │     Grounds response in context          │           │
│  │                                          │           │
│  │  Response: "Based on the retrieved       │           │
│  │           protocols, first-line          │           │
│  │           treatment is Metformin         │           │
│  │           500mg, titrated to 2000mg..."  │           │
│  └──────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────┘

FINE-TUNING METHODS:
- RALT: Train generator to use retrieved context
- LSR:  Train retriever based on what helps generator
- RA-DIT: Both in sequence → optimal performance

```

## 4. Why Federation?

**SCENARIO: Multiple Hospitals Want RAG Systems**

```text
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Hospital A  │  │  Hospital B  │  │  Hospital C  │
│              │  │              │  │              │
│  10K patient │  │  15K patient │  │  8K patient  │
│  records     │  │  records     │  │  records     │
│  (PRIVATE)   │  │  (PRIVATE)   │  │  (PRIVATE)   │
└──────────────┘  └──────────────┘  └──────────────┘

```

**CENTRALIZED APPROACH (Can't do this!)**

```text
┌─────────────────────────────────────────────────────┐
│                                                     │
│  Hospital A ──┐                                     │
│               │   Send all patient                  │
│  Hospital B ──┼─> data to central        ILLEGAL!   │
│               │   server                 HIPAA      │
│  Hospital C ──┘                          Privacy    │
│                                                     │
│          ┌──────────────────┐                       │
│          │  Central Server  │                       │
│          │  trains on ALL   │                       │
│          │  hospital data   │                       │
│          └──────────────────┘                       │
└─────────────────────────────────────────────────────┘

```

**FEDERATED LEARNING APPROACH (Legal & Private!)**

```text
┌─────────────────────────────────────────────────────┐
│                                                     │
│  Hospital A        Hospital B        Hospital C     │
│  ┌──────────┐      ┌──────────┐      ┌──────────┐   │
│  │  Train   │      │  Train   │      │  Train   │   │
│  │  LOCALLY │      │  LOCALLY │      │  LOCALLY │   │
│  │          │      │          │      │          │   │
│  │  Data    │      │  Data    │      │  Data    │   │
│  │  STAYS   │      │  STAYS   │      │  STAYS   │   │
│  │  HERE!   │      │  HERE!   │      │  HERE!   │   │
│  └────┬─────┘      └────┬─────┘      └────┬─────┘   │
│       │                 │                 │         │
│       │ Only send       │ Only send       │ Only send
│       │ MODEL           │ MODEL           │ MODEL   │
│       │ UPDATES         │ UPDATES         │ UPDATES │
│       │ (not data!)     │ (not data!)     │ (not data
│       ▼                 ▼                 ▼         │
│  ┌────────────────────────────────────────────┐     │
│  │          CENTRAL AGGREGATION SERVER        │     │
│  │                                            │     │
│  │  Combines model updates from all hospitals │     │
│  │  WITHOUT seeing patient data               │     │
│  │                                            │     │
│  │  global_model = avg(updates_A, updates_B,  │     │
│  │                     updates_C)             │     │
│  └────────────────────────────────────────────┘     │
│       │                                             │
│       │ Send improved global model back             │
│       ▼                                             │
│  All hospitals get better model trained on          │
│  33K total patients, but data never left hospitals! │
└─────────────────────────────────────────────────────┘

BENEFITS:
   Privacy: Data never leaves organization
   Legal: Complies with HIPAA, GDPR
   Better models: Learn from distributed data
   Collaboration: Organizations benefit without sharing secrets

```

## 5. Intro to Flower (Federated Learning Framework)

**FLOWER: Framework for Federated Learning**

```text
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  Flower provides abstraction for federated learning:    │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │                FLOWER SERVER                     │   │
│  │                                                  │   │
│  │  - Orchestrates federated rounds                 │   │
│  │  - Aggregates client updates (FedAvg)            │   │
│  │  - Broadcasts global model                       │   │
│  │                                                  │   │
│  │  class Server:                                   │   │
│  │    def aggregate(client_updates):                │   │
│  │      return avg(client_updates)                  │   │
│  └──────────────────────────────────────────────────┘   │
│                          │                              │
│                          │ Communication                │
│                          │ (gRPC)                       │
│            ┌─────────────┼─────────────┐                │
│            │             │             │                │
│  ┌────────▼───────┐ ┌──▼──────────┐ ┌▼─────────────┐    │
│  │ FLOWER CLIENT  │ │ FLOWER      │ │ FLOWER       │    │
│  │      #1        │ │ CLIENT #2   │ │ CLIENT #3    │    │
│  │                │ │             │ │              │    │
│  │ - Gets model   │ │ - Gets model│ │ - Gets model │    │
│  │ - Trains local │ │ - Trains    │ │ - Trains     │    │
│  │ - Sends updates│ │ - Sends     │ │ - Sends      │    │
│  │                │ │             │ │              │    │
│  │ class Client:  │ │             │ │              │    │
│  │    def fit():  │ │             │ │              │    │
│  │      train_on_ │ │             │ │              │    │
│  │      local_data│ │             │ │              │    │
│  └────────────────┘ └─────────────┘ └──────────────┘    │
│                                                         │
└─────────────────────────────────────────────────────────┘

```

**FEDRAG INTEGRATION WITH FLOWER:**

```text
┌─────────────────────────────────────────────────────────┐
│  FedRAG abstracts FL complexity:                        │
│                                                         │
│  manager = HuggingFaceRAGTrainerManager(...)            │
│  fl_task = manager.get_federated_task()  # ← Magic!     │
│                                                         │
│  # FedRAG creates Flower components automatically       │
│  server = fl_task.server(model)                         │
│  client = fl_task.client(model, dataset)                │
│                                                         │
│  # Standard Flower training                             │
│  fl.simulation.start_simulation(                        │
│       server=server,                                    │
│       clients=[client1, client2, ...]                   │
│  )                                                      │
└─────────────────────────────────────────────────────────┘

KEY POINT: Flower handles communication, FedRAG handles RAG-specific logic

```

## Complete RAG Data Flow

```
"What are tulips?"
        │
        ▼
retriever.encode_query()
        │  → torch.Tensor or EncodeResult["text"]
        │  → .tolist() → list[float]  (query embedding)
        ▼
knowledge_store.retrieve(query_emb, top_k=2)
        │  → similarity search against pre-stored KnowledgeNode embeddings
        │  → list[tuple[float, KnowledgeNode]]
        ▼
[SourceNode(score=0.91, node=KnowledgeNode(text="tulips are flowers...")),
 SourceNode(score=0.87, node=KnowledgeNode(text="tulips originated in..."))]
        │
        ▼
_format_context()
        │  → "tulips are flowers...\ntulips originated in..."
        ▼
generator.generate(query="What are tulips?", context="tulips are flowers...")
        │  → prompt_template.format(query, context)
        │  → model inference
        │  → "Tulips are flowering plants native to..."
        ▼
RAGResponse(
    response="Tulips are flowering plants native to...",
    source_nodes=[SourceNode(...), SourceNode(...)],
    raw_response=None
)
```


## Complete End-to-End Flow Diagram

```
INDEXING (one-time setup)
──────────────────────────────────────────────────────────────────
text chunks
    → retriever.context_encoder.encode()
    → KnowledgeNode(embedding, text_content)
    → knowledge_store.load_nodes()


CENTRALIZED TRAINING
──────────────────────────────────────────────────────────────────
train_dataset [(query, response), ...]
    │
    ▼
HuggingFaceRAGTrainerManager.train()
    │
    ├─ mode="retriever" → LSR path:
    │       generator.eval()  [frozen]
    │       retriever.train() [active]
    │           ↓
    │       DataCollatorForLSR per batch:
    │           retrieve(query) → document selection (non-differentiable)
    │           → context_texts: list[str]  (just the text of retrieved docs)
    │           generator.compute_target_sequence_proba() → lm_scores [no_grad]
    │           ↓
    │       compute_loss() in LSRSentenceTransformerTrainer:
    │           model.forward(query) → query_embedding [grad=True]
    │           model.encode(context_texts) → context_embedding [no_grad]
    │           cosine_sim(query_emb, context_emb) → retrieval_scores [grad=True]
    │           ↓
    │       LSRLoss: KL(log_softmax(retrieval) ‖ softmax(lm))
    │           ↓
    │       backward() → encoder.params.grad populated
    │           ↓
    │       create_optimizer() → AdamW with model params (not loss params)
    │           ↓
    │       optimizer.step() → retriever weights updated ✅
    │
    └─ mode="generator" → RALT path:
            retriever.eval()  [frozen]
            generator.train() [active]
                ↓
            DataCollatorForRALT per batch:
                retrieve(query) → top_k source_nodes
                build top_k instruction examples with retrieved context
                tokenize → input_ids, labels, attention_mask (left-padded)
                ↓
            HuggingFace Trainer: causal LM cross-entropy loss
                ↓
            backprop → generator weights updated


FEDERATION (Manual FedAvg)
──────────────────────────────────────────────────────────────────
Initialize global_weights from base model

[Each Round]
    For each client:
        Create fresh RAG system + trainer
        Load global_weights into client model
            ↓
        Run full LSR training (same pipeline as centralized)
            → DataCollatorForLSR → compute_loss() → LSRLoss
            → backward() → optimizer.step()
            ↓
        Extract updated local_weights
        ↓
    FedAvg: global_weights = Σ(local_weights × dataset_size) / Σ(dataset_size)
        ↓
    Next round
```