# 01 — Project Overview

## Title
Quality-Aware and Domain-Aware Federated Learning for Retrieval-Augmented Generation

## Abstract / Summary
This project addresses a critical limitation in Federated Learning (FL) for Retrieval-Augmented Generation (RAG) systems. Standard FL aggregation (FedAvg) weights client contributions purely by dataset size, which fails when clients have noisy, low-quality, or domain-irrelevant training data—leading to degraded global model performance. We propose **Quality-Aware Federated Aggregation (QA-FedAvg)**, which replaces the naive weighting scheme with a combination of dataset size and an inverse-loss quality score. This allows the federation to automatically detect and down-weight poorly performing clients. Experiments on the NFCorpus (Medical) dataset show that QA-FedAvg at α=0.6 achieves a **7.3% reduction in global training loss** compared to standard FedAvg, while reducing the noisy client's aggregation influence by 32%.

## 1. Introduction & Motivation
Retrieval-Augmented Generation (RAG) enhances Large Language Models by grounding their responses in factual documents retrieved from a knowledge store. A RAG system consists of three core components:
- **Retriever**: An encoder model (e.g., SentenceTransformer) that embeds queries and documents, then performs similarity search to find the top-k most relevant documents.
- **Generator**: An LLM (e.g., GPT-2) that produces a response conditioned on the query and the retrieved context.
- **Knowledge Store**: A vector database storing pre-embedded document chunks.

Fine-tuning RAG systems—particularly the retriever—requires large, diverse training datasets of (query, relevant_document) pairs. In practice, this data is distributed across organizations (hospitals, enterprises, research labs) that **cannot share raw data** due to privacy regulations (HIPAA, GDPR) or competitive concerns.

**Federated Learning (FL)** solves this by enabling collaborative model training without data sharing. Each organization (client) trains locally on its own data and only shares model weight updates with a central server, which aggregates them into a global model.

## 2. Problem Definition
The standard aggregation algorithm **FedAvg** (McMahan et al., 2017) computes the global model as a weighted average of client models, where the weight of client $j$ is purely proportional to its dataset size:

$$w_j = \frac{n_j}{\sum_{k=1}^K n_k}$$

**This creates two concrete problems for federated RAG:**

1. **Quality Blindness**: A client with 10,000 noisy, mislabeled query-document pairs gets 10x the influence of a client with 1,000 high-quality pairs. In RAG, "quality" has a measurable meaning: how well the retriever's selected documents help the generator produce correct answers. This quality signal (the training loss) exists during FL but is completely ignored by FedAvg.

2. **Domain Irrelevance**: In a multi-domain federation (e.g., cardiology, oncology, radiology hospitals), training a cardiology-focused retriever is actively harmed by updates from radiology clients whose data pushes the retriever toward irrelevant documents.

## 3. System Architecture
Our implementation is built on the open-source **FedRAG** library (https://github.com/nerdai/fed-rag) and the **Flower (flwr)** federated learning framework.

**Components:**
- **Retriever Model**: `sentence-transformers/all-MiniLM-L6-v2` (22M parameters, 384-dim embeddings)
- **Generator Model**: `distilgpt2` (82M parameters, used frozen as a teacher signal)
- **Knowledge Store**: `InMemoryKnowledgeStore` (fed-rag built-in, stores pre-embedded document vectors)
- **Dataset**: NFCorpus from the BEIR benchmark (Medical/Nutrition domain — 3,633 documents, 3,237 queries)
- **Non-IID Dataset**: SciFact (Scientific Claims domain — 5,183 documents) used for domain-heterogeneity experiments
- **FL Framework**: Flower `start_simulation()` for multi-client simulation

**Federated Training Round:**
1. Server broadcasts global model weights θ to all K clients
2. Each client loads θ, trains locally for 1 epoch using LSR, reports back (updated_weights, dataset_size, training_loss)
3. Server aggregates using QA-FedAvg strategy
4. Next round begins

## 4. Our Contributions (Mid-Semester)

### Contribution 1: FedRAG Bug Fixes (Enabling Retriever Training)
We identified and fixed 3 critical bugs in the FedRAG library that completely prevented the retriever from learning. These fixes restored gradient flow from the loss function back to the encoder parameters. (Details in file 03.)

### Contribution 2: Quality-Aware Federated Aggregation (QA-FedAvg)
We designed and implemented a custom Flower Strategy (`QualityAwareFedAvg`) that uses the client's training loss as a quality signal. The α hyperparameter controls the balance between data-size and quality weighting. (Formula in file 02, code in file 05.)

### Contribution 3: Baseline Comparison Framework
We built a complete baseline comparison pipeline:
- Centralized (upper bound)
- Federated IID (standard FedAvg, homogeneous data)
- Federated Non-IID (standard FedAvg, heterogeneous domains)
- QA-FedAvg with α sweep (0.0, 0.2, 0.4, 0.6, 1.0)

### Future Work: Domain-Aware Client Selection (DAS-FedAvg)
Selective client participation based on cosine similarity between client domain embeddings and the target domain. (Details in file 06.)
