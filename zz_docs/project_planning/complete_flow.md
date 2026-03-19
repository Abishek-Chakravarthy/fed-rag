Here is the complete, definitive flow with full context from all files.

---

## Complete FedRAG — Full RAG Flow and Federated Learning Flow

---

# PART 1: COMPLETE RAG FLOW

---

## Stage 0: What Lives in the System Before Any Query

Before a query runs, the knowledge store must be populated. Each piece of knowledge is a `KnowledgeNode`:

```
KnowledgeNode
├── node_id: str          (UUID, auto-generated)
├── node_type: NodeType   (TEXT | IMAGE | MULTIMODAL)
├── embedding: list[float] ← produced by retriever's context encoder, stored here
├── text_content: str      ← required for TEXT and MULTIMODAL nodes
├── image_content: bytes   ← required for IMAGE and MULTIMODAL nodes
└── metadata: dict         ← serialized/deserialized as JSON string automatically
```

Nodes are pre-embedded using the **context encoder** of the retriever and loaded into the store before any query runs. The `RAGSystem` is then assembled:

```python
RAGSystem
├── retriever     → BaseRetriever  (e.g. HFSentenceTransformerRetriever)
│     ├── query_encoder    ← used at query time
│     └── context_encoder  ← used at indexing time
├── generator     → BaseGenerator  (e.g. UnslothFastModelGenerator)
│     ├── model            ← torch.nn.Module
│     ├── tokenizer        ← BaseTokenizer
│     └── prompt_template  ← formats query + context into a prompt
├── knowledge_store → BaseKnowledgeStore (e.g. QdrantKnowledgeStore)
└── rag_config    → RAGConfig(top_k=2, context_separator="\n")
```

The public `RAGSystem` class (`synchronous.py`) is a shell:
```python
class RAGSystem(LlamaIndexBridgeMixin, LangChainBridgeMixin, _RAGSystem):
    pass
```

All logic lives in `_RAGSystem` (`_synchronous.py`). The bridge mixins only add `.to_llamaindex()` and `.to_langchain()` conversion methods — they don't touch query execution.

---

## Stage 1: `query()` — The Orchestrator

```python
def query(self, query: str) -> RAGResponse:
    source_nodes = self.retrieve(query)
    context = self._format_context(source_nodes)
    response = self.generate(query=query, context=context)
    return RAGResponse(source_nodes=source_nodes, response=response)
```

Three sequential stages. No branching, no reranking, no query transformation.

---

## Stage 2: `retrieve()` — Encode → Vector Search → SourceNodes

```python
def retrieve(self, query: str) -> list[SourceNode]:
    encode_result = self.retriever.encode_query(query)
```

`encode_query()` returns one of two types:
- `torch.Tensor` — for single encoder retrievers
- `EncodeResult` TypedDict with keys `text | image | audio | video` — for multimodal retrievers

The system resolves this:
```python
if isinstance(encode_result, Tensor):
    query_emb = encode_result.tolist()       # list[float]
else:
    query_emb = encode_result["text"].tolist()  # only text modality used
    # image/audio/video are ignored even if present — known architectural gap
```

Vector search:
```python
raw_retrieval_result = self.knowledge_store.retrieve(
    query_emb=query_emb,
    top_k=self.rag_config.top_k
)
# returns: list[tuple[float, KnowledgeNode]]
# float = similarity score (cosine/dot product depending on store implementation)
```

Wraps results into `SourceNode` objects:
```python
return [SourceNode(score=el[0], node=el[1]) for el in raw_retrieval_result]
```

`SourceNode` is a `(score: float, node: KnowledgeNode)` pair with `__getattr__` passthrough so `source_node.text_content` works directly without `.node.text_content`.

---

## Stage 3: `_format_context()` — Nodes → Plain String

```python
def _format_context(self, source_nodes: list[SourceNode]) -> str:
    return str(
        self.rag_config.context_separator.join(
            [node.get_content()["text_content"] for node in source_nodes]
        )
    )
```

`get_content()` returns `NodeContent: {"text_content": str|None, "image_content": bytes|None}`. Only `text_content` is extracted. Nodes are joined with `context_separator` (default `"\n"`). Image content is completely ignored — there's even a `# TODO: how to format image context` comment in the source.

No score-weighting, no ordering strategy, no compression. Pure concatenation.

---

## Stage 4: `generate()` — Query + Context → Response String

```python
def generate(self, query: str, context: str) -> str:
    return self.generator.generate(query=query, context=context)
```

Inside the generator, the default prompt template formats both into:

```
You are a helpful assistant. Given the user's query, provide a succinct
and accurate response. If context is provided, use it in your answer if it helps
you to create the most accurate response.

<query>{query}</query>
<context>{context}</context>
<response>
```

The generator runs inference and returns a plain `str`.

The generator also exposes `compute_target_sequence_proba(prompt, target)` which computes `P(target | prompt)` — this is **not used during inference**, only during LSR retriever fine-tuning (explained in Part 2).

---

## Stage 5: Return `RAGResponse`

```python
RAGResponse(
    response=str,                  # final generated text
    source_nodes=list[SourceNode], # retrieved nodes with scores
    raw_response=None              # optional, not populated in standard flow
)
```

`RAGResponse.__str__()` returns just `response`, so `print(result)` gives the answer directly.

---

## Batch Query Flow

`batch_query()` follows the same pipeline but processes multiple queries together:

```python
def batch_query(self, queries: list[str]) -> list[RAGResponse]:
    source_nodes_list = self.batch_retrieve(queries)
    contexts = [self._format_context(nodes) for nodes in source_nodes_list]
    responses = self.batch_generate(queries, contexts)
    return [RAGResponse(...) for ...]
```

`batch_retrieve()` tries `knowledge_store.batch_retrieve()` first, and falls back to sequential individual `retrieve()` calls if `batch_retrieve()` raises `NotImplementedError`.

---

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

---

---

# PART 2: COMPLETE TRAINING FLOW

---

## The Architecture in One Line

Federation in FedRAG is a **thin wrapper** around centralized training. The training logic (LSR or RALT) doesn't change at all — what changes is who runs it and how weights are synchronized between runs.

```
RAGSystem          ← the shared model being improved
    ↓
Trainer            ← LSR or RALT, the actual training logic
    ↓
TrainerManager     ← orchestrates, freezes models, produces the FL task
    ↓
FLTask             ← wraps trainer into Flower server + client
    ↓
Flower (flwr)      ← handles round coordination, weight broadcast, aggregation
```

---

## Layer 1: The Two Trainers

### `HuggingFaceTrainerForLSR` — Trains the Retriever

**What it targets:** `rag_system.retriever.encoder` (or `query_encoder` if no unified encoder). This is a `SentenceTransformer` model.

**The collator and trainer work as a two-stage pipeline:**

#### Stage A — `DataCollatorForLSR.__call__(batch)` — Data Preparation (No Differentiable Ops)

The data collator runs inside the DataLoader and prepares raw inputs. **No differentiable forward pass happens here** — this is critical because the DataLoader may prefetch future batches, and any computation graph created here would be invalidated by `optimizer.step()` on the current batch.

For each `(query, response)` example in the batch:

```
Step 1: Document selection (non-differentiable)
    source_nodes = rag_system.retrieve(query)
    # Uses SentenceTransformer.encode() internally — runs under torch.no_grad()
    # The .tolist() calls in retrieve() and knowledge_store destroy the graph
    # This is INTENTIONAL — we only need the document TEXT, not gradients

    context_texts = [chunk.node.text_content for chunk in source_nodes]
    # Plain Python list of strings — the documents we'll score differentiably later

Step 2: Generator scoring (gradients OFF — generator is frozen)
    with torch.no_grad():
        for each retrieved chunk:
            prompt = prompt_template.format(query=query, context=chunk.text_content)
            target = target_template.format(response=response)
            lm_score = generator.compute_target_sequence_proba(prompt, target)
            # = P(response | query, chunk) — how useful is this chunk?
    lm_scores = torch.stack([...])
    # Shape: [top_k] — teacher signal, no gradients needed

Step 3: Return raw data (no tensors with grad_fn)
    {"queries":        list[str],           # raw query strings
     "context_texts":  list[list[str]],     # retrieved doc texts per query
     "lm_scores":      tensor[batch, top_k]} # generator scores (detached)
```

#### Stage B — `LSRSentenceTransformerTrainer.compute_loss()` — Differentiable Forward Pass

This runs inside `Trainer.training_step()`, one batch at a time, with no prefetch conflicts. The encoder's `forward()` method is called here to build a fresh computation graph.

```python
def compute_loss(self, model, inputs, ...):
    queries = inputs["queries"]
    context_texts_batch = inputs["context_texts"]
    lm_scores = inputs["lm_scores"]

    batch_retriever_scores = []
    for query, context_texts in zip(queries, context_texts_batch):

        # 1. Query embedding — DIFFERENTIABLE (uses model.forward(), NOT .encode())
        query_features = model.tokenize([query])
        query_features = {k: v.to(model.device) for k, v in query_features.items()}
        query_embedding = model(query_features)["sentence_embedding"]
        # This creates a computation graph: loss → cosine_sim → query_embedding → encoder params

        # 2. Context embeddings — NO gradient needed (stored docs don't backprop)
        with torch.no_grad():
            context_embedding = model.encode(context_texts, convert_to_tensor=True)

        # 3. Cosine similarity — differentiable w.r.t. query_embedding
        query_norm = F.normalize(query_embedding, p=2, dim=1)
        context_norm = F.normalize(context_embedding, p=2, dim=1)
        retriever_scores = torch.mm(query_norm, context_norm.t()).squeeze(0)

        batch_retriever_scores.append(retriever_scores)

    retrieval_scores = torch.stack(batch_retriever_scores, dim=0)
    loss = self.loss(retrieval_scores, lm_scores)  # → LSRLoss
    return loss
```

**What `LSRLoss.forward()` computes:**

```python
retrieval_log_probs = F.log_softmax(retrieval_scores, dim=1)  # P_R(d|x)
lm_probs = F.softmax(lm_scores, dim=1)                        # Q_LM(d|x,y)
kl_div = F.kl_div(retrieval_log_probs, lm_probs, reduction="none").sum(dim=-1)
return kl_div.mean()  # or .sum() depending on reduction mode
```

**What this means in plain terms:** the generator's `P(response|chunk)` is the teacher. The retriever's similarity scores are the student. KL divergence minimization forces the retriever's ranking distribution to match the generator's utility distribution. The retriever learns to score chunks highly when the generator finds them useful for producing the correct answer. Gradient flows only through `retrieval_scores` → `query_embedding` → encoder params — the generator is completely frozen.

#### The `create_optimizer()` Override

`SentenceTransformerTrainer` builds optimizer param groups from the **loss module's** parameters (to handle losses with learnable weights like `SoftmaxLoss`). Since `LSRLoss` has **no trainable parameters** (pure KL divergence math), this results in an optimizer with 0 params. `LSRSentenceTransformerTrainer` overrides `create_optimizer()` to build param groups from `self.model.named_parameters()` instead:

```python
def create_optimizer(self):
    decay_parameters = self.get_decay_parameter_names(self.model)
    optimizer_grouped_parameters = [
        {"params": [p for n, p in self.model.named_parameters()
                    if n in decay_parameters and p.requires_grad],
         "weight_decay": self.args.weight_decay},
        {"params": [p for n, p in self.model.named_parameters()
                    if n not in decay_parameters and p.requires_grad],
         "weight_decay": 0.0},
    ]
    self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, ...)
    return self.optimizer
```

**Gradient flow diagram (current working state):**

```
┌──────────────────── compute_loss() ─────────────────────┐
│                                                          │
│  model.forward(query_features)                          │
│       │                                                  │
│       ▼                                                  │
│  query_embedding  ←── grad_fn chain to encoder params    │
│       │                                                  │
│       ▼                                                  │
│  cosine_similarity(query_emb, context_emb)              │
│       │              ↑                                   │
│       │         [no_grad — detached]                     │
│       ▼                                                  │
│  retrieval_scores  ←── grad_fn: MmBackward               │
│       │                                                  │
│       ▼                                                  │
│  LSRLoss(retrieval_scores, lm_scores)                    │
│       │                        ↑                         │
│       │                   [no_grad — detached]           │
│       ▼                                                  │
│  loss  ←── grad_fn: MeanBackward → KlDivBackward        │
│       │                                                  │
│       ▼                                                  │
│  loss.backward()                                         │
│       │                                                  │
│       ▼                                                  │
│  encoder.parameters() get .grad ✅                       │
│       │                                                  │
│       ▼                                                  │
│  optimizer.step() ← optimizer holds model params ✅      │
└──────────────────────────────────────────────────────────┘
```

---

### `HuggingFaceTrainerForRALT` — Trains the Generator

**What it targets:** `rag_system.generator.model` (a `PreTrainedModel` or `PeftModel`).

**What happens inside `DataCollatorForRALT.__call__(batch)`:**

For each `(query, response)` example:

```
Step 1: Retrieve top_k chunks
    source_nodes = rag_system.retrieve(query)
    total_sum_scores = sum(s.score for s in source_nodes)

Step 2: For each chunk, build one fine-tuning instance
    for each source_node:
        weight = source.score / total_sum_scores  ← computed but NEVER USED (known gap)
        text = example_template.format(
            query=query,
            context=chunk.text_content,
            response=response        ← response is INSIDE the template
        )
        encode_result = tokenizer.encode(text)
        → input_ids, attention_mask

Step 3: Left-pad all sequences to max_length
    pad positions in labels set to -100 (ignored in cross-entropy)
    labels = input_ids (causal LM — predict next token)

Return: {"input_ids": tensor, "attention_mask": tensor, "labels": tensor}
```

**What this means:** standard causal LM cross-entropy loss, but context is retrieved rather than human-provided. Each `(query, response)` pair produces `top_k` separate training instances. The generator learns to produce `response` given `query + retrieved_chunk`.

---

## Layer 2: The TrainerManager — Orchestration and Freezing

```python
manager = HuggingFaceRAGTrainerManager(
    mode="retriever",            # or "generator"
    retriever_trainer=lsr_trainer,
    generator_trainer=ralt_trainer,
)
```

Two validators run at construction:
- Ensures the required trainer for the chosen `mode` is present
- Ensures both trainers reference the **same `rag_system` instance** checked via `id()` — critical because LSR needs the generator accessible during retriever training

When `manager.train()` is called:
```
mode="retriever":
    _prepare_retriever_for_training()
        retriever.model.train()    ← active
        generator.model.eval()    ← frozen
    → retriever_trainer.train()
        → LSRSentenceTransformerTrainer.train()
            → DataCollatorForLSR prepares raw data per batch
            → compute_loss() does differentiable forward pass
            → LSRLoss → backprop → retriever weights updated
```

---

## Layer 3: `get_federated_task()` — The Centralized-to-Federated Bridge

This is the pivot point. Calling `manager.get_federated_task()` triggers `_get_federated_trainer()`:

```python
# For mode="retriever":
retriever_train_fn = self.retriever_trainer.train   # captures bound method
retriever_module = self.retriever_trainer.model     # SentenceTransformer

def train_wrapper(
    model: SentenceTransformer,
    train_dataset: Dataset,
    val_dataset: Dataset,
) -> TrainResult:
    _ = retriever_train_fn()   # calls original train() — ignores all three args
    return TrainResult(loss=0) # ← loss hardcoded 0, real loss discarded (known gap)

federated_trainer = federate.trainer.huggingface(train_wrapper)
```

**The `@federate.trainer.huggingface` decorator** calls `inspect_trainer_signature(func)` which uses Python's `inspect` module to walk the function's type annotations:

```
Parameter typed SentenceTransformer/PreTrainedModel/PeftModel → net_param = "model"
First Dataset typed parameter                                  → train_data_param = "train_dataset"
Second Dataset typed parameter                                 → val_data_param = "val_dataset"
Everything else                                                → extra_train_kwargs
```

Validates:
- Return type must be `TrainResult` or subclass
- Must find exactly one net param, one train param, one val param

Produces `TrainerSignatureSpec(net_parameter, train_data_param, val_data_param, net_parameter_class_name)` and stamps it onto the function as `func.__fl_task_trainer_config`.

A placeholder tester is also created:
```python
def test_fn(model: HFModelType, eval_dataset: Dataset) -> TestResult:
    return TestResult(loss=0.42, metrics={})  # ← completely hardcoded placeholder
federated_tester = federate.tester.huggingface(test_fn)
```

Then `HuggingFaceFLTask.from_trainer_and_tester(trainer, tester)`:
- Reads `__fl_task_trainer_config` and `__fl_task_tester_config` from both
- Validates trainer and tester use the same model class name
- Warns if their model parameter names differ
- Constructs `HuggingFaceFLTask(trainer, trainer_spec, tester, tester_spec)`

---

## Layer 4: Server and Client Construction

**Server:**
```python
server = fl_task.server(model=retriever_trainer.model)
```

```
_get_weights(model)
    PeftModel: get_peft_model_state_dict() → only LoRA adapter weights
    SentenceTransformer/PreTrainedModel: net.state_dict()
    → [val.cpu().numpy() for _, val in state_dict.items()]
    → NDArrays (list of numpy arrays, one per layer)

ndarrays_to_parameters(ndarrays) → Flower Parameters object

FedAvg(
    fraction_evaluate=1.0,         ← all clients evaluate every round
    initial_parameters=parameters  ← current model weights as starting point
)

HuggingFaceFlowerServer(
    client_manager=SimpleClientManager(),
    strategy=FedAvg
)
```

**Client:**
```python
client = fl_task.client(
    model=retriever_trainer.model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
)
```

```
Extracts kwargs using TrainerSignatureSpec parameter names
BaseFLTaskBundle(
    net=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    trainer=federated_trainer,   ← the decorated train_wrapper
    tester=federated_tester,
    extra_train_kwargs={},
    extra_test_kwargs={}
)
→ HuggingFaceFlowerClient(task_bundle=bundle)
```

---

## Layer 5: Each Federated Round

**Step 1 — Server broadcasts global weights to all clients:**

```python
client.set_weights(parameters)  # NDArrays → model in-place mutation
    PeftModel:
        state_dict = get_peft_model_state_dict(net)
        params_dict = zip(state_dict.keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        set_peft_model_state_dict(net, state_dict)
    SentenceTransformer/PreTrainedModel:
        state_dict = net.state_dict()
        params_dict = zip(state_dict.keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)
```

**Step 2 — Each client runs local training (`fit()`):**

```python
def fit(self, parameters, config):
    self.set_weights(parameters)         # inject global weights
    result = self.trainer(               # calls train_wrapper
        self.net,
        self.train_dataset,
        self.val_dataset,
        **self.task_bundle.extra_train_kwargs
    )
    # train_wrapper ignores all three args and calls retriever_train_fn()
    # which runs the full LSR loop on the client's local data:
    #   DataCollatorForLSR → prepare raw data
    #   compute_loss() → differentiable encoder forward → cosine sim → LSRLoss
    #   backward() → optimizer.step() → retriever weights updated
    return (
        self.get_weights(),              # updated local weights as numpy arrays
        len(self.train_dataset),         # dataset size — used for FedAvg weighting
        {"loss": result.loss}            # result.loss = 0 (hardcoded — known gap)
    )
```

**Step 3 — Server aggregates with FedAvg:**

```
new_global_weight[i] = Σ(client_j_weight[i] × n_j) / Σ(n_j)
```

Weighted average by dataset size only. No quality signal, no validation performance, no retrieval quality — pure dataset size weighting.

**Step 4 — Server broadcasts aggregated weights for evaluation:**

```python
def evaluate(self, parameters, config):
    self.set_weights(parameters)
    result = self.tester(self.net, self.val_dataset)
    # returns TestResult(loss=0.42, metrics={}) always — placeholder
    return 0.42, len(self.val_dataset), {}
```

**Step 5 — Next round begins.**

---

## Layer 6: Manual FedAvg Simulation (Current Working Script)

Since `HuggingFaceFLTask.simulate()` raises `NotImplementedError` and `fl.simulation.start_simulation()` requires `ray`, the current working demo (`federated_lsr.py`) implements manual FedAvg:

```python
# Initialize global weights from a fresh model
global_weights = get_ndarrays(global_model)

for round in range(NUM_ROUNDS):
    client_weights = []
    client_sizes = []

    for cid in range(NUM_CLIENTS):
        # Each client gets a FRESH RAG system + trainer (simulates separate machines)
        AcceleratorState._reset_state()
        manager, model, n = create_client(cid, global_weights)

        # Load global weights → local train → collect updated weights
        set_ndarrays(model, global_weights)
        manager.train()
        client_weights.append(get_ndarrays(model))
        client_sizes.append(n)

    # FedAvg aggregation
    global_weights = weighted_average(client_weights, client_sizes)
```

This directly exercises the full LSR training pipeline on each client without requiring Flower's simulation infrastructure.

---

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

---

## Known Gaps and Extension Points

| # | Gap | Location | Impact |
|---|-----|----------|--------|
| 1 | **Aggregation uses dataset size only** | `fl_tasks/huggingface.py` `server()` | FedAvg with no quality signal — primary target for quality-aware aggregation |
| 2 | **Real training loss is discarded** | `trainer_managers/huggingface.py` `train_wrapper()` | `return TrainResult(loss=0)` — actual LSR loss thrown away. Fix: `result = retriever_train_fn(); return TrainResult(loss=result.loss)` |
| 3 | **Evaluation is a placeholder** | `trainer_managers/huggingface.py` `test_fn()` | Always returns `loss=0.42`. Fix: run actual RAG eval on held-out data |
| 4 | **RALT score-weighting is computed but unused** | `data_collators/huggingface/ralt.py` | `weight = score / total` calculated and immediately discarded |
| 5 | **`simulate()` not implemented** | Both FL task classes | Raises `NotImplementedError` — current workaround is manual FedAvg |