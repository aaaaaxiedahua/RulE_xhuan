# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RulE is a neural-symbolic knowledge graph reasoning framework that combines Knowledge Graph Embeddings (KGE) with logical rule embeddings. The system uses a two-phase training approach: pre-training (learning entity/relation embeddings + rule embeddings) followed by grounding (learning to combine rules for inference).

## Core Architecture

### Three-Component Pipeline

1. **Pre-training Phase** (`PreTrainer` in `trainer.py`)
   - Jointly trains Knowledge Graph Embeddings (RotatE) and Rule Embeddings
   - Uses margin-based loss with adversarial negative sampling
   - Trains on triplets (h, r, t) and logical rules simultaneously
   - Loss = loss_fact + loss_rule (weighted combination)

2. **Grounding Phase** (`GroundTrainer` in `trainer.py`)
   - Freezes pre-trained embeddings (entity, relation, rule)
   - Learns rule aggregation weights via MLP layers
   - Trains on how to ground and combine rules for specific queries
   - Uses label smoothing and knowledge distillation from pre-training

3. **Inference** (`RulE.forward()` in `model.py`)
   - For query (h, r, ?), finds applicable rules for relation r
   - Grounds each rule by following its body relations in the KG
   - Aggregates grounded rules using learned MLP weights
   - Combines with KGE scores (optional, controlled by alpha parameter)

### Key Components

- **`model.py`**: `RulE` class containing:
  - RotatE-based triplet scoring (`compute_KGE`, `compute_g_KGE`)
  - Rule embedding scoring (`compute_ruleE`, `add_ruleE`)
  - Rule grounding and aggregation (`forward`, `grounding`)
  - MLP-based score prediction (`score_model`)

- **`data.py`**: Data structures for:
  - `KnowledgeGraph`: Graph structure with adjacency lists for rule grounding
  - `RuleDataset`: Loads rules from mined_rules.txt with negative sampling
  - `KGETrainDataset`: Triplet dataset with filtered negative sampling
  - `TrainDataset`, `ValidDataset`, `TestDataset`: Grounding phase datasets

- **`layers.py`**: Neural network modules:
  - `MLP`: Standard multi-layer perceptron
  - `FuncToNodeSum`: Aggregates rule groundings with learned weights

- **`trainer.py`**: Training loops for both phases:
  - `PreTrainer`: Pre-training with alternating triplet/rule batches
  - `GroundTrainer`: Grounding with rule-specific batching

## Running the Code

### Environment Setup
```bash
conda create -n RulE python=3.8.0
pip install -r requirements.txt
```

### Training
```bash
cd src
python main.py --init ../config/umls_config.json
```

The main entry point is `main.py`, which:
1. Loads configuration from config file
2. Creates data loaders (graph, rules, train/valid/test splits)
3. Runs pre-training phase
4. Loads best pre-training checkpoint
5. Runs grounding phase
6. Evaluates final model

### Configuration Files

Located in `config/` directory, one per dataset:
- `data_path`: Path to dataset folder
- `rule_file`: Path to mined rules (format: rule_head followed by rule_body relations)
- Pre-training hyperparameters: `hidden_dim`, `gamma_fact`, `gamma_rule`, `max_steps`, `learning_rate`
- Grounding hyperparameters: `mlp_rule_dim`, `alpha`, `g_lr`, `num_iters`, `smoothing`

## Data Format

Each dataset directory must contain:
- `entities.dict`: Entity ID mappings (format: `id\tentity_name`)
- `relations.dict`: Relation ID mappings (format: `id\relation_name`)
- `train.txt`, `valid.txt`, `test.txt`: Triplets (format: `head\trelation\ttail`)
- `mined_rules.txt`: Logical rules (format: `rule_head body_relation_1 body_relation_2 ...`)

Rules use relation IDs where inverse relations are encoded as `relation_id + num_relations`.

## Key Design Patterns

### Rule Representation
- Rules stored as: `[rule_id, rule_head, body_relation_1, body_relation_2, ...]`
- Padded to max rule length with padding index = `num_relations * 2`
- Each rule has an embedding learned during pre-training

### Relation Indexing
- Forward relations: `0` to `num_relations - 1`
- Inverse relations: `num_relations` to `2 * num_relations - 1`
- Padding index: `2 * num_relations`

### Rule Grounding
- Uses message passing over KG adjacency lists
- For rule body `r1 ∧ r2 → r_head`, starting from head entity h:
  - Propagate through r1 to get intermediate entities
  - Propagate through r2 to get tail predictions
- Removes query edge to prevent information leakage

### Evaluation Metrics
- Filtered ranking: Removes all known true triplets except target
- Metrics: Hits@1, Hits@3, Hits@10, MR (Mean Rank), MRR (Mean Reciprocal Rank)
- Uses expectation over tied ranks

## Model Checkpoints

Checkpoints saved in `outputs/` directory (or path specified in config):
- `checkpoint`: Pre-training model (best validation MRR)
- `grounding.pt`: Grounding phase model (best validation MRR)
- `g_rule_embedding.npy`: Grounded rule embeddings for analysis

## Important Implementation Details

- **Two-stage freezing**: Pre-training learns all embeddings; grounding freezes them and only trains MLP layers
- **Bidirectional facts**: All facts stored as both (h, r, t) and (t, r_inv, h) for completeness
- **Learning rate decay**: Cuts learning rate by 10× at warm-up steps (default: half of max_steps)
- **Adversarial sampling**: Temperature-weighted negative sampling (can be disabled with `disable_adv`)
- **Edge removal during grounding**: Prevents using the query triplet in its own grounding

## Testing

To evaluate a trained model without retraining, comment out training calls in `main.py` and load the checkpoint directly before calling `evaluate()`.
