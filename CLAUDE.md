# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

RulE is a neural-symbolic knowledge graph reasoning framework that jointly represents entities, relations, and logical rules in a unified embedding space. The model consists of three main components:

1. **Pre-training**: Learns embeddings for entities, relations, and rules using RotatE for triplets and a specialized loss for rules
2. **Grounding**: Propagates logic rules through the knowledge graph to identify applicable instances
3. **Inference**: Combines rule-based reasoning with knowledge graph embeddings for predictions

## Quick Reference

**Most common commands:**
```bash
# Setup (first time only)
conda create -n RulE python=3.8.0
pip install -r requirements.txt

# Train from scratch (run from src/ directory)
cd src
python main.py --init ../config/umls_config.json

# Train on different datasets
python main.py --init ../config/fb15k237_config.json
python main.py --init ../config/wn18rr_config.json
```

**Key file locations:**
- Source code: `src/` (model.py, trainer.py, data.py, main.py)
- Configs: `config/<dataset>_config.json`
- Data: `data/<dataset>/` (entities.dict, relations.dict, train/valid/test.txt, mined_rules.txt)
- Outputs: `../outputs/<timestamp>/` or custom path specified in config

**Current branch: `rule_rl`**
- This branch is for RulE-RL development (reinforcement learning enhancement)
- Pre-training is currently **enabled** by default (line 155 in main.py)
- See "RulE-RL Enhancement" section below for details

## Repository Branches

- **`main`**: Original RulE implementation
- **`rule_rl`**: RulE-RL enhancement using reinforcement learning (current branch, in development)
- **`rule_gnn`**: GNN-based enhancement (experimental)

## Running the Code

### Setup Environment

```bash
conda create -n RulE python=3.8.0
pip install -r requirements.txt
```

Note: The README says `requirement.txt` but the actual file is `requirements.txt`.

### Training

Train on a specific dataset (all commands should be run from the `src/` directory):

```bash
cd src
python main.py --init ../config/umls_config.json
```

Available datasets (based on existing config files):
- `umls` → `../config/umls_config.json`
- `fb15k237` → `../config/fb15k237_config.json`
- `wn18rr` → `../config/wn18rr_config.json`
- `kinship` → `../config/kinship_config.json`
- `family` → `../config/family_config.json`
- `yago` → `../config/yago_config.json`

### Current Training Workflow

**Current State (as of November 2024)**: The `main.py` on the `rule_rl` branch has pre-training **enabled** (line 155 is uncommented), meaning it will train from scratch by default.

**The training pipeline in main.py:155-194:**

1. **Pre-training phase** (line 155, **ACTIVE**):
   - `pre_trainer.train(args)` - Trains entity/relation/rule embeddings jointly
   - Saves best checkpoint to `args.save_path/checkpoint`

2. **Load pre-training checkpoint** (lines 164-165):
   - Loads the checkpoint just created (or existing if resuming)

3. **Evaluate pre-training** (lines 170-171):
   - Tests the pre-trained model on validation and test sets

4. **Grounding phase** (lines 178-194):
   - Lines 175-176 for loading existing grounding checkpoint are **commented out**
   - `ground_trainer.train(args)` - Trains grounding phase from scratch
   - Trains only MLP parameters while freezing embeddings
   - Saves checkpoint to `args.save_path/grounding.pt`

**To train from scratch (current default):**
- Just run as-is: `python main.py --init ../config/umls_config.json`
- Pre-training and grounding will both run

**To resume from existing pre-training:**
1. Comment out line 155 (`# pre_trainer.train(args)`)
2. Ensure lines 164-165 are uncommented to load existing checkpoint
3. For grounding: uncomment lines 175-176 if resuming grounding phase

**To evaluate only:**
1. Comment out line 155 (skip pre-training)
2. Comment out line 194 (skip grounding training)
3. Uncomment lines 164-165 and 175-176 to load both checkpoints

## Architecture

### Core Model (src/model.py)

**RulE class** - Main model containing:
- `entity_embedding`: Entity embeddings (dimension: num_entities × hidden_dim*2 for complex numbers)
- `relation_embedding`: Relation embeddings (dimension: num_relations × hidden_dim)
- `rule_emb`: Rule embeddings learned during pre-training
- `mlp_feature`: MLP features for rules used during grounding

**Key methods:**
- `compute_KGE()`: Computes RotatE scores for knowledge graph triplets
- `compute_ruleE()`: Computes rule embedding scores by aggregating relation embeddings in rule body
- `forward()`: Main grounding inference - grounds rules on KG and scores candidates using learned MLP

**RotatE Implementation**: Uses complex-valued embeddings (real/imaginary parts) with rotation in complex space

### Training Process (src/trainer.py)

**PreTrainer** (src/trainer.py:13-327) - Pre-training phase:
- `train()` method: Jointly trains entity/relation embeddings (RotatE) and rule embeddings
- Uses adversarial negative sampling for both triplets and rules
- Loss function (src/trainer.py:188-191): `loss = loss_rule + loss_fact`
  - `loss_fact`: RotatE triplet loss (positive + negative samples)
  - `loss_rule`: Rule embedding loss (positive + negative samples)
  - Weighted by `args.weight_rule` parameter
- Learning rate schedule: Reduces by 10x at `warm_up_steps` (default: halfway through training)
- Saves best checkpoint based on validation MRR to `args.save_path/checkpoint`
- `evaluate()` method: Evaluates using only KGE scores (no rule grounding)

**GroundTrainer** (src/trainer.py:369-760) - Grounding phase:
- Freezes pre-trained embeddings: `entity_embedding`, `relation_embedding`, `rule_emb` (src/trainer.py:391-393)
- Only trains MLP parameters: `mlp_feature`, `score_model`, `rule_to_entity`, `bias`
- `train_step()` method:
  - Uses cross-entropy loss with label smoothing (src/trainer.py:496)
  - Removes training edges to prevent trivial solutions (src/trainer.py:498)
  - Only updates on samples where rules have groundings (`if mask.sum().item() != 0`)
- Saves checkpoint to `args.save_path/grounding.pt`
- Two evaluation modes:
  - `evaluate()`: Rule-based inference only (no KGE)
  - `evaluate_t()`: Combines rule scores with KGE scores using alpha parameter

### Data Processing (src/data.py)

**KnowledgeGraph** (src/data.py:222-447) - Central data structure:
- Loads entities.dict, relations.dict, train/valid/test.txt files
- Maintains both forward and inverse relations (actual relation_size is 2× num_relations)
  - Forward relation `r` has ID `r`
  - Inverse relation has ID `r + relation_size`
- Key attributes:
  - `relation2adjacency`: Adjacency lists for graph propagation (list of [index_tensor, value_tensor] pairs)
  - `hr2o`: Maps (head, relation) to tails in training set only
  - `hr2oo`: Maps (head, relation) to tails in training + validation
  - `hr2ooo`: Maps (head, relation) to tails in training + validation + test (used for filtering during evaluation)
- `grounding()` (src/data.py:410-421): Performs multi-hop graph traversal to ground rules
  - Takes head entities `h`, query relation `r`, and rule body
  - Returns count tensor showing how many groundings reach each candidate entity
- `propagate()` (src/data.py:423-447): Single-hop message passing using torch_scatter
  - Supports edge removal for training (prevents trivial solutions)

**Dataset classes:**
- `RuleDataset` (src/data.py:11-68): Loads mined_rules.txt
  - Format: `rule_head rule_body_1 rule_body_2 ...` (space-separated relation IDs)
  - Example: `3 1 2` means rule r1 ∧ r2 → r3
  - Performs negative sampling by randomly replacing rule body relations
- `KGETrainDataset` (src/data.py:71-177): Training data for pre-training phase
  - Supports 'head-batch' and 'tail-batch' modes for negative sampling
  - Uses filtered negative sampling (excludes true triplets)
  - Computes subsampling weights for common entities
- `TrainDataset` (src/data.py:449-493): Training batches for grounding phase
  - Groups triplets by relation (required for rule grounding)
  - Provides `edges_to_remove` tensor to mask training edges during grounding
- `ValidDataset` / `TestDataset` (src/data.py:495-567): Evaluation data
  - Returns mask tensor indicating which entities to filter (all true triplets)

### Rule Representation

Rules are stored as lists: `[rule_id, rule_head, relation_1, relation_2, ..., relation_n]`

Example: Rule "r1 ∧ r2 → r3" is represented as `[id, r3, r1, r2]`

Relations are doubled to handle inverse relations: relation `r` has ID `r`, inverse has ID `r + relation_size`

### Grounding Mechanism

The grounding process in `model.forward()` (src/model.py:337-409):
1. For query (h, r), retrieve all rules with head = r from `self.relation2rules[query_r]`
2. For each rule, perform multi-hop propagation from h following rule body relations
   - Calls `graph.grounding(all_h, r_head, r_body, edges_to_remove)` (src/model.py:354)
   - Returns count tensor: number of grounding paths reaching each candidate entity
3. Collect all rules that have at least one grounding (skip if `mask.sum().item() == 0`)
4. For each candidate entity with groundings:
   - Aggregate rule features weighted by grounding counts using `FuncToNodeSum` (src/model.py:380)
   - `FuncToNodeSum` performs:
     - Matrix multiply grounding counts with rule MLP features (src/layers.py:68)
     - Apply layer norm and ReLU (src/layers.py:69-70)
     - Average over rules (src/layers.py:71)
5. Score candidates using trained MLP `score_model` (src/model.py:390)
6. Return scores and mask (all entities are valid candidates for grounding phase)

Key insight: The grounding count acts as an attention mechanism - rules that ground more paths to an entity contribute more to its score.

## Configuration Files

Located in `config/` directory. Each dataset has its own JSON config. Key parameters:

**Pre-training hyperparameters:**
- `hidden_dim`: Embedding dimension (varies by dataset: 500-2000)
  - Larger datasets (FB15k-237, YAGO, WN18RR) use 500-1000
  - Smaller datasets (UMLS, Kinship, Family) use 1000-2000
- `gamma_fact`: Margin for triplet loss (typically 6)
- `gamma_rule`: Margin for rule loss (typically 5-8)
- `learning_rate`: Learning rate (0.00005-0.0001)
- `max_steps`: Pre-training steps (15000-30000)
- `weight_rule`: Weight for rule loss vs fact loss (typically 1.0)
- `adversarial_temperature`: Temperature for adversarial sampling (0.25-0.5)
  - Set to 0 or use `disable_adv: false` to disable adversarial sampling after warm-up
- `batch_size`: Batch size for triplets (typically 256)
- `negative_sample_size`: Number of negative samples per positive triplet (256-512)
- `rule_batch_size`: Batch size for rules (128-256)
- `rule_negative_size`: Number of negative samples per positive rule (64-128)

**Grounding hyperparameters:**
- `mlp_rule_dim`: MLP feature dimension for rules (typically 100)
- `alpha`: Weight for combining KGE and rule scores (2.0-5.0)
  - Only used in `evaluate_t()` method, not in `evaluate()`
- `smoothing`: Label smoothing factor (0.2-0.5)
- `g_lr`: Learning rate for grounding (0.0001)
- `g_batch_size`: Batch size for grounding (typically 16)
- `num_iters`: Number of grounding training iterations/epochs (typically 20)
- `batch_per_epoch`: Max batches per grounding epoch (default 1000000 = process all data)
- `print_every`: Log training loss every N batches (typically 10-1000)

**MLP architecture note**:
- For large datasets (FB15k-237, WN18RR, YAGO): uses `MLP(100, [128, 1])` - two layers
- For small datasets (UMLS, Kinship, Family): uses `MLP(100, [1])` - single layer
- Controlled in `src/model.py:32-35` based on dataset name

## Data Format

All data files are located in `data/<dataset>/`:

- `entities.dict`: Format: `<entity_id>\t<entity_name>`
- `relations.dict`: Format: `<relation_id>\t<relation_name>`
- `train.txt`, `valid.txt`, `test.txt`: Format: `<head>\t<relation>\t<tail>` (entity/relation names, not IDs)
- `mined_rules.txt`: Format: `<rule_head> <rule_body_1> <rule_body_2> ...` (relation IDs as integers)

## Model Outputs

Training outputs are saved to `../outputs/<timestamp>/` or custom path via `save_path` config.

For example, pre-trained UMLS models may be saved in `src/umls/`:

**Checkpoint files:**
- `checkpoint`: Model state dict (all parameters from pre-training)
- `grounding.pt`: Grounding phase checkpoint

**Configuration and logs:**
- `config.json`: Training configuration
- `run.log`: Training logs

**Embeddings (saved as numpy arrays):**
- `entity_embedding.npy`: Entity embeddings (dimension: num_entities × hidden_dim*2)
- `relation_embedding.npy`: Relation embeddings (dimension: num_relations × hidden_dim)
- `rule_embedding.npy`: Rule embeddings from pre-training
- `g_rule_embedding.npy`: MLP rule features from grounding phase

## Implementation Notes

**Training workflow in main.py:**
1. Load knowledge graph and datasets (src/main.py:114-119)
2. Initialize RulE model with RotatE embeddings (src/main.py:129-130)
3. **Pre-training phase** (line 155, **currently ACTIVE**):
   - Creates PreTrainer and runs `pre_trainer.train(args)`
   - Trains entity/relation/rule embeddings jointly
   - Saves best checkpoint to `args.save_path/checkpoint`
4. Load pre-training checkpoint (src/main.py:164-165)
5. Evaluate pre-training results (src/main.py:170-171)
6. **Grounding phase** (src/main.py:178-194):
   - Lines 175-176 (loading existing grounding checkpoint) are **commented out**
   - Creates GroundTrainer and runs `ground_trainer.train(args)`
   - Trains only MLP parameters while freezing embeddings
   - Saves checkpoint to `args.save_path/grounding.pt`
7. Final evaluation on test set (within ground_trainer.train)

**Evaluation metrics:**
- MRR (Mean Reciprocal Rank): Primary metric for model selection
- Hits@1, Hits@3, Hits@10: Percentage of correct answers in top-k
- MR (Mean Rank): Average rank of correct answer

Uses filtered setting: excludes all true triplets (train/valid/test) from ranking when computing metrics via `hr2ooo` mapping.

**Two evaluation modes in grounding:**
- `evaluate()`: Rule-based inference only (no KGE contribution)
- `evaluate_t()`: Combines rule scores with KGE scores: `logits = rule_score + alpha * kge_score`

**Key implementation details:**
- Complex-valued embeddings in RotatE: Entity embeddings are `hidden_dim * 2` (real + imaginary parts)
- Relation embeddings use phase representation converted to complex via `cos(phase)` and `sin(phase)`
- Rule embeddings learned via distance metric: `gamma_rule - ||rule_body_sum + rule_emb - rule_head||`
- Inverse relations created automatically: every relation `r` has inverse `r + relation_size`
- Edge removal during grounding training prevents model from memorizing training edges

## RulE-RL Enhancement (In Development)

The repository includes design documents for **RulE-RL**, a reinforcement learning-based enhancement currently being developed on the `rule_rl` branch. This approach aims to improve efficiency and performance by:

### Key Improvements
- **Dynamic Rule Selection**: High-level RL agent selects top-K relevant rules instead of using all rules
- **Guided Path Exploration**: Low-level RL agent performs targeted path finding instead of exhaustive BFS
- **Efficiency Gains**: Expected 2x speedup with 35% rule usage and 12% path enumeration

### Architecture Overview
RulE-RL uses hierarchical reinforcement learning with two agents:

1. **RuleSelectorAgent** (High-level)
   - Selects top-K rules using Contextual Bandit + UCB strategy
   - Input: Query entity and relation embeddings, rule embeddings
   - Output: Selected rules (e.g., top-5 most relevant)

2. **PathFinderAgent** (Low-level)
   - Performs guided path exploration using REINFORCE with baseline
   - Components: Policy network, Value network (for variance reduction)
   - Action space: Valid relations constrained by selected rules

3. **Supporting Components**
   - `StateEncoder`: Encodes current position, query relation, rule context, and path history
   - `KGReasoningEnv`: Environment for KG reasoning with state transitions
   - `RewardCalculator`: Multi-component reward (final, rule consistency, proximity, diversity, penalties)

### Implementation Notes
- **Frozen Components**: Original RulE embeddings (entity, relation, rule) are frozen
- **New Trainable**: Only RL components (policy/value networks, rule selector) are trained
- **Reward Shaping**: Combines final reward with intermediate signals (rule consistency, getting closer, diversity)
- **Exploration**: UCB for rule selection, ε-greedy for path finding

### File Structure (Planned)
```
src/
├── rl/                     # RL module directory (to be created)
│   ├── agents.py           # RuleSelectorAgent, PathFinderAgent
│   ├── env.py              # KGReasoningEnv
│   ├── reward.py           # RewardCalculator
│   ├── encoder.py          # StateEncoder
│   └── trainer.py          # RuleRLTrainer
└── main_rl.py              # RulE-RL main entry point (to be created)
```

### Documentation
Detailed RulE-RL design documentation is available in:
- `RulE-RL方案分析.md`: Architecture and component analysis
- `RulE-RL方案详细流程分析.md`: Detailed training and inference flow (1200+ lines)

**Note**: RulE-RL is currently in design/planning phase. Implementation should follow the phased approach outlined in the design documents.

## Common Issues and Troubleshooting

**Issue: Module not found or import errors**
- Solution: Ensure you're in the `src/` directory when running `main.py`
- All imports in the code assume execution from `src/` directory

**Issue: FileNotFoundError for data files**
- Solution: Check that config paths are relative to `src/` directory
- Example: `"data_path": "../data/umls"` assumes you're in `src/`

**Issue: CUDA out of memory**
- Solution: Reduce `batch_size`, `g_batch_size`, or `hidden_dim` in config file
- Alternatively, set `"cuda": false` to use CPU (slower)

**Issue: Training not making progress**
- Check that `max_steps` is sufficient (typically 15000-30000)
- Verify learning rate isn't too high or too low
- Check that `warm_up_steps` is set appropriately (usually half of `max_steps`)

**Issue: Model checkpoint not found**
- Verify the `save_path` matches where you expect checkpoints
- Pre-training saves to `<save_path>/checkpoint`
- Grounding saves to `<save_path>/grounding.pt`

**Note about dependencies:**
- The README mentions `requirement.txt` but the actual file is `requirements.txt`
- Make sure to use `torch==1.11.0` and `torch-scatter==2.0.9` for compatibility

## Development Notes

**Working with pre-trained models:**
- To skip pre-training: comment out line 155 in `src/main.py`
- To resume grounding: uncomment lines 175-176 in `src/main.py`
- Always ensure checkpoint paths are correct before loading

**Modifying the model:**
- Entity/relation/rule embeddings are in `src/model.py:20-22`
- Training loop modifications go in `src/trainer.py`
- Data loading and KG structure in `src/data.py`

**Adding new datasets:**
1. Create `data/<dataset_name>/` directory
2. Add required files: entities.dict, relations.dict, train/valid/test.txt, mined_rules.txt
3. Create `config/<dataset_name>_config.json` based on existing configs
4. Adjust `hidden_dim` and other hyperparameters based on dataset size

## Additional Documentation

The repository contains additional Chinese-language documentation files that provide detailed analysis:

**In `md/` directory:**
- `RulE_Complete_Example.md`: Complete example walkthrough
- `RulE_详细模型分析.md`: Detailed model analysis (Chinese)
- `UMLS训练日志详细分析.md`: UMLS training log analysis (Chinese)
- `模型超参数说明.md`: Hyperparameter explanations (Chinese)
- `术语解释与概念说明.md`: Terminology and concepts (Chinese)
- `测试集数据泄露分析.md`: Test set data leakage analysis (Chinese)
- `深度学习核心概念解释.md`: Deep learning core concepts (Chinese)
- `RulE-RL强化学习创新方案.md`: RulE-RL reinforcement learning proposal (Chinese)

**Root directory:**
- `RulE-RL方案分析.md`: Architecture and component analysis for RulE-RL
- `RulE-RL方案详细流程分析.md`: Detailed training and inference flow (1200+ lines)
- `RulE-RL完整训练步骤文档.md`: Complete step-by-step RulE-RL training guide
