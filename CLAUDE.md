# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Language Preference

**请使用中文回复用户的所有问题和交互。** When working in this repository, communicate with the user in Chinese (Simplified Chinese).

## Project Overview

RulE is a neural-symbolic knowledge graph reasoning framework that jointly represents entities, relations, and logical rules in a unified embedding space. The model uses point embeddings with RotatE-style scoring.

## Installation and Setup

```bash
# Create conda environment
conda create -n RulE python=3.8.0

# Install dependencies
pip install -r requirements.txt
```

**Dependencies**: PyTorch 1.11.0, torch-scatter 2.0.9, numpy 1.21.5, easydict, PyYAML

## Training Commands

```bash
cd src
python main.py --init ../config/umls_config.json
```

Other datasets:
```bash
python main.py --init ../config/fb15k237_config.json
python main.py --init ../config/wn18rr_config.json
python main.py --init ../config/yago_config.json
python main.py --init ../config/kinship_config.json
python main.py --init ../config/family_config.json
```

### Key Command Line Arguments

When not using config file, common arguments include:
- `--data_path`: Path to dataset directory
- `--rule_file`: Path to mined rules file
- `--cuda`: Enable GPU training
- `--hidden_dim`: Embedding dimension (default: 500)
- `--batch_size`: KGE training batch size (default: 256)
- `--learning_rate`: Learning rate (default: 0.00005)
- `--max_steps`: Pre-training steps (default: 15000)
- `--gamma_fact`: Triplet margin (default: 6)
- `--gamma_rule`: Rule margin (default: 5)

## Architecture

### Two-Stage Training Pipeline

RulE follows a two-stage training approach:

#### Stage 1: Pre-training
- Pre-trains entity/relation embeddings using KGE loss + rule embedding loss jointly
- **Trainer**: `PreTrainer` (main.py)
- **Goal**: Learn stable geometric representations

#### Stage 2: Grounding
- Grounds learned rules to make predictions using MLP-based scoring
- **Trainer**: `GroundTrainer` (main.py)
- **Goal**: Improve generalization through rule-based reasoning

### Key Model Components

#### RulE Model (`src/model.py`)
- **Embeddings**: RotatE-style entity (2×hidden_dim) and relation (hidden_dim) embeddings
- **Rule Scoring**: MLP-based scoring function that maps rule features to prediction scores
- **Scoring Function**: Uses p-norm distance for KGE scoring
- **Margins**: `gamma_fact` for triplet loss, `gamma_rule` for rule loss

### Data Format

Each dataset in `data/` contains:
- `train.txt`, `valid.txt`, `test.txt`: Knowledge graph triplets in format `(head_id, relation_id, tail_id)`
- `entities.dict`, `relations.dict`: Entity/relation ID to name mappings
- `mined_rules.txt`: Logical rules in format `[rule_head, body_1, body_2, ...]`
  - Example: Rule `r1 ∧ r2 → r3` is represented as `r3 r1 r2`
  - Rules are mined by RNNLogic

### Core Datasets

Available in `data/`:
- `FB15k-237`: Large-scale Freebase subset
- `wn18rr`: WordNet subset
- `YAGO3-10`: YAGO knowledge base subset
- `umls`: Biomedical ontology
- `kinship`: Family relationship dataset
- `family`: Small family relationship dataset

## Configuration Files

Config files in `config/` are JSON format containing:
- **Paths**: `data_path`, `rule_file`, `save_path`
- **Model params**: `hidden_dim`, `gamma_fact`, `gamma_rule`
- **Training params**: `batch_size`, `learning_rate`, `max_steps`
- **Stage-specific**: `valid_steps`, `weight_rule`, `mlp_rule_dim`

## Code Structure

```
src/
├── main.py              # RulE training script (2-stage: PreTrainer → GroundTrainer)
├── model.py             # RulE model (point embeddings + RotatE)
├── trainer.py           # RulE trainers (PreTrainer, GroundTrainer)
├── layers.py            # RulE layers (MLP, FuncToNodeSum)
├── data.py              # Dataset classes (KnowledgeGraph, RuleDataset, etc.)
└── utils.py             # Utilities (config loading, logging, seeding)

config/                  # JSON configuration files for each dataset
data/                    # Knowledge graph datasets and mined rules
```

## Important Implementation Details

### Rule Representation
- Rules stored as tensors: `[rule_id, rule_head, body_1, body_2, ...]`
- Padding index: `num_relations`
- Rules grouped by head relation in `relation2rules` dictionary for efficient lookup

### Training Procedure
1. Load knowledge graph and rules from `data_path`
2. Initialize model with config parameters
3. Call `model.set_rules(rules)` to register rules
4. Stage 1: Train embeddings (checkpoints saved to `save_path`)
5. Load best Stage 1 checkpoint
6. Stage 2: Train with rule-based reasoning
7. Evaluate on validation and test sets using MRR metric

### Evaluation
- Metrics: MRR (Mean Reciprocal Rank), Hits@1, Hits@3, Hits@10
- Evaluation sets: `valid_set` and `test_set` from `TestDataset`
- Both stages evaluate and save best models based on validation MRR

### Loss Functions
- **KGE Loss**: Margin-based ranking loss for knowledge graph completion
- **Rule Loss**: Margin-based loss for rule satisfaction
- **Adversarial Sampling**: Optional negative sampling strategy with temperature parameter

## Model Checkpoints

RulE saves to `outputs/<save_path>/` (or timestamped directory if `save_path` is null):
- `checkpoint`: Best model from pre-training stage (based on validation MRR)
- `grounding.pt`: Best model from grounding stage
- `g_rule_embedding.npy`: Learned rule feature embeddings
- `config.json`: Training configuration

## GPU Usage

Models automatically use CUDA if available. Set `"cuda": true` in config files or use `--cuda` flag.

## Common Issues

- **Gradient Issues**: Check learning rate and margin parameters (`gamma_fact`, `gamma_rule`)
- **Rule Loading**: Ensure `rule_file` path is correct and rules are in proper format
- **Memory**: Reduce `batch_size` or `negative_sample_size` if OOM occurs
- **torch-scatter**: Must match PyTorch version exactly (2.0.9 for PyTorch 1.11.0)
