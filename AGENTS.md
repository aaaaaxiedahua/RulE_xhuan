# Repository Guidelines

## Project Structure & Module Organization
`src/` holds executable code: `main.py` orchestrates runs, `model.py` defines RulE, `trainer.py` handles pre-training and grounding loops, and `data.py`/`utils.py` wrap datasets plus helpers. Dataset configs live under `config/*.json`; keep dataset assets (triples, mined rules) under `data/<dataset>/` and stash heavy experiment logs in `experimentalData/`. Generated checkpoints and logs are written to `outputs/` (created automatically), while documentation artifacts remain in `figures/`, `md/`, and `paper/`.

## Build, Test, and Development Commands
- `python -m venv venv && source venv/bin/activate`: spin up a Python 3.8 environment before touching dependencies.
- `pip install -r requirements.txt`: install PyTorch, numpy, and logging utilities exactly as pinned.
- `python src/main.py --init config/wn18rr_config.json --cuda --save_path outputs/wn18rr_run1`: train/evaluate on WN18RR with checkpoints and logs stored in `outputs/wn18rr_run1`.
- `python src/main.py --init config/fb15k237_config.json --max_steps 500 --valid_steps 100 --log_steps 20`: run a fast regression pass to sanity-check recent edits.

## Coding Style & Naming Conventions
Follow PEP 8 with four-space indents, `snake_case` functions, and `CamelCase` classes (e.g., `GroundTrainer`). Keep CLI arguments aligned with config names, use `utils.set_seed` for reproducibility, and route status updates through the shared `logging` logger so `train.log` stays uniform. Prefer vectorized tensor operations and keep path literals relative to `src/`.

## Testing Guidelines
There is no dedicated unit-test harness, so rely on trainer evaluations. After modifying training, data, or sampling code, run a short job such as `python src/main.py --init config/umls_config.json --max_steps 1000 --valid_steps 200` and inspect the MRR/Hits reported in `outputs/<run>/train.log`. When adding datasets, supply matching `config/*.json` and `data/<dataset>/` folders plus `mined_rules.txt`, then document baseline metrics in your PR.

## Commit & Pull Request Guidelines
Git history favors concise action-first Chinese messages (`创建md文件夹`, `修改trainer文件`, `新增模型论文`); follow the same verb-first style and keep each commit focused. Pull requests must summarize scope, list the dataset/config used, share key metrics or log excerpts, and link related issues. Attach rule files, figures, or papers that changed and mention any environment or hardware assumptions.

## Configuration & Data Tips
Treat JSON configs as the single source of hyperparameters—edit them instead of hard-coding values in Python. Keep large raw resources out of version control; only derived triples and shareable assets belong in `data/`, while intermediate checkpoints stay in `outputs/` or `experimentalData/`. Double-check that relative paths remain valid when commands are executed from within `src/`.
