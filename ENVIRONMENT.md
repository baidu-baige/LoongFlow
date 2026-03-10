# ENVIRONMENT

## Host snapshot

- Timestamp captured: `2026-03-10T04:59:48Z`
- OS: Linux `5.10.134-013.5.kangaroo.al8.x86_64`
- Hostname: `dsw-383994-58767c9899-2q7sk`
- System Python: `Python 3.13.11`
- `conda`: `25.11.1`
- `git-lfs`: `3.0.2`
- `uv`: not installed
- `mamba`: not installed
- Kaggle credentials: `~/.kaggle/kaggle.json` not present
- GPUs: `8 x NVIDIA L20Z (81559 MiB each)`

## Repo-declared benchmark environment

Benchmark shell scripts use conda-style environment management:
- CPU env manifest: `agents/ml_agent/examples/environment_cpu.yaml`
- GPU env manifest: `agents/ml_agent/examples/environment_gpu.yaml`
- CPU pip requirements: `agents/ml_agent/examples/requirements_cpu.txt`
- GPU pip requirements: `agents/ml_agent/examples/requirements_gpu.txt`
- Environment name expected by scripts: `loongflow_ml`

Notable repo-declared requirements:
- ML env manifest pins `python=3.11`
- `run_mlebench.sh init` requires `mamba`
- `run_mlebench.sh init` clones and installs `openai/mle-bench`
- `run_mlebench.sh prepare` expects Kaggle-backed data preparation through `mlebench prepare`

## Reproduction-impacting mismatches

1. Project-level instructions say Python `3.12+`, but benchmark env YAML pins Python `3.11`.
2. Project instructions prefer `uv`, but benchmark bootstrap is conda/mamba-based.
3. Current host lacks `mamba`, so benchmark init cannot run unchanged.
4. Live benchmark preparation requires Kaggle access, and credentials are currently absent.

## Planned setup path

Planned order:
1. Decide whether to install/use `mamba` or add a minimal `conda` fallback for trackable local reproduction.
2. Install or reuse a benchmark environment matching the repo manifests as closely as practical.
3. Install `openai/mle-bench` in editable mode inside that environment.
4. Prepare one representative competition dataset.
5. Run one representative LoongFlow benchmark attempt with full logging.

Current decision:
- Use a minimal `conda` fallback in `run_mlebench.sh` rather than introducing a separate bootstrap wrapper.

Current setup blocker:
- `conda env create -n loongflow_ml -f agents/ml_agent/examples/environment_gpu.yaml` is blocked until Terms of Service are accepted for:
  - `https://repo.anaconda.com/pkgs/main`
  - `https://repo.anaconda.com/pkgs/r`

## Success criteria for environment reproduction

- Able to activate `loongflow_ml` or documented equivalent.
- Able to import `mlebench` inside the benchmark environment.
- Able to run benchmark preparation for one competition.
- Able to launch `agents/ml_agent/ml_evolve_agent.py` through the LoongFlow shell path or a documented minimal wrapper.
