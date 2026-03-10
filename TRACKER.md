# TRACKER

## Objective

Reproduce the LoongFlow Kaggle / MLE-Bench benchmark setting in a way that is trackable, auditable, and easy to inspect, including:
- locating the benchmark entrypoints and configs,
- reproducing the benchmark environment as faithfully as practical,
- executing at least one representative benchmark run,
- documenting fidelity, deviations, and remaining gaps with evidence.

## Current status

Research stage completed for the repo-local benchmark map. First expensive setup attempt completed and failed on an external environment prerequisite.

Current benchmark structure identified:
- Canonical Kaggle benchmark entrypoint: `./run_mlebench.sh`
- Environment bootstrap: `run_mlebench.sh init`
- Dataset preparation: `run_mlebench.sh prepare <competition_id>`
- Benchmark run: `run_mlebench.sh run <competition_id> [--background] [--debug]`
- Runner launched by shell script: `agents/ml_agent/ml_evolve_agent.py`
- Benchmark config template copied per competition: `agents/ml_agent/examples/mlebench/task_config.yaml`
- Final benchmark evaluator: `agents/ml_agent/examples/mlebench/eval_program.py`
- Competition assets / packaged reference solutions: `agents/ml_agent/examples/mlebench/competitions/*/*`

Evaluation structure identified:
- LoongFlow wraps MLE-Bench locally rather than submitting to Kaggle during normal evaluation.
- Final evaluation scores OOF predictions with MLE-Bench's grader and normalizes the raw score against the competition leaderboard.
- Optional test scoring is only enabled when `MLEBENCH_COMPUTE_TEST_SCORE=true`.

Important repo-local discrepancies already found:
- Project instructions say Python `3.12+`, but ML environment manifests pin Python `3.11`.
- `run_mlebench.sh` requires `mamba`; current machine has `conda` but no `mamba`.
- Current workflow evaluator requires OOF artifacts (`oof_submission_file_path`, `oof_answer_file_path`, `oof_coverage`), but packaged competition `workflow.py` files sampled so far only return final submission artifacts.
- Kaggle credentials are not present at `~/.kaggle/kaggle.json`.
- Host has 8x NVIDIA L20Z GPUs, so the canonical script will choose the GPU environment by default.
- Conda environment creation is blocked by unaccepted Terms of Service for `https://repo.anaconda.com/pkgs/main` and `https://repo.anaconda.com/pkgs/r`.

## Completed steps

- Located the benchmark entrypoints, setup scripts, task config template, evaluator, and ML agent runner.
- Mapped the end-to-end shell pipeline from `run_mlebench.sh`.
- Confirmed the final evaluator contract in `agents/ml_agent/examples/mlebench/eval_program.py`.
- Confirmed the ML executor stage order: `load_data -> get_splitter -> preprocess -> train_and_predict -> ensemble -> workflow`.
- Inspected the local machine state relevant to reproduction: OS, Python, conda, git-lfs, missing `uv`, missing `mamba`.
- Verified that the repo currently contains 48 packaged MLE-Bench competition solution directories.
- Identified a likely upstream drift between packaged benchmark workflows and the current evaluator contract.

## In-progress step

Resolve the Conda Terms-of-Service blocker for the canonical benchmark environment init, or decide on a documented divergence from the upstream channel configuration.

## Next 3 steps

1. Decide whether to accept Conda channel Terms of Service for the upstream `defaults` channels or remove those channels as a documented fidelity deviation.
2. Re-run `./run_mlebench.sh init` after the ToS/channel decision.
3. If init succeeds, attempt `./run_mlebench.sh prepare <competition_id>` and record whether Kaggle credentials become the next blocker.

## Known blockers

- No LLM API credentials are configured yet in the benchmark task config template.
- `run_mlebench.sh init` required a local compatibility patch because `mamba` is not installed.
- `run_mlebench.sh init` is currently blocked by Conda Terms-of-Service acceptance for `defaults` channels.
- MLE-Bench dataset preparation will likely require Kaggle API credentials and network access.
- Kaggle credentials are currently absent on this host.
- Packaged competition solutions appear to use hard-coded absolute data/output paths from the authors' environment.
- The evaluator/workflow artifact contract mismatch may block faithful reuse of packaged competition workflows without controlled adjustments.

## Reproduction fidelity notes

- Exact reproduction requirements identified so far:
  - use `run_mlebench.sh` as the benchmark shell entrypoint,
  - use the MLE-Bench-backed dataset preparation flow,
  - use `agents/ml_agent/examples/mlebench/eval_program.py` for final scoring,
  - preserve per-competition task configs under `output/mlebench/<competition_id>/task_config.yaml`.
- Current approximations:
  - added a minimal `conda` fallback to `run_mlebench.sh` so the canonical shell path can run on this host without `mamba`.
- Current failed reproduction attempt:
  - `./run_mlebench.sh init` started successfully but stopped before dependency resolution because Conda refused access to the upstream `defaults` channels until Terms of Service are accepted.
- Known uncertainties:
  - whether the current repo state is internally consistent enough to rerun packaged competition workflows unchanged,
  - whether a faithful run requires external MLE-Bench data preparation plus live LLM access,
  - whether a representative reproduction should target the full agent loop or the evaluator-backed workflow stage only if credentials are unavailable.
