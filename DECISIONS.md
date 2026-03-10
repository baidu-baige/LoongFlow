# DECISIONS

## Decision 1

Status: provisional

Use `./run_mlebench.sh` as the canonical benchmark entrypoint for this reproduction rather than `./run_ml.sh`.

Reason:
- `run_mlebench.sh` is the dedicated shell path for MLE-Bench / Kaggle competitions.
- It performs benchmark-specific setup: cloning `openai/mle-bench`, preparing competition data, copying the benchmark task config template, and optionally grading the latest submission with `mlebench grade-sample`.

Implication:
- Reproduction fidelity will be judged primarily against the `run_mlebench.sh` path, not the generic ML demo path.

## Decision 2

Status: provisional

Do not start any expensive benchmark run until the benchmark structure, evaluator contract, and environment blockers are recorded in repo-local tracking files.

Reason:
- The task explicitly prioritizes observability and supervisor inspection over speed.
- The repo already shows potentially important inconsistencies that need to be acknowledged first.

## Decision 3

Status: provisional

Treat the final evaluator contract in `agents/ml_agent/examples/mlebench/eval_program.py` as the ground truth for current reproduction, even if packaged competition solutions appear older or inconsistent.

Reason:
- This file is what `run_mlebench.sh run ...` passes to `agents/ml_agent/ml_evolve_agent.py`.
- The ML executor and workflow-stage validator both now expect OOF evaluation artifacts.

Known consequence:
- Some packaged competition `workflow.py` files may not be directly replayable without adjustment.
- Any such adjustment will be documented as a deviation from historical artifacts, not as faithful upstream behavior.

## Decision 4

Status: applied

Add a minimal `conda` fallback to `run_mlebench.sh` instead of creating a new wrapper script.

Reason:
- The canonical benchmark entrypoint should remain `./run_mlebench.sh`.
- The host has `conda` installed but not `mamba`.
- The script was already partially `conda`-aware for activation and grading, so the smallest coherent change was to reuse that path for env existence checks and env creation.

Scope of change:
- Added `get_env_manager()` helper.
- Switched environment existence checks and environment creation from hard-coded `mamba` calls to `mamba`-or-`conda`.
- Updated the post-init activation hint to `conda activate loongflow_ml`.

Fidelity impact:
- Low. The benchmark shell entrypoint and its semantics remain unchanged; only the environment manager backend can differ when `mamba` is unavailable.

## Decision 5

Status: pending external input

Do not auto-accept Conda channel Terms of Service on behalf of the user.

Reason:
- Environment creation is now blocked by Conda's legal/TOS gate for the `defaults` channels.
- Accepting third-party terms is an external dependency step rather than a normal repo modification.

Fallback if not accepted:
- Remove `defaults` from the environment manifests and attempt a `conda-forge`-only environment as a documented fidelity deviation.

Fidelity impact if fallback is used:
- Medium. Channel set would differ from upstream, which may affect package resolution and benchmark behavior.

## Exact reproduction requirements

- Use the LoongFlow repository's own benchmark shell entrypoint and evaluator.
- Reproduce the environment assumptions required by `run_mlebench.sh` as closely as practical.
- Use MLE-Bench-prepared competition data under `output/mlebench/<competition_id>/prepared/public`.
- Preserve command lines, logs, and resulting artifacts in human-readable form.

## Optional optimizations

- Add minimal logging or wrapper automation only if it improves auditability without changing benchmark semantics.
- Add a small compatibility patch for `conda` fallback only if the benchmark is otherwise blocked by missing `mamba`.

## Unknown / underspecified parts

- Which exact repository revision produced the published 48-competition medal table.
- Whether packaged competition solution directories are meant as exact replay artifacts or illustrative final code snapshots.
- Whether current benchmark reproduction is expected to include live LLM generation or whether replaying a representative packaged benchmark workflow is acceptable when credentials are unavailable.

## Observed deviations from documentation

- Repo instructions in `AGENTS.md` state Python `3.12+`, but the ML environment YAML pins Python `3.11`.
- Repo instructions emphasize `uv`, but benchmark scripts use conda/mamba-based environments instead.
- Current machine has `conda` but not `mamba`, while `run_mlebench.sh init` hard-fails without `mamba`.
- Kaggle credentials are absent on this host, so benchmark data preparation is expected to be blocked unless credentials are added later.
- Conda blocks the current env creation until the `defaults` channel Terms of Service are accepted.
