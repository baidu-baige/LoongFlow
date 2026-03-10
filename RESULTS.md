# RESULTS

## Benchmark outcomes

| task name | configuration | hardware used | model/API used | wall-clock time | success/failure | main metrics | notes on divergence from LoongFlow |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Not run yet | Research phase only | Linux host `dsw-383994-58767c9899-2q7sk`; Python 3.13.11 outside benchmark env | Not configured | N/A | Not started | N/A | No benchmark execution yet; environment and evaluator mismatches still being resolved |
| Environment init (`run_mlebench.sh init`) | Canonical GPU path with `conda` fallback patch; upstream GPU env manifest | 8x NVIDIA L20Z; system Python 3.13.11; `conda` 25.11.1 | N/A | ~4s | Failed | No environment created | Blocked before solve/install because Conda requires accepting Terms of Service for upstream `defaults` channels |
| `detecting-insults-in-social-commentary` prepare | Canonical `run_mlebench.sh prepare` path in `loongflow_ml` | 8x NVIDIA L20Z; `loongflow_ml` active via script | N/A | ~6s | Failed | No dataset prepared | MLE-Bench launched correctly but Kaggle auth failed because it expected `kaggle.json` in `/newcpfs/lxh/vibe-kanban/config/kaggle/` |

## Expected evaluation behavior

- Final benchmark score: normalized OOF score in `[0.0, 1.0]` from `agents/ml_agent/examples/mlebench/eval_program.py`
- Optional debug artifact: `submission_test_result.json` when `MLEBENCH_COMPUTE_TEST_SCORE=true`
- Likely benchmark success evidence:
  - prepared competition data under `output/mlebench/<competition_id>/prepared/public`,
  - benchmark task config under `output/mlebench/<competition_id>/task_config.yaml`,
  - executor outputs and evaluation results under `output/`,
  - `agent.log` and `output/logs/evolux.log`,
  - graded or normalized score captured in evaluator artifacts.
