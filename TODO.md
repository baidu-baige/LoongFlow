# TODO

## Highest priority

1. Resolve the Conda ToS blocker for `defaults` channels, or explicitly decide to remove `defaults` from the environment manifests as a fidelity tradeoff.
2. Re-run `./run_mlebench.sh init` and capture the full outcome.
3. If init succeeds, choose a small text competition for the first data preparation attempt.

## Next

4. If prepare is blocked by missing Kaggle credentials, document the blocker precisely and decide the best partial reproduction scope.
5. Produce a concrete pre-run plan for the first `prepare` attempt, including chosen competition, expected logs, and success criteria.
6. If benchmark data becomes available, configure LLM credentials and attempt one LoongFlow MLE-Bench agent run.

## If blocked

7. Fall back to a partial but credible reproduction:
   - reproduce environment setup,
   - reproduce evaluator behavior on benchmark artifacts,
   - document exactly why the full agent run cannot be completed.
8. If packaged competition workflows are inconsistent with the current evaluator, isolate the smallest patch or wrapper needed and record it in `DECISIONS.md`.
