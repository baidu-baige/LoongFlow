# TODO

## Highest priority

1. Accept the Kaggle rules for `detecting-insults-in-social-commentary` in the browser with the same Kaggle account used for the API token.
2. Re-run `./run_mlebench.sh prepare detecting-insults-in-social-commentary` and capture the full outcome.
3. If prepare succeeds, verify `prepared/public/description.md` and the expected dataset files exist.

## Next

4. If prepare succeeds, run the first tracked benchmark attempt for the same competition.
5. If benchmark data becomes available, configure LLM credentials and attempt one LoongFlow MLE-Bench agent run.
6. After the first simple competition, continue with the remaining planned simple / medium / hard set.

## If blocked

7. Fall back to a partial but credible reproduction:
   - reproduce environment setup,
   - reproduce evaluator behavior on benchmark artifacts,
   - document exactly why the full agent run cannot be completed.
8. If packaged competition workflows are inconsistent with the current evaluator, isolate the smallest patch or wrapper needed and record it in `DECISIONS.md`.
