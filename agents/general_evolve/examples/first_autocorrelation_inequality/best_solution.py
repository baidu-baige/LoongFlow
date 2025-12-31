"""Best solution for first autocorrelation inequality problem."""

import concurrent.futures
import random
import time
from multiprocessing import cpu_count

import numpy as np
import scipy.sparse as sp
from scipy.fft import dct, idct
from scipy.optimize import minimize, linprog


def evaluate_sequence(sequence):
    if not isinstance(sequence, list) or not sequence:
        return np.inf
    for x in sequence:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return np.inf
        if np.isnan(x) or np.isinf(x):
            return np.inf
    sequence = [float(x) for x in sequence]
    sequence = [max(0, x) for x in sequence]
    sequence = [min(1000.0, x) for x in sequence]
    n = len(sequence)
    b_sequence = np.convolve(sequence, sequence)
    max_b = max(b_sequence)
    sum_a = np.sum(sequence)
    if sum_a < 0.01:
        return np.inf
    return float(2 * n * max_b / (sum_a**2))


def solve_convolution_lp_sparse(f_sequence, rhs):
    n = len(f_sequence)
    c = -np.ones(n)

    row_indices = []
    col_indices = []
    data_vals = []
    b_ub = []

    for k in range(2 * n - 1):
        b_ub.append(rhs)
        for i in range(max(0, k - n + 1), min(n, k + 1)):
            j = k - i
            if 0 <= j < n:
                row_indices.append(k)
                col_indices.append(j)
                data_vals.append(f_sequence[i])

    A_ub = sp.coo_matrix(
        (data_vals, (row_indices, col_indices)), shape=(2 * n - 1, n)
    ).tocsr()
    bounds = [(0, None)] * n
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if result.success:
        return result.x
    else:
        return None


def gradient_optimization_worker(initial_seq):
    n = len(initial_seq)
    bounds = [(0, 1000)] * n

    def objective(x):
        return evaluate_sequence(x.tolist())

    res = minimize(
        objective,
        np.array(initial_seq),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 50, "ftol": 1e-6},
    )
    if res.success:
        return res.x.tolist(), res.fun
    return initial_seq, evaluate_sequence(initial_seq)


def search_for_best_sequence():
    total_time = 1000
    start_time = time.time()
    phase_times = [150, 400, 300, 100, 50]
    phase_end_times = [start_time + sum(phase_times[: i + 1]) for i in range(5)]

    if "best_sequence_prev" in globals():
        current_sequence = globals()["best_sequence_prev"]
    else:
        current_sequence = [1.0] * 600
    best_sequence = current_sequence
    best_score = evaluate_sequence(best_sequence)
    n = len(current_sequence)

    # Phase 1: Fourier Initialization (DCT with Bayesian optimization)
    if time.time() < phase_end_times[0]:
        try:
            dct_coeffs = dct(current_sequence, norm="ortho")
            power_spectrum = np.abs(dct_coeffs) ** 2
            dominant_count = max(1, int(0.05 * n))
            dominant_idx = np.argsort(power_spectrum)[-dominant_count:]

            def dct_objective(opt_coeffs):
                new_dct_coeffs = dct_coeffs.copy()
                new_dct_coeffs[dominant_idx] = opt_coeffs
                new_seq = idct(new_dct_coeffs, norm="ortho")
                new_seq = np.clip(new_seq, 0, 1000).tolist()
                return evaluate_sequence(new_seq)

            bounds = [
                (min(dct_coeffs[i], -1000), max(dct_coeffs[i], 1000))
                for i in dominant_idx
            ]
            best_coeffs = None
            best_coeffs_score = np.inf

            for restart in range(10):
                if time.time() >= phase_end_times[0]:
                    break
                x0 = [dct_coeffs[i] * random.uniform(0.8, 1.2) for i in dominant_idx]
                res = minimize(
                    dct_objective,
                    x0,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": 5, "maxfun": 10},
                )
                if res.success and res.fun < best_coeffs_score:
                    best_coeffs_score = res.fun
                    best_coeffs = res.x

            if best_coeffs is not None and best_coeffs_score < best_score:
                new_dct_coeffs = dct_coeffs.copy()
                new_dct_coeffs[dominant_idx] = best_coeffs
                new_seq = idct(new_dct_coeffs, norm="ortho")
                new_seq = np.clip(new_seq, 0, 1000).tolist()
                new_score = evaluate_sequence(new_seq)
                if new_score < best_score:
                    best_sequence = new_seq
                    best_score = new_score
                    current_sequence = best_sequence
        except Exception as e:
            pass

    # Phase 2: Parallel Gradient Refinement
    if time.time() < phase_end_times[1]:
        try:
            num_workers = min(8, cpu_count())
            num_points = 24
            sequences = []
            for _ in range(num_points):
                perturbed = [x * random.uniform(0.97, 1.03) for x in current_sequence]
                sequences.append(perturbed)

            candidates = []
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=num_workers
            ) as executor:
                futures = [
                    executor.submit(gradient_optimization_worker, seq)
                    for seq in sequences
                ]
                for future in concurrent.futures.as_completed(futures):
                    if time.time() >= phase_end_times[1]:
                        break
                    candidate_seq, candidate_score = future.result()
                    candidates.append((candidate_seq, candidate_score))

            for candidate_seq, candidate_score in candidates:
                if candidate_score < best_score:
                    best_sequence = candidate_seq
                    best_score = candidate_score
                    current_sequence = best_sequence
        except Exception as e:
            pass

    # Phase 3: Adaptive LP Fusion
    if time.time() < phase_end_times[2]:
        t = 0.15
        improvement_threshold = 1e-6
        consecutive_no_improve = 0
        max_no_improve = 10

        while (
            time.time() < phase_end_times[2] and consecutive_no_improve < max_no_improve
        ):
            try:
                sum_a = np.sum(current_sequence)
                if sum_a < 0.01:
                    break
                normalized_seq = [x * np.sqrt(2 * n) / sum_a for x in current_sequence]
                rhs = max(np.convolve(normalized_seq, normalized_seq))

                g_seq = solve_convolution_lp_sparse(normalized_seq, rhs)
                if g_seq is None:
                    consecutive_no_improve += 1
                    t = max(0.01, t * 0.8)
                    continue

                sum_g = np.sum(g_seq)
                if sum_g < 0.01:
                    consecutive_no_improve += 1
                    t = max(0.01, t * 0.8)
                    continue

                normalized_g = [x * np.sqrt(2 * n) / sum_g for x in g_seq]
                candidate_seq = [
                    (1 - t) * x + t * y for x, y in zip(current_sequence, normalized_g)
                ]
                candidate_score = evaluate_sequence(candidate_seq)

                if candidate_score < best_score - improvement_threshold:
                    best_sequence = candidate_seq
                    best_score = candidate_score
                    current_sequence = best_sequence
                    consecutive_no_improve = 0
                    t = min(0.3, t * 1.1)
                else:
                    consecutive_no_improve += 1
                    t = max(0.01, t * 0.8)
            except Exception as e:
                consecutive_no_improve += 1
                t = max(0.01, t * 0.8)

    # Phase 4: Variance-Targeted Polish
    if time.time() < phase_end_times[3]:
        try:
            window_size = 60
            seq_arr = np.array(current_sequence)
            local_var = np.zeros(n)
            for i in range(n):
                start = max(0, i - window_size // 2)
                end = min(n, i + window_size // 2 + 1)
                local_var[i] = np.var(seq_arr[start:end])

            n_refine = min(30, n)
            refine_indices = np.argsort(local_var)[-n_refine:]

            candidate_seq = current_sequence.copy()
            candidate_score = best_score
            for idx in refine_indices:
                if time.time() >= phase_end_times[3]:
                    break
                original_val = candidate_seq[idx]
                best_val = original_val
                best_val_score = candidate_score
                for factor in [0.85, 0.92, 1.0, 1.08, 1.15]:
                    new_val = original_val * factor
                    if new_val < 0 or new_val > 1000:
                        continue
                    test_seq = candidate_seq.copy()
                    test_seq[idx] = new_val
                    test_score = evaluate_sequence(test_seq)
                    if test_score < best_val_score:
                        best_val_score = test_score
                        best_val = new_val
                if best_val != original_val:
                    candidate_seq[idx] = best_val
                    candidate_score = best_val_score
            if candidate_score < best_score:
                best_sequence = candidate_seq
                best_score = candidate_score
                current_sequence = best_sequence
        except Exception as e:
            pass

    # Phase 5: Ensemble Validation
    if time.time() < phase_end_times[4]:
        try:
            weights = [0.6, 0.25, 0.15]
            sequences = [
                current_sequence,
                globals().get("best_sequence_prev", [1.0] * n),
                [1.0] * n,
            ]
            weighted_seq = np.zeros(n)
            for i in range(3):
                weighted_seq += weights[i] * np.array(sequences[i])
            ensemble_seq = weighted_seq.tolist()
            ensemble_score = evaluate_sequence(ensemble_seq)

            if ensemble_score < best_score:
                best_sequence = ensemble_seq
                best_score = ensemble_score
                current_sequence = best_sequence

            def nelder_mead_obj(x):
                return evaluate_sequence(x.tolist())

            x0 = np.array(current_sequence)
            res = minimize(
                nelder_mead_obj,
                x0,
                method="Nelder-Mead",
                options={"maxiter": 20, "xatol": 1e-4, "fatol": 1e-4},
            )
            if res.success:
                polished_seq = np.clip(res.x, 0, 1000).tolist()
                polished_score = evaluate_sequence(polished_seq)
                if polished_score < best_score:
                    best_sequence = polished_seq
                    best_score = polished_score
        except Exception as e:
            pass

    if any(x < 0 for x in best_sequence) or sum(best_sequence) < 0.01:
        uniform_seq = [1.0] * n
        uniform_score = evaluate_sequence(uniform_seq)
        if uniform_score < best_score:
            best_sequence = uniform_seq

    return best_sequence


def run_search_for_best_sequence():
    best_sequence = search_for_best_sequence()
    return best_sequence


if __name__ == "__main__":
    for i in range(100):
        best_sequence = run_search_for_best_sequence()
        c1 = evaluate_sequence(best_sequence)
        print("C_upper_bound:", c1)
        if c1 < 1.5098 or c1 < 1.5053:
            print(f"Best sequence: {best_sequence}")
