""" Best solution search for Erdős' minimum overlap problem """
import numpy as np
from scipy.optimize import minimize


# Parent solution's best half sequence (pre-optimized starting point)
parent_best_half = np.array([
    4.035139886429729e-17, 7.16960195434492e-17, 5.9078979812056395e-18, 0.0,
    1.779398806690346e-18, 0.0, 1.9648993913784086e-18, 8.376027783342738e-19,
    7.567529007107377e-17, 1.3737268666645326e-17, 1.7936917242345572e-17, 6.938798242790737e-17,
    4.725247071440962e-17, 7.915964732761881e-17, 6.351500634791069e-17, 2.0699010773544334e-17,
    2.6519373699769035e-17, 1.5454790363366356e-18, 3.223594581606167e-17, 2.821227314312901e-17,
    3.014843414571012e-17, 1.1226369280282843e-21, 5.971472267868357e-18, 1.0795898755943663e-17,
    0.0, 1.2419073540967882e-18, 2.126775330057013e-19, 4.8764366272699484e-17, 1.0120893156030333e-19,
    7.179428324157126e-18, 5.522071798137353e-17, 2.912877840990052e-17, 4.3729924938364105e-21,
    1.509563104180246e-17, 4.5761285629589346e-17, 6.401006849519653e-17, 2.0441796758068278e-17,
    5.1607127045172663e-17, 1.8073424034043397e-18, 1.5158543049993332e-18, 0.05012544636362727,
    0.2770494695049894, 0.3773797894645708, 0.4057721537971578, 0.4957311778395181, 0.518698099393886,
    0.6295096182050749, 0.782339766362895, 0.7117532049263942, 0.7417185411203764, 0.8165817685850314,
    0.8377922613832232, 0.8487537828039374, 0.7252405534246363, 0.6041913622575344, 0.7462312473539685,
    0.8689931510414327, 0.706364345047413, 0.8254857503923212, 0.9046222788682133, 0.8852455294962013,
    0.8185114118200576, 0.7564109634753666, 0.6358110514974646, 0.4010093533284147, 0.06634163673127796,
    8.147386635793247e-18, 0.02142652955383468, 0.0016482658596185483, 9.1344518783959e-19,
    2.101306301196181e-17, 1.2269958158820573e-17, 8.650307927916947e-18, 3.23994518773283e-17,
    0.010456397558342942, 0.11873731206359764, 0.1951763674427824, 0.3161107155819945,
    0.5999676496135841, 0.696399821928383, 0.7687792348827345, 0.8674194249660626, 0.7606116773936508,
    0.6915172672090195, 0.7926038882540918, 0.7919896621964436, 0.6615960964769708, 0.6051194459216244,
    0.5220586543803861, 0.3441057785170132, 0.4139628862680378, 0.4265108327381685, 0.38488624532326826,
    0.565417853419319, 0.4294828280170623, 0.5162984111124105, 0.5209929451993578, 0.6508170216130884,
    0.6247381974195462, 0.7541512417811539, 0.7103704054479334, 0.7419921889867561, 0.8680862239418141,
    0.8128643114717964, 0.7125341016424233, 0.5752673383885315, 0.6655589825057584, 0.5563098723145047,
    0.4950290856243467, 0.4612458525563718, 0.6128147196095701, 0.4314904053003348, 0.3939597603066263,
    0.40820415819954775, 0.5004853235053063, 0.4592380122851127, 0.3291565170109253, 0.358634915869889,
    0.5057849088228203, 0.6448227599612589, 0.46643342535626864, 0.5282102848921744, 0.5439547837073755,
    0.5119645360539827, 0.6026669129464475, 0.7901707840838204, 0.9117967277463173, 0.720399466622046,
    0.7216851591960946, 0.8209442017030795, 0.93948226771011, 0.9071728346976791, 0.6190082692414445,
    0.5585285329088203, 0.650517278912133, 0.6950231035140447, 0.6977283404017327, 0.8574329274560492,
    0.7653720379145755, 0.6853017063532518, 0.8342378898156905, 0.8590852850639218, 0.9373544686053968,
    0.8321410691083152, 0.7171358552459666, 0.6036368741268432, 0.6878584277332989, 0.6763776008943667,
    0.7378676516621554, 0.7961071396882138, 0.5983884009847601, 0.7646870189243954, 0.7510700075908905,
    0.706867309027408, 0.8742885153276346, 0.9954447111937649, 0.9997326334996035, 0.9766702195431387,
    0.8804535917537768, 0.8328949102345362, 0.8387913531466297, 1.0, 0.9628577893223523,
    0.8280289801533194, 0.7718044166482061, 0.7442459128492331, 0.745819692333584, 0.9077213974775252,
    0.9347838952218003, 0.8552113037077053, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9260992085733399,
    0.9440466781966186, 1.0])


def initialize_solution() -> np.ndarray:
    """
    Initialize the solution by adding small random noise to the known best half sequence.
    Returns:
        np.ndarray: A sequence clipped between 0 and 1.
    """
    base_seq = parent_best_half.copy()
    noise = np.random.uniform(-0.1, 0.1, size=base_seq.shape)
    return np.clip(base_seq + noise, 0, 1)


def repair_sequence(genes: np.ndarray) -> np.ndarray:
    """
    Highly robust repair function that ensures the total sum of the full sequence 
    is exactly n/2 and all values stay within [0, 1].
    
    Args:
        genes: The half-sequence to be repaired.
    Returns:
        np.ndarray: The repaired half-sequence.
    """
    n_half = len(genes)
    full_len = 2 * n_half - 1
    target_sum = full_len / 2.0
    
    # Force clip values to [0, 1]
    res = np.clip(genes, 0, 1)
    
    # Iterative projection method to fix the sum constraint
    for _ in range(10):
        # Calculate current sum based on symmetric property
        current_sum = 2 * np.sum(res[:-1]) + res[-1]
        diff = target_sum - current_sum
        if abs(diff) < 1e-13:
            break
        
        # Identify elements that can still be adjusted (not yet at boundaries)
        if diff > 0:  # Need to increase values
            can_adjust = np.where(res < 1.0, 1.0, 0.0)
        else:  # Need to decrease values
            can_adjust = np.where(res > 0.0, 1.0, 0.0)
            
        # Weights: Left part elements appear twice in the full sequence, 
        # the Center element (last in half_seq) appears once.
        weights = can_adjust.copy()
        weights[:-1] *= 2.0
        
        total_w = np.sum(weights)
        if total_w < 1e-12: 
            break  # Cannot adjust further
        
        step = diff / total_w
        res += can_adjust * step
        res = np.clip(res, 0, 1)
        
    return res


def minimax_objective(half_seq: np.ndarray, t: float = 250.0) -> float:
    """
    Minimax objective function using Log-Sum-Exp to smoothly approximate the Max overlap.
    Includes a penalty term to enforce the sum constraint during optimization.
    
    Args:
        half_seq: The half-sequence variables being optimized.
        t: Smoothness parameter for Log-Sum-Exp (higher is more accurate but less smooth).
    Returns:
        float: The calculated objective value.
    """
    reversed_seq = half_seq[::-1]
    full_seq = np.concatenate((half_seq[:-1], reversed_seq))
    
    # Compute self-convolution (overlap)
    conv = np.correlate(full_seq, 1 - full_seq, mode='full')
    max_val = np.max(conv)
    
    # Log-Sum-Exp trick for numerical stability
    shifted_conv = conv - max_val
    exp_conv = np.exp(t * shifted_conv)
    log_sum_exp = max_val + np.log(np.sum(exp_conv)) / t
    
    # Auxiliary penalty term for the sum constraint
    target_sum = len(full_seq) / 2.0
    current_sum = 2 * np.sum(half_seq[:-1]) + half_seq[-1]
    penalty = 2000 * (current_sum - target_sum)**2
    
    return log_sum_exp + penalty


def compute_upper_bound(half_seq: np.ndarray) -> float:
    """
    Calculates the actual normalized upper bound of the sequence overlap.
    
    Args:
        half_seq: The half-sequence.
    Returns:
        float: Max overlap normalized by sequence length.
    """
    reversed_seq = half_seq[::-1]
    full_seq = np.concatenate((half_seq[:-1], reversed_seq))
    conv = np.correlate(full_seq, 1 - full_seq, mode='full')
    return np.max(conv) / len(full_seq) * 2


def trust_region_optimization(solution: np.ndarray, resolution: int) -> np.ndarray:
    """
    Performs local optimization using the SLSQP method within a trust region.
    Supports resolution scaling via interpolation.
    
    Args:
        solution: Current solution sequence.
        resolution: Target number of elements in the half-sequence.
    Returns:
        np.ndarray: Optimized and repaired sequence.
    """
    if len(solution) != resolution:
        # Interpolate to new resolution
        x_old = np.linspace(0, 1, len(solution))
        x_new = np.linspace(0, 1, resolution)
        solution = np.interp(x_new, x_old, solution)
        solution = repair_sequence(solution)  # Repair immediately after interpolation
    
    # Define bounds for trust region
    bound_range = 0.10 + (resolution - 100) * 0.002
    bounds = [(max(0, x - bound_range), min(1, x + bound_range)) for x in solution]
    
    # Define the sum constraint for SLSQP
    def constraint_func(x):
        full_len = 2 * len(x) - 1
        return 2 * np.sum(x[:-1]) + x[-1] - full_len / 2
    
    result = minimize(
        minimax_objective,
        solution,
        method='SLSQP',
        bounds=bounds,
        constraints={'type': 'eq', 'fun': constraint_func},
        options={'maxiter': 500 + (resolution - 100) * 10, 'ftol': 1e-14}
    )
    
    # Final repair to ensure constraints are strictly met regardless of solver success
    return repair_sequence(result.x)


def optimize_resolutions(solution: np.ndarray) -> np.ndarray:
    """
    Progressively optimizes the solution through multiple resolutions.
    
    Args:
        solution: Initial solution.
    Returns:
        np.ndarray: Best sequence found across all resolutions.
    """
    resolutions = [100, 120, 140, 160, 180]
    best_solution = repair_sequence(solution.copy())
    best_ub = compute_upper_bound(best_solution)
    
    for res in resolutions:
        solution = trust_region_optimization(solution, res)
        new_ub = compute_upper_bound(solution)
        
        if new_ub < best_ub:
            best_solution, best_ub = solution.copy(), new_ub
            
        # Early exit if target performance is reached
        if best_ub < 0.380927:
            print(f"Target achieved! UB: {best_ub:.8f} at resolution {res}")
            return repair_sequence(best_solution)
    
    return repair_sequence(best_solution)


def generate_erdos_data() -> np.ndarray:
    """
    Main entry point for generating and optimizing a single candidate sequence.
    
    Returns:
        np.ndarray: Optimized half-sequence.
    """
    current = initialize_solution()
    optimized = optimize_resolutions(current)
    return optimized


def verify_sequence(sequence: np.ndarray):
    """
    Asserts that the sequence satisfies all mathematical constraints of the problem.
    
    Args:
        sequence: The full symmetric sequence.
    Raises:
        AssertionError: If values are out of bounds or sum is incorrect.
    """
    # Check value bounds
    if not np.all((sequence >= -1e-12) & (sequence <= 1.0 + 1e-12)):
        raise AssertionError("All values in the sequence must be between 0 and 1.")
    
    n = len(sequence)
    target_val = n / 2.0
    current_sum = np.sum(sequence)
    
    # Check sum constraint
    if not np.isclose(current_sum, target_val, rtol=1e-7, atol=1e-7):
        raise AssertionError(
            f"Constraint failed: Sum is {current_sum:.10f}, expected {target_val:.10f}. "
            f"Diff: {abs(current_sum - target_val):.2e}")
    
    print(f"Verification passed: Length={n}, Sum={current_sum:.4f}")


if __name__ == "__main__":
    num_trials = 50000
    best_global_ub = float('inf')
    best_global_seq = None

    for i in range(num_trials):
        print(f"Trial {i + 1}...")
        best_half = generate_erdos_data()
        
        # Construct the full symmetric sequence
        reversed_half = best_half[::-1]
        final_seq = np.concatenate((best_half[:-1], reversed_half))

        try:
            verify_sequence(final_seq)
            ub = compute_upper_bound(best_half)
            print(f"Trial {i + 1} upper bound: {ub:.15f}")
            
            # Track the best sequence found across all trials
            if ub < best_global_ub:
                best_global_ub = ub
                best_global_seq = final_seq.copy()
                print(f"New best: {ub:.8f}")
                print(f"Sequence: {best_global_seq.tolist()}")
        except AssertionError as e:
            print(f"Validation Error: {e}")

    print(f"Final Best UB: {best_global_ub:.8f}")
