def compute_epsilon(step: int, max_explore_steps: int, initial_epsilon: float, final_epsilon: float):
    assert 0 <= step
    delta_epsilon = step / max_explore_steps * (initial_epsilon - final_epsilon)
    epsilon = initial_epsilon - delta_epsilon
    return max(epsilon, final_epsilon)
