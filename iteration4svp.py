#imports AC
import numpy as np
import time
from sklearn.decomposition import PCA
import plotly.graph_objs as go

#LLL-like reduction AC
def lll_like_reduce(B):
    B = B.copy().astype(float)
    n = B.shape[1]
    for i in range(1, n):
        for j in range(i):
            mu = np.dot(B[:, i], B[:, j]) / np.dot(B[:, j], B[:, j])
            B[:, i] -= round(mu) * B[:, j]
    return B.astype(int)

#condition Score AC
def basis_condition_score(B):
    B_T_B = B.T @ B
    eigvals = np.linalg.eigvals(B_T_B)
    return np.max(eigvals).real / np.min(eigvals).real

#naive Solver AC
def naive_svp_solver(B, max_range=1):
    n = B.shape[1]
    shortest_vec = None
    min_norm = float('inf')
    for coeffs in np.ndindex(*((2 * max_range + 1,) * n)):
        coeffs = np.array(coeffs) - max_range
        if np.all(coeffs == 0):
            continue
        vec = B @ coeffs
        norm = np.linalg.norm(vec)
        if norm < min_norm:
            min_norm = norm
            shortest_vec = vec
            best_coeffs = coeffs
    return shortest_vec, best_coeffs, min_norm

#myalg AC
def pruned_randomized_greedy_svp_with_local_search(B, max_trials=10000, initial_radius=10.0):
    n = B.shape[1]
    best_vec = None
    best_norm = float('inf')
    radius = initial_radius
    best_coeffs = None  # declare upfront

    for trial in range(max_trials):
        coeffs = np.random.randint(-3, 4, size=n)
        if np.all(coeffs == 0):
            continue
        vec = B @ coeffs
        norm = np.linalg.norm(vec)
        if norm < radius:
            if norm < best_norm:
                best_norm = norm
                best_vec = vec
                best_coeffs = coeffs
                radius = norm * 1.2

    #only proceed to local search if found a valid best_coeffs AC
    if best_coeffs is None:
        return None, None, float('inf')

    for delta in np.ndindex(*((3,) * n)):
        offset = np.array(delta) - 1
        neighbor = best_coeffs + offset
        if np.all(neighbor == 0):
            continue
        vec = B @ neighbor
        norm = np.linalg.norm(vec)
        if norm < best_norm:
            best_norm = norm
            best_vec = vec
            best_coeffs = neighbor

    return best_vec, best_coeffs, best_norm


#lattice Setup AC
n = 4 #dimensions
B = np.random.randint(-5, 6, size=(n, n))
print("Original Basis Matrix B:\n", B)
original_score = basis_condition_score(B)
print(f"Condition Score (original): {original_score:.2f}")

#apply LLL-like Reduction pre AC
B_reduced = lll_like_reduce(B)
print("\nReduced Basis Matrix B:\n", B_reduced)
reduced_score = basis_condition_score(B_reduced)
print(f"Condition Score (reduced): {reduced_score:.2f}")

#naive Solver AC
start = time.time()
shortest_naive, coeffs_naive, norm_naive = naive_svp_solver(B_reduced, max_range=1)
time_naive = time.time() - start
print("\n--- Naive Brute-Force Solver (LLL basis) ---")
print(f"Shortest vector: {shortest_naive}")
print(f"Norm length: {norm_naive:.3f}")
print(f"Time taken: {time_naive:.3f} seconds")

#myalg AC
start = time.time()
shortest_unique, coeffs_unique, norm_unique = pruned_randomized_greedy_svp_with_local_search(B_reduced)
time_unique = time.time() - start
print("\n--- Pruned Randomized Greedy + Local Search Solver (LLL basis) ---")
print(f"Shortest vector: {shortest_unique}")
print(f"Norm length: {norm_unique:.3f}")
print(f"Time taken: {time_unique:.3f} seconds")
