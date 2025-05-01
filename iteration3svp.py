#imports CA
import numpy as np
import time
from sklearn.decomposition import PCA
import plotly.graph_objs as go

# functions AC

def basis_condition_score(B):
#simple measure of basis conditioning.
    B_T_B = B.T @ B
    eigvals = np.linalg.eigvals(B_T_B)
    return np.max(eigvals) / np.min(eigvals)

def naive_svp_solver(B, max_range=2):
#naive brute-force SVP solver
    n = B.shape[1]
    shortest_vec = None
    min_norm = float('inf')
    
    for coeffs in np.ndindex(*((2*max_range+1,) * n)):
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

def pruned_randomized_greedy_svp_with_local_search(B, max_trials=5000, initial_radius=10.0):
#randomized greedy SVP solver + Local Search Refinement AC
    n = B.shape[1]
    best_vec = None
    best_norm = float('inf')
    radius = initial_radius

#random Sampling with Pruning AC
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
                radius = norm * 1.2  # shrink search focus AC

#local search around best_coeffs AC
    for delta in np.ndindex(*((3,) * n)):  # explore neighbors in [-1, 0, 1] AC
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

#lattice setup AC
n = 4  # 4D lattice AC
B = np.random.randint(-5, 6, size=(n, n))

print("Basis matrix B:\n", B)

#basis quality AC
score = basis_condition_score(B)
print(f"Basis condition score (higher = worse): {score:.2f}")

#solve with naive solver AC
start_time = time.time()
shortest_naive, coeffs_naive, norm_naive = naive_svp_solver(B, max_range=2)
time_naive = time.time() - start_time

#solve with my improved algorithm AC
start_time = time.time()
shortest_unique, coeffs_unique, norm_unique = pruned_randomized_greedy_svp_with_local_search(B, max_trials=5000, initial_radius=10.0)
time_unique = time.time() - start_time

#print results AC
print("\n--- Naive Brute-Force Solver (limited search) ---")
print(f"Shortest vector: {shortest_naive}")
print(f"Coefficients: {coeffs_naive}")
print(f"Norm length: {norm_naive:.3f}")
print(f"Time taken: {time_naive:.3f} seconds")

print("\n--- Pruned Randomized Greedy + Local Search Solver (My Algorithm) ---")
print(f"Shortest vector: {shortest_unique}")
print(f"Coefficients: {coeffs_unique}")
print(f"Norm length: {norm_unique:.3f}")
print(f"Time taken: {time_unique:.3f} seconds")

#visualization of lattice and shortest vectors AC
#generate lattice points for plotting AC
sample_vectors = []
max_range = 3  #larger plot range AC

for coeffs in np.ndindex(*((2*max_range+1,) * n)):
    coeffs = np.array(coeffs) - max_range
    sample_vectors.append(B @ coeffs)

sample_vectors = np.array(sample_vectors)

#reduce dimension to 3D for plotting AC
pca_3d = PCA(n_components=3)
data_3d = pca_3d.fit_transform(sample_vectors)

shortest_naive_proj = pca_3d.transform(shortest_naive.reshape(1, -1))
shortest_unique_proj = pca_3d.transform(shortest_unique.reshape(1, -1))

#plot AC
fig = go.Figure()

#lattice points
fig.add_trace(go.Scatter3d(
    x=data_3d[:, 0], y=data_3d[:, 1], z=data_3d[:, 2],
    mode='markers', marker=dict(size=3, color='blue'),
    name='Lattice Points'
))

#naive shortest vector AC
fig.add_trace(go.Scatter3d(
    x=[0, shortest_naive_proj[0, 0]],
    y=[0, shortest_naive_proj[0, 1]],
    z=[0, shortest_naive_proj[0, 2]],
    mode='lines+markers',
    marker=dict(size=5, color='red'),
    line=dict(width=5, color='red'),
    name='Naive brute-force Shortest Vector'
))

#unique shortest vector AC
fig.add_trace(go.Scatter3d(
    x=[0, shortest_unique_proj[0, 0]],
    y=[0, shortest_unique_proj[0, 1]],
    z=[0, shortest_unique_proj[0, 2]],
    mode='lines+markers',
    marker=dict(size=5, color='green'),
    line=dict(width=5, color='green'),
    name='My Shortest Vector (Randomizzed Greedy Agl + Local Search)'
))

fig.update_layout(title='SVP Solvers Comparison in Random 4D Lattice (PCA projected)',
                  scene=dict(xaxis_title='PCA 1',
                             yaxis_title='PCA 2',
                             zaxis_title='PCA 3'),
                  width=1200, height=1200)

fig.show()



"""


# Plot 1: Histogram of shortest vector norms
plt.figure(figsize=(10, 6))
plt.hist(results["naive_norms"], bins=12, alpha=0.6, label='Naive Solver', color='red', edgecolor='black')
plt.hist(results["unique_norms"], bins=12, alpha=0.6, label='Greedy + Local Search', color='green', edgecolor='black')
plt.title("Distribution of Shortest Vector Norms (50 Lattices)")
plt.xlabel("Vector Norm")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 2: Scatter plot comparing both solvers lattice by lattice
plt.figure(figsize=(8, 8))
plt.scatter(results["naive_norms"], results["unique_norms"], color='purple', alpha=0.8)
plt.plot([0, max(results["naive_norms"])], [0, max(results["naive_norms"])], linestyle='--', color='gray', label='y = x')
plt.title("Naive vs. Unique Solver Norms (Per Lattice)")
plt.xlabel("Naive Solver Norm")
plt.ylabel("Greedy + Local Search Norm")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 3: Time comparison boxplot
plt.figure(figsize=(10, 6))
plt.boxplot(time_naive, time_unique, labels=['Naive Solver', 'Greedy + Local Search'])
plt.ylabel("Time (seconds)")
plt.title("Solver Runtime Comparison (50 Trials)")
plt.grid(True)
plt.tight_layout()
plt.show()


import numpy as np
import time
import matplotlib.pyplot as plt

# --- Functions ---

def basis_condition_score(B):
    B_T_B = B.T @ B
    eigvals = np.linalg.eigvals(B_T_B)
    return np.max(eigvals) / np.min(eigvals)

def naive_svp_solver(B, max_range=2):
    n = B.shape[1]
    shortest_vec = None
    min_norm = float('inf')
    
    for coeffs in np.ndindex(*((2*max_range+1,) * n)):
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

def pruned_randomized_greedy_svp_with_local_search(B, max_trials=5000, initial_radius=10.0):
    n = B.shape[1]
    best_vec = None
    best_norm = float('inf')
    radius = initial_radius

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

# --- Experiment Loop ---

def run_trials_and_plot_graphs(trials=200, dim=4):
    results = {
        "naive_norms": [],
        "unique_norms": [],
        "naive_times": [],
        "unique_times": [],
        "condition_scores": [],
        "norm_differences": [],
        "wins_by_unique": 0
    }

    for _ in range(trials):
        B = np.random.randint(-5, 6, size=(dim, dim))
        condition = basis_condition_score(B)
        results["condition_scores"].append(condition)

        t0 = time.time()
        _, _, norm_naive = naive_svp_solver(B)
        t1 = time.time()
        results["naive_norms"].append(norm_naive)
        results["naive_times"].append(t1 - t0)

        t2 = time.time()
        _, _, norm_unique = pruned_randomized_greedy_svp_with_local_search(B)
        t3 = time.time()
        results["unique_norms"].append(norm_unique)
        results["unique_times"].append(t3 - t2)

        if norm_unique < norm_naive:
            results["wins_by_unique"] += 1

        results["norm_differences"].append(norm_naive - norm_unique)

    # Plot: Histogram of Norms
    plt.figure(figsize=(10, 6))
    plt.hist(results["naive_norms"], bins=20, alpha=0.6, label='Naive Solver', color='red', edgecolor='black')
    plt.hist(results["unique_norms"], bins=20, alpha=0.6, label='Greedy + Local Search', color='green', edgecolor='black')
    plt.title("Vector Norm Distribution Across Trials")
    plt.xlabel("Vector Norm Length")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot: Scatter naive vs unique norm
    plt.figure(figsize=(8, 8))
    plt.scatter(results["naive_norms"], results["unique_norms"], alpha=0.7, color='purple')
    plt.plot([0, max(results["naive_norms"])], [0, max(results["naive_norms"])], 'k--', label="y = x")
    plt.xlabel("Naive SVP Norm")
    plt.ylabel("Greedy + Local Search Norm")
    plt.title("Norms Comparison Per Trial")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot: Boxplot of runtimes
    plt.figure(figsize=(8, 6))
    plt.boxplot([results["naive_times"], results["unique_times"]],
                labels=["Naive", "Greedy + Local Search"])
    plt.ylabel("Time (seconds)")
    plt.title("Solver Time Comparison")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot: Norm difference vs condition score
    plt.figure(figsize=(10, 6))
    plt.scatter(results["condition_scores"], results["norm_differences"], alpha=0.6, color='orange')
    plt.axhline(0, linestyle='--', color='black')
    plt.xlabel("Basis Condition Score")
    plt.ylabel("Naive Norm - Your Norm")
    plt.title("Relationship Between Basis Quality and Solver Performance")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    

    return results

# Run the trial analysis and plots
run_trials_and_plot_graphs(trials=200, dim=4)

# --- Imports ---
import numpy as np
import time
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import plotly.graph_objs as go

# --- Functions ---

def basis_condition_score(B):
    B_T_B = B.T @ B
    eigvals = np.linalg.eigvals(B_T_B)
    return np.max(eigvals) / np.min(eigvals)

def naive_svp_solver(B, max_range=2):
    n = B.shape[1]
    shortest_vec = None
    min_norm = float('inf')
    for coeffs in np.ndindex(*((2*max_range+1,) * n)):
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

def pruned_randomized_greedy_svp_with_local_search(B, max_trials=5000, initial_radius=10.0):
    n = B.shape[1]
    best_vec = None
    best_norm = float('inf')
    radius = initial_radius

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

# --- Experiment Loop ---
trials = 50
n = 4

basis_scores = []
norm_naive_list = []
norm_yours_list = []

for _ in range(trials):
    B = np.random.randint(-5, 6, size=(n, n))

    score = basis_condition_score(B)
    shortest_naive, _, norm_naive = naive_svp_solver(B, max_range=2)
    shortest_yours, _, norm_yours = pruned_randomized_greedy_svp_with_local_search(B)

    basis_scores.append(score)
    norm_naive_list.append(norm_naive)
    norm_yours_list.append(norm_yours)

import matplotlib.pyplot as plt
import numpy as np

# Ratio (Naive / Your Algorithm)
norm_ratios = np.array(norm_naive_list) / np.array(norm_yours_list)

plt.figure(figsize=(12, 6))
scatter = plt.scatter(
    basis_scores,
    norm_ratios,
    c=(norm_ratios > 1),  # Colored by whether your algorithm wins
    cmap='coolwarm',
    edgecolors='k',
    alpha=0.85
)

plt.axhline(1, linestyle='--', color='black', linewidth=1)
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Basis Condition Score (log scale)")
plt.ylabel("Norm Ratio (Naive / Yours)")
plt.title("Solver Performance vs Basis Quality (Higher Ratio = Your Algorithm Better)")
plt.grid(True, linestyle=':', linewidth=0.6)
plt.tight_layout()
plt.show()

"""