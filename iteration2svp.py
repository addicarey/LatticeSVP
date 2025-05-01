# --- Imports ---
import numpy as np
import time
from sklearn.decomposition import PCA
import plotly.graph_objs as go

# --- Functions ---

def basis_condition_score(B):
    #Simple measure of basis conditioning.
    B_T_B = B.T @ B
    eigvals = np.linalg.eigvals(B_T_B)
    return np.max(eigvals) / np.min(eigvals)

def naive_svp_solver(B, max_range=2):
    #Naive brute-force SVP solver with limited search range.
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

def pruned_randomized_greedy_svp(B, max_trials=5000, initial_radius=10.0):
#randomiaed SVP solver AC
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
                radius = norm * 1.2  # shrink search focus AC
    
    return best_vec, best_coeffs, best_norm

#lattice setup AC
n = 4  #4D lattice AC
B = np.random.randint(-5, 6, size=(n, n))

print("Basis matrix B:\n", B)

#basis quality AC
score = basis_condition_score(B)
print(f"Basis condition score (higher = worse): {score:.2f}")

#solve with naive solver AC
start_time = time.time()
shortest_naive, coeffs_naive, norm_naive = naive_svp_solver(B, max_range=2)
time_naive = time.time() - start_time

#solve with my algorithm AC
start_time = time.time()
shortest_unique, coeffs_unique, norm_unique = pruned_randomized_greedy_svp(B, max_trials=5000, initial_radius=10.0)
time_unique = time.time() - start_time

# --- Print Results ---

print("\n--- Naive Brute-Force Solver (limited search) ---")
print(f"Shortest vector: {shortest_naive}")
print(f"Coefficients: {coeffs_naive}")
print(f"Norm length: {norm_naive:.3f}")
print(f"Time taken: {time_naive:.3f} seconds")

print("\n--- Pruned Randomized Greedy Solver (Your Algorithm) ---")
print(f"Shortest vector: {shortest_unique}")
print(f"Coefficients: {coeffs_unique}")
print(f"Norm length: {norm_unique:.3f}")
print(f"Time taken: {time_unique:.3f} seconds")

#sisualization of lattice and shortest vectors AC
#generate lattice points for plotting AC
sample_vectors = []
max_range = 3  # larger plot range AC

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

# Lattice points
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
    name='Naive Shortest Vector'
))

#unique shortest vector
fig.add_trace(go.Scatter3d(
    x=[0, shortest_unique_proj[0, 0]],
    y=[0, shortest_unique_proj[0, 1]],
    z=[0, shortest_unique_proj[0, 2]],
    mode='lines+markers',
    marker=dict(size=5, color='green'),
    line=dict(width=5, color='green'),
    name='Your Shortest Vector (Pruned Randomized Search)'
))

fig.update_layout(title='SVP Solvers Comparison in Random 4D Lattice (PCA projected)',
                  scene=dict(xaxis_title='PCA 1',
                             yaxis_title='PCA 2',
                             zaxis_title='PCA 3'),
                  width=900, height=900)

fig.show()