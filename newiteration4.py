#imports AC
import numpy as np
import time
from sklearn.decomposition import PCA
import plotly.graph_objs as go
from tabulate import tabulate 

#basis reduction AC
def lll_reduce(B):
    B = B.copy().astype(float)
    n = B.shape[1]
    for i in range(1, n):
        for j in range(i):
            mu = np.dot(B[:, i], B[:, j]) / np.dot(B[:, j], B[:, j])
            B[:, i] -= round(mu) * B[:, j]
    return B.astype(int)

#condition score AC
def basis_condition_score(B):
    B_T_B = B.T @ B
    eigvals = np.linalg.eigvals(B_T_B)
    return np.max(eigvals).real / np.min(eigvals).real

#naive brute-force solver AC
def naivebf_svp_solver(B, max_range=1):
    n = B.shape[1]
    shortest_vec = None
    min_norm = float('inf')
    best_coeffs = None
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

#PRG AC
def PRG_svp_localsearch(B, max_trials=10000, initial_radius=10.0):
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

#apply LLL-like reduction prep AC
B_reduced = lll_reduce(B)
print("\nReduced Basis Matrix B:\n", B_reduced)
reduced_score = basis_condition_score(B_reduced)
print(f"Condition Score (reduced): {reduced_score:.2f}")

#brute-force solver AC
start = time.time()
shortest_naivebf, coeffs_naivebf, norm_naivebf = naivebf_svp_solver(B_reduced, max_range=1)
time_naivebf = time.time() - start

#PRG AC
start = time.time()
shortest_prg, coeffs_prg, norm_prg = PRG_svp_localsearch(B_reduced)
time_prg = time.time() - start

#results Table AC
table = [
    ["Naive Brute-Force", str(coeffs_naivebf), f"{norm_naivebf:.3f}", f"{time_naivebf:.4f} s"],
    ["PRG + Local Search", str(coeffs_prg), f"{norm_prg:.3f}", f"{time_prg:.4f} s"]
]

print("\nComparison of Solvers (Iteration 4, LLL-Reduced Basis):")
print(tabulate(table, headers=["Method", "Coefficients", "Norm Length", "Runtime"]))


#PCA visualisation AC
#generate lattice points AC
sample_vectors = []
max_range = 3  # slightly larger range for clarity AC

for coeffs in np.ndindex(*((2*max_range+1,) * n)):
    coeffs = np.array(coeffs) - max_range
    sample_vectors.append(B_reduced @ coeffs)

sample_vectors = np.array(sample_vectors)

#reduce to 3D with PCA AC
pca_3d = PCA(n_components=3)
data_3d = pca_3d.fit_transform(sample_vectors)

naive_proj = pca_3d.transform(shortest_naivebf.reshape(1, -1))
prg_proj = pca_3d.transform(shortest_prg.reshape(1, -1))

#plot AC
fig = go.Figure()

#lattice points AC
fig.add_trace(go.Scatter3d(
    x=data_3d[:, 0], y=data_3d[:, 1], z=data_3d[:, 2],
    mode='markers', marker=dict(size=2, color='blue'),
    name='Lattice Points'
))

#naive brute-force shortest vector AC
fig.add_trace(go.Scatter3d(
    x=[0, naive_proj[0, 0]],
    y=[0, naive_proj[0, 1]],
    z=[0, naive_proj[0, 2]],
    mode='lines+markers',
    marker=dict(size=5, color='red'),
    line=dict(width=4, color='red'),
    name='Naive Brute-Force SV'
))

#PRG shortest vector AC
fig.add_trace(go.Scatter3d(
    x=[0, prg_proj[0, 0]],
    y=[0, prg_proj[0, 1]],
    z=[0, prg_proj[0, 2]],
    mode='lines+markers',
    marker=dict(size=5, color='green'),
    line=dict(width=4, color='green'),
    name='PRG + Local Search SV'
))

fig.update_layout(title='Iteration 4: SVP Solutions with LLL-Reduced Basis',
                  scene=dict(xaxis_title='PCA 1',
                             yaxis_title='PCA 2',
                             zaxis_title='PCA 3'),
                  width=1000, height=1000)

fig.show()




