#imports AC
import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objs as go

#functions AC
#simple measure of basis conditioning AC
def basis_condition_score(B):
    B_T_B = B.T @ B
    eigvals = np.linalg.eigvals(B_T_B)
    return np.max(eigvals) / np.min(eigvals)

#naive brute-force svp solver AC
def naive_svp_solver(B, max_range=5):
    n = B.shape[1]
    shortest_vec = None
    min_norm = float('inf')
    
#search all integer coefficient vectors AC
    for coeffs in np.ndindex(*(2*max_range+1 for _ in range(n))):
        coeffs = np.array(coeffs) - max_range
        if np.all(coeffs == 0):
            continue  #skip the zero vector AC
        vec = B @ coeffs
        norm = np.linalg.norm(vec)
        if norm < min_norm:
            min_norm = norm
            shortest_vec = vec
            best_coeffs = coeffs
    return shortest_vec, best_coeffs, min_norm

#lattice setup AC
n = 3  #dimensions
B = np.random.randint(-5, 6, size=(n, n))  #random 3D lattice AC

print("Basis matrix B:\n", B)

#basis quality AC
score = basis_condition_score(B)
print(f"Basis condition score (higher = worse): {score:.2f}")

#solve SVP AC
shortest_vec, best_coeffs, shortest_norm = naive_svp_solver(B, max_range=5)

print("\n--- Shortest Vector Found ---")
print("Shortest vector:", shortest_vec)
print("Coefficients:", best_coeffs)
print(f"Vector length (norm): {shortest_norm:.3f}")

#visualization of lattice and SV AC
#generate lattice points for plotting AC
# Generate lattice points for plotting
sample_vectors = []
max_range = 5  # Consistent with the solver

for coeffs in np.ndindex(*((2*max_range+1,) * n)):
    coeffs = np.array(coeffs) - max_range  # Shift range from [0,10] to [-5,5]
    sample_vectors.append(B @ coeffs)

sample_vectors = np.array(sample_vectors)


#reduce dimension if needed AC
pca_3d = PCA(n_components=3)
data_3d = pca_3d.fit_transform(sample_vectors)

shortest_vec_proj = pca_3d.transform(shortest_vec.reshape(1, -1))

#plot lattice and SVP AC
fig = go.Figure()

#lattice points on plot AC
fig.add_trace(go.Scatter3d(
    x=data_3d[:, 0], y=data_3d[:, 1], z=data_3d[:, 2],
    mode='markers',
    marker=dict(size=3, color='blue'),
    name='Lattice Points'
))

#shortest vector on plot AC
fig.add_trace(go.Scatter3d(
    x=[0, shortest_vec_proj[0, 0]],
    y=[0, shortest_vec_proj[0, 1]],
    z=[0, shortest_vec_proj[0, 2]],
    mode='lines+markers',
    marker=dict(size=5, color='red'),
    line=dict(width=5, color='red'),
    name='Shortest Vector'
))

fig.update_layout(title='SVP Approximation in Random 3D Lattice',
                  scene=dict(xaxis_title='PCA 1',
                             yaxis_title='PCA 2',
                             zaxis_title='PCA 3'),
                  width=800, height=800)

fig.show()


