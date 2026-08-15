import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
import os, warnings, gc
warnings.filterwarnings("ignore")
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.circuit.library import QAOAAnsatz
from qiskit_algorithms import NumPyMinimumEigensolver

np.random.seed(42)
output_dir = "quantum_output"
os.makedirs(output_dir, exist_ok=True)

G = nx.gnp_random_graph(6, 0.5, seed=42)
pos = nx.spring_layout(G, seed=42)
num_colors = 4
color_map = ['#e41a1c', '#377eb8', '#4daf4a', '#ff7f00']
color_names = ['Red', 'Blue', 'Green', 'Orange']
n = G.number_of_nodes()
penalty = 10

qp = QuadraticProgram(name='GC4')
for i in range(n):
    for c in range(num_colors):
        qp.binary_var(f'x_{i}_{c}')
linear = {}
quadratic = {}
for i in range(n):
    for c in range(num_colors):
        linear[f'x_{i}_{c}'] = linear.get(f'x_{i}_{c}', 0) - penalty
    for c1 in range(num_colors):
        for c2 in range(c1+1, num_colors):
            quadratic[(f'x_{i}_{c1}', f'x_{i}_{c2}')] = quadratic.get((f'x_{i}_{c1}', f'x_{i}_{c2}'), 0) + 2*penalty
for u, v in G.edges():
    for c in range(num_colors):
        quadratic[(f'x_{u}_{c}', f'x_{v}_{c}')] = quadratic.get((f'x_{u}_{c}', f'x_{v}_{c}'), 0) + penalty
qp.minimize(constant=n*penalty, linear=linear, quadratic=quadratic)
qubo = QuadraticProgramToQubo().convert(qp)

result = MinimumEigenOptimizer(NumPyMinimumEigensolver()).solve(qubo)
x_opt = np.array(result.x, dtype=float)
colors = []
for i in range(n):
    vals = [x_opt[qp.variables_index[f'x_{i}_{c}']] for c in range(num_colors)]
    colors.append(int(np.argmax(vals)))
conf = sum(1 for u, v in G.edges() if colors[u] == colors[v])
print(f"Colors: {colors}, Conflicts: {conf}, fval: {result.fval}, Valid: {conf==0}")

fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
for u, v in G.edges():
    ec = '#e74c3c' if colors[u]==colors[v] else '#7f8c8d'
    ew = 4.0 if colors[u]==colors[v] else 2.5
    es = 'dashed' if colors[u]==colors[v] else 'solid'
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[(u,v)], width=ew, edge_color=ec, style=es, alpha=0.9)
nx.draw_networkx_nodes(G, pos, ax=ax, node_size=1800, node_color='#ecf0f1', edgecolors='none', alpha=0.4)
nx.draw_networkx_nodes(G, pos, ax=ax, node_size=1400, node_color=[color_map[c] for c in colors], edgecolors='#2c3e50', linewidths=2.5)
lbl = {i: f"{i}\nC{colors[i]+1}" for i in G.nodes()}
txt = nx.draw_networkx_labels(G, pos, labels=lbl, ax=ax, font_size=13, font_weight='bold', font_color='white')
for _, t in txt.items():
    t.set_path_effects([pe.withStroke(linewidth=2.5, foreground='black')])
for u, v in G.edges():
    mx, my = (pos[u][0]+pos[v][0])/2, (pos[u][1]+pos[v][1])/2
    ax.text(mx, my, '1', fontsize=8, ha='center', va='center', bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='gray', alpha=0.85))
handles = [Patch(facecolor=color_map[c], edgecolor='black', label=f'C{c+1} ({color_names[c]}): {sorted([i for i,cc in enumerate(colors) if cc==c])}') for c in range(num_colors) if any(cc==c for cc in colors)]
ax.legend(handles=handles, loc='upper left', frameon=True, fancybox=True, framealpha=0.95, fontsize=10)
ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'graph_colored.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
plt.close()
gc.collect()

adj = nx.to_numpy_array(G, dtype=int)
mat = np.zeros((n, n))
annot_arr = np.empty((n, n), dtype=object)
for i in range(n):
    for j in range(n):
        if i == j:
            mat[i,j] = colors[i]+2
            annot_arr[i,j] = f'C{colors[i]+1}'
        elif adj[i,j] == 1:
            mat[i,j] = -1 if colors[i]==colors[j] else 1
            annot_arr[i,j] = '✗' if colors[i]==colors[j] else '✓'
        else:
            mat[i,j] = 0
            annot_arr[i,j] = '·'
cmap2 = LinearSegmentedColormap.from_list('m2', ['#ff6b6b','#ffeaa7','#dfe6e9','#74b9ff','#55efc4','#a29bfe'])
fig, ax = plt.subplots(figsize=(8, 7), facecolor='white')
sns.heatmap(mat, ax=ax, cmap=cmap2, square=True, cbar=True, linewidths=2.5, linecolor='white', annot=annot_arr, fmt='', annot_kws={'fontsize':14, 'fontweight':'bold', 'color':'#2d3436'}, xticklabels=[f'N{i}' for i in range(n)], yticklabels=[f'N{i}' for i in range(n)], cbar_kws={'shrink':0.8})
ax.tick_params(axis='both', labelsize=11, rotation=0)
for spine in ax.spines.values():
    spine.set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'adjacency_matrix.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
plt.close()
gc.collect()

q_matrix = qubo.objective.quadratic.to_array()
linear_vec = qubo.objective.linear.to_array()
n_vars = q_matrix.shape[0]
xl, yl, zl = [], [], []
for i in range(n_vars):
    if abs(linear_vec[i]) > 1e-9:
        xl.append(i); yl.append(i); zl.append(linear_vec[i])
for i in range(n_vars):
    for j in range(n_vars):
        if abs(q_matrix[i,j]) > 1e-9:
            xl.append(i); yl.append(j); zl.append(q_matrix[i,j])
xa, ya, za = np.array(xl), np.array(yl), np.array(zl)
vlabels = [f'x{i//num_colors}_{i%num_colors}' for i in range(n_vars)]
fig = plt.figure(figsize=(14, 10), facecolor='white')
ax3 = fig.add_subplot(111, projection='3d')
ax3.set_facecolor('white')
for x, y, z in zip(xa, ya, za):
    ax3.plot([x,x],[y,y],[0,z], color='#ff6b6b' if z<0 else '#74b9ff', alpha=0.35, linewidth=0.9)
sc = ax3.scatter(xa, ya, za, c=za, cmap='turbo', s=110, alpha=0.95, edgecolors='black', linewidth=0.5)
step = max(1, n_vars//8)
ticks = list(range(0, n_vars, step))
ax3.set_xticks(ticks)
ax3.set_yticks(ticks)
ax3.set_xticklabels([vlabels[i] for i in ticks], rotation=20, ha='right', fontsize=9)
ax3.set_yticklabels([vlabels[i] for i in ticks], fontsize=9)
ax3.set_xlabel('var i', fontsize=11, labelpad=8)
ax3.set_ylabel('var j', fontsize=11, labelpad=8)
ax3.set_zlabel('coef', fontsize=11, labelpad=8)
ax3.xaxis.pane.set_alpha(0)
ax3.yaxis.pane.set_alpha(0)
ax3.zaxis.pane.set_alpha(0)
ax3.grid(True, alpha=0.2)
ax3.view_init(elev=28, azim=42)
fig.colorbar(sc, ax=ax3, shrink=0.7, pad=0.08).ax.tick_params(labelsize=9)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'qubo_3d.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
plt.close()
gc.collect()

full_q = q_matrix.copy()
for i in range(n_vars):
    full_q[i,i] += linear_vec[i]
ms = min(24, n_vars)
fig, ax = plt.subplots(figsize=(13, 11), facecolor='white')
sns.heatmap(full_q[:ms,:ms], ax=ax, cmap='Spectral_r', center=0, annot=True, fmt='.0f', annot_kws={'fontsize':7, 'fontweight':'bold'}, square=True, linewidths=0.6, linecolor='white', cbar=True, xticklabels=vlabels[:ms], yticklabels=vlabels[:ms], cbar_kws={'shrink':0.85})
ax.tick_params(axis='x', rotation=90, labelsize=7)
ax.tick_params(axis='y', rotation=0, labelsize=7)
for spine in ax.spines.values():
    spine.set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'qubo_heatmap.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
plt.close()
gc.collect()

try:
    cost_op, _ = qubo.to_ising()
    qc = QAOAAnsatz(cost_operator=cost_op, reps=1).decompose()
    nq = qc.num_qubits
    fig, ax = plt.subplots(figsize=(max(24, nq*2), max(10, nq*1.1)), facecolor='white')
    qc.draw('mpl', ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'quantum_circuit.png'), dpi=200, bbox_inches='tight', facecolor='white')
    plt.show()
    plt.close()
except:
    cost_op, _ = qubo.to_ising()
    nq = cost_op.num_qubits
    fig, ax = plt.subplots(figsize=(18, max(7, nq*0.65)), facecolor='white')
    for q in range(nq):
        ax.plot([0, 6], [q, q], color='#34495e', linewidth=2, alpha=0.85)
        ax.text(-0.2, q, f'q{q}', va='center', ha='right', fontsize=8, fontweight='bold')
    for q in range(nq):
        r = plt.Rectangle((0.35, q-0.22), 0.48, 0.44, facecolor='#00b4d8', edgecolor='black', linewidth=1.5, zorder=3)
        ax.add_patch(r)
        ax.text(0.59, q, 'H', va='center', ha='center', fontsize=9, fontweight='bold', color='white', zorder=4)
    ax.add_patch(plt.Rectangle((1.2, -0.35), 1.5, nq-0.3, facecolor='#ff9f1c', edgecolor='black', linewidth=1.8, alpha=0.9, zorder=2))
    ax.text(1.95, nq/2-0.5, 'Uc(γ)', ha='center', va='center', fontsize=13, fontweight='bold', zorder=4)
    ax.add_patch(plt.Rectangle((3.1, -0.35), 1.5, nq-0.3, facecolor='#06d6a0', edgecolor='black', linewidth=1.8, alpha=0.9, zorder=2))
    ax.text(3.85, nq/2-0.5, 'Um(β)', ha='center', va='center', fontsize=13, fontweight='bold', zorder=4)
    for q in range(nq):
        ax.add_patch(plt.Circle((5.1, q), 0.2, facecolor='#9d4edd', edgecolor='black', linewidth=1.3, zorder=3))
        ax.text(5.1, q, 'M', va='center', ha='center', fontsize=8, fontweight='bold', color='white', zorder=4)
    ax.legend(handles=[Patch(facecolor='#00b4d8', edgecolor='black', label='Hadamard'), Patch(facecolor='#ff9f1c', edgecolor='black', label='Cost Uc(γ)'), Patch(facecolor='#06d6a0', edgecolor='black', label='Mixer Um(β)'), Patch(facecolor='#9d4edd', edgecolor='black', label='Measure')], loc='lower right', fontsize=10, framealpha=0.95)
    ax.set_xlim(-0.5, 5.8)
    ax.set_ylim(-0.8, nq-0.2)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'quantum_circuit.png'), dpi=200, bbox_inches='tight', facecolor='white')
    plt.show()
    plt.close()
gc.collect()
