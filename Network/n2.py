"""
Quantum Computing in Community Detection for Anti-Fraud Applications
===============================================================
Implementation of the QUBO-based community detection approach from:
  Wang et al., "Quantum Computing in Community Detection for Anti-Fraud Applications",
  Entropy 2024, 26, 1026

This script implements:
  1. Synthetic fraud transaction network generation
  2. QUBO formulation for community detection (Two-community & General multi-community)
  3. Simulated Annealing solver (classical simulation of quantum QUBO/CIM approach)
  4. Louvain algorithm comparison
  5. NetworkX visualization
"""

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import random
import time
from matplotlib.patches import Patch

# Set random seeds
np.random.seed(42)
random.seed(42)

# Chinese font
rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. Generate Synthetic Fraud Transaction Network
# ============================================================
def generate_fraud_network(n_communities=4, community_sizes=None,
                            p_in=0.15, p_out=0.008, seed=42):
    """Generate synthetic fraud transaction network with community structure."""
    if community_sizes is None:
        community_sizes = [61, 80, 77, 90]
    random.seed(seed)
    np.random.seed(seed)
    
    G = nx.Graph()
    true_partition = {}
    
    for c_idx, size in enumerate(community_sizes):
        for i in range(size):
            node_id = f"c{c_idx}_n{i}"
            G.add_node(node_id)
            true_partition[node_id] = c_idx
    
    for c_idx, size in enumerate(community_sizes):
        nodes_c = [f"c{c_idx}_n{i}" for i in range(size)]
        for i in range(len(nodes_c)):
            for j in range(i + 1, len(nodes_c)):
                if random.random() < p_in:
                    G.add_edge(nodes_c[i], nodes_c[j])
    
    for c1 in range(n_communities):
        for c2 in range(c1 + 1, n_communities):
            nodes1 = [f"c{c1}_n{i}" for i in range(community_sizes[c1])]
            nodes2 = [f"c{c2}_n{i}" for i in range(community_sizes[c2])]
            for n1 in nodes1:
                for n2 in nodes2:
                    if random.random() < p_out:
                        G.add_edge(n1, n2)
    
    return G, true_partition


# ============================================================
# 2. QUBO Solver - Two Community (Optimized SA)
# ============================================================
def qubo_sa_two_community(G, n_iterations=200000, temperature=50.0,
                           cooling_rate=0.99998, seed=42, warm_start=None):
    """
    QUBO two-community detection via Simulated Annealing.
    Implements Equation 6 from the paper.
    Uses O(n) incremental energy updates.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    nodes = list(G.nodes())
    n = len(nodes)
    node_idx = {node: i for i, node in enumerate(nodes)}
    m = G.number_of_edges()
    
    if m == 0 or n < 2:
        return [nodes], 0.0, [0.0]
    
    adj = np.zeros((n, n))
    degrees = np.zeros(n)
    for v, w in G.edges():
        iv, iw = node_idx[v], node_idx[w]
        adj[iv][iw] = 1
        adj[iw][iv] = 1
        degrees[iv] += 1
        degrees[iw] += 1
    
    C = adj.copy()
    for i in range(n):
        for j in range(n):
            C[i][j] -= degrees[i] * degrees[j] / (2 * m)
    
    # Initialize
    if warm_start is not None:
        x = np.array([warm_start.get(node, 0) for node in nodes])
    else:
        x = np.random.randint(0, 2, size=n)
    
    # Initial energy
    same_matrix = (x[:, None] == x[None, :]).astype(np.float64)
    current_energy = -np.sum(C * same_matrix) / (2 * m)
    
    best_x = x.copy()
    best_energy = current_energy
    energy_history = [current_energy]
    
    temp = temperature
    for iteration in range(n_iterations):
        k = np.random.randint(0, n)
        
        # O(n) incremental energy delta
        same_mask = (x == x[k])
        same_mask[k] = False
        delta_S = 2.0 * np.sum(C[k] * (1.0 - 2.0 * same_mask.astype(np.float64)))
        delta_e = -delta_S / (2 * m)
        
        if delta_e < 0 or random.random() < np.exp(-delta_e / temp):
            x[k] = 1 - x[k]
            current_energy += delta_e
        
        if current_energy < best_energy:
            best_x = x.copy()
            best_energy = current_energy
        
        temp *= cooling_rate
        if temp < 1e-15:
            break
        
        energy_history.append(current_energy)
    
    community1 = [nodes[i] for i in range(n) if best_x[i] == 0]
    community2 = [nodes[i] for i in range(n) if best_x[i] == 1]
    
    partition = [community1, community2]
    modularity = nx.community.modularity(G, partition)
    
    return partition, modularity, energy_history


# ============================================================
# 3. QUBO Solver - General Multi-Community (Optimized SA)
# ============================================================
def qubo_sa_general(G, n_communities=4, n_iterations=200000, temperature=10.0,
                     cooling_rate=0.99998, seed=42, warm_start=None):
    """
    QUBO general multi-community detection via Simulated Annealing.
    Implements Equation 5 from the paper (general QUBO with penalty).
    Uses O(n) incremental updates for efficiency.
    Simulates CIM ground state search for the corresponding Ising model.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    nodes = list(G.nodes())
    n = len(nodes)
    m = G.number_of_edges()
    
    if m == 0:
        return [nodes], 0.0, [0.0]
    
    # Precompute cost matrix
    node_idx = {node: i for i, node in enumerate(nodes)}
    adj = np.zeros((n, n))
    degrees = np.zeros(n)
    for v, w in G.edges():
        iv, iw = node_idx[v], node_idx[w]
        adj[iv][iw] = 1
        adj[iw][iv] = 1
        degrees[iv] += 1
        degrees[iw] += 1
    
    C = adj.copy()
    for i in range(n):
        for j in range(n):
            C[i][j] -= degrees[i] * degrees[j] / (2 * m)
    
    # Initialize
    if warm_start is not None:
        node_to_comm = warm_start
        assignment = np.array([node_to_comm.get(node, 0) % n_communities for node in nodes])
    else:
        assignment = np.random.randint(0, n_communities, size=n)
    
    # Precompute sum_C
    sum_C = np.zeros((n, n_communities))
    for c in range(n_communities):
        members = np.where(assignment == c)[0]
        for i in range(n):
            sum_C[i][c] = np.sum(C[i][members])
    
    # Compute initial modularity
    current_mod = 0.0
    for c in range(n_communities):
        members = np.where(assignment == c)[0]
        if len(members) > 1:
            current_mod += np.sum(C[np.ix_(members, members)])
    current_mod /= (2 * m)
    
    best_assignment = assignment.copy()
    best_mod = current_mod
    mod_history = [current_mod]
    
    temp = temperature
    for iteration in range(n_iterations):
        node_i = np.random.randint(0, n)
        old_c = assignment[node_i]
        new_c = np.random.randint(0, n_communities)
        
        if old_c == new_c:
            continue
        
        # Incremental delta
        delta_mod_2m = 2.0 * (sum_C[node_i][new_c] - sum_C[node_i][old_c]) + 2.0 * C[node_i][node_i]
        delta_mod = delta_mod_2m / (2 * m)
        
        # Metropolis criterion
        if delta_mod > 0 or random.random() < np.exp(delta_mod / temp):
            assignment[node_i] = new_c
            
            # Update sum_C
            for k in range(n):
                sum_C[k][old_c] -= C[k][node_i]
            for k in range(n):
                sum_C[k][new_c] += C[k][node_i]
            
            current_mod += delta_mod
        
        # Track best
        if current_mod > best_mod:
            best_assignment = assignment.copy()
            best_mod = current_mod
        
        temp *= cooling_rate
        if temp < 1e-15:
            break
        
        mod_history.append(current_mod)
    
    # Build partition
    partition = []
    for c in range(n_communities):
        members = [nodes[i] for i in range(n) if best_assignment[i] == c]
        if members:
            partition.append(members)
    
    assigned = set()
    for comm in partition:
        assigned.update(comm)
    for node in nodes:
        if node not in assigned:
            partition.append([node])
    
    modularity = nx.community.modularity(G, partition)
    
    return partition, modularity, mod_history


# ============================================================
# 4. Visualization Functions
# ============================================================
def visualize_community_graph(G, partition, title, filename, node_size=25, edge_alpha=0.2):
    """Visualize graph with community coloring."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    
    cmap = plt.cm.Set3
    num_comm = len(partition)
    colors_list = [cmap(i / num_comm) for i in range(num_comm)]
    
    node_colors = {}
    for c_idx, community in enumerate(partition):
        for node in community:
            node_colors[node] = colors_list[c_idx]
    
    pos = nx.spring_layout(G, seed=42, k=0.5, iterations=100, scale=2)
    
    nodes = list(G.nodes())
    node_color_list = [node_colors.get(node, 'white') for node in nodes]
    
    nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=node_color_list,
                          node_size=node_size, edgecolors='black', linewidths=0.3, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=edge_alpha, width=0.5, edge_color='gray', ax=ax)
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    legend_handles = []
    for c_idx, community in enumerate(partition):
        label = f'Community {c_idx} ({len(community)} nodes)'
        legend_handles.append(Patch(facecolor=colors_list[c_idx], 
                                   edgecolor='black', label=label))
    ax.legend(handles=legend_handles, loc='upper right', fontsize=9,
             framealpha=0.9, edgecolor='black', ncol=2)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def plot_modularity_comparison(methods, modularities, filename):
    """Bar chart of modularity comparison."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    colors = ['#4472C4', '#ED7D31', '#70AD47', '#FFC000']
    bars = ax.bar(methods, modularities, color=colors[:len(methods)], 
                   width=0.4, edgecolor='black', linewidth=1.2)
    ax.set_ylabel('Modularity (M)', fontsize=12, fontweight='bold')
    ax.set_title('Community Detection Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(modularities) * 1.15)
    for bar, val in zip(bars, modularities):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
               f'{val:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def plot_convergence_curves(qubo_hist, general_hist, filename):
    """Plot convergence curves."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    step = max(1, len(qubo_hist) // 800)
    ax.plot(range(0, len(qubo_hist), step), qubo_hist[::step], 
           color='#4472C4', linewidth=0.8, alpha=0.8, label='QUBO+SA (Two-Community)')
    
    if general_hist:
        step2 = max(1, len(general_hist) // 800)
        ax.plot(range(0, len(general_hist), step2), general_hist[::step2], 
               color='#ED7D31', linewidth=0.8, alpha=0.8, label='General QUBO+SA (4-Comm)')
    
    ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Modularity', fontsize=12, fontweight='bold')
    ax.set_title('Community Detection Convergence Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def plot_combined_comparison(times, modularities, methods, filename):
    """Combined comparison chart."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = ['#4472C4', '#ED7D31', '#70AD47', '#FFC000']
    
    bars1 = axes[0].bar(methods, modularities, color=colors[:len(methods)], 
                        width=0.4, edgecolor='black')
    axes[0].set_ylabel('Modularity', fontsize=11, fontweight='bold')
    axes[0].set_title('Modularity Comparison', fontsize=13, fontweight='bold')
    axes[0].set_ylim(0, max(modularities) * 1.15)
    for bar, val in zip(bars1, modularities):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    
    bars2 = axes[1].bar(methods, times, color=colors[:len(methods)], 
                        width=0.4, edgecolor='black')
    axes[1].set_ylabel('Time (seconds)', fontsize=11, fontweight='bold')
    axes[1].set_title('Execution Time Comparison', fontsize=13, fontweight='bold')
    for bar, val in zip(bars2, times):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{val:.2f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    
    plt.suptitle('Quantum QUBO vs Classical Algorithms - Performance Comparison', 
                fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Quantum Computing in Community Detection for Anti-Fraud")
    print("Implementation & Visualization using NetworkX")
    print("=" * 70)
    
    output_dir = "/home/gem/workhub/output"
    
    # Step 1: Generate graph
    print("\n[1/8] Generating synthetic fraud transaction network...")
    G, true_partition = generate_fraud_network(
        n_communities=4, community_sizes=[61, 80, 77, 90], p_in=0.15, p_out=0.008
    )
    print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    print(f"  True communities: {len(set(true_partition.values()))}")
    
    true_partitions = {}
    for node, comm in true_partition.items():
        true_partitions.setdefault(comm, []).append(node)
    true_partitions = list(true_partitions.values())
    true_mod = nx.community.modularity(G, true_partitions)
    print(f"  Ground truth modularity: {true_mod:.4f}")
    
    # Step 2: Run Louvain first (for warm start and comparison)
    print("\n[2/8] Running Louvain algorithm (for warm start & baseline)...")
    t0 = time.time()
    louvain_comms = nx.community.louvain_communities(G, seed=42)
    louvain_partition = [list(c) for c in louvain_comms]
    louvain_time = time.time() - t0
    louvain_modularity = nx.community.modularity(G, louvain_partition)
    print(f"  Time: {louvain_time:.2f}s | Modularity: {louvain_modularity:.4f}")
    
    # Create warm start from Louvain (map to 0,1 for two-community)
    louvain_warm_start = {}
    for c_idx, comm in enumerate(louvain_partition):
        for node in comm:
            louvain_warm_start[node] = c_idx % 2  # Map to binary for two-community
    
    # Step 3: QUBO + SA (two-community) with warm start
    print("\n[3/8] Running QUBO + SA (two-community, Eq.6) with Louvain warm start...")
    print("  (Simulating quantum QUBO/CIM approach)")
    t0 = time.time()
    qubo_partition, qubo_modularity, energy_history = qubo_sa_two_community(
        G, n_iterations=200000, temperature=50.0, cooling_rate=0.99998,
        warm_start=louvain_warm_start
    )
    qubo_time = time.time() - t0
    print(f"  Time: {qubo_time:.2f}s | Modularity: {qubo_modularity:.4f} | Communities: {len(qubo_partition)}")
    
    # Step 4: General QUBO + SA (multi-community) with warm start
    print("\n[4/8] Running General QUBO + SA (multi-community, Eq.5) with warm start...")
    print("  (Simulating CIM ground state search for Ising model)")
    # Create warm start for 4 communities
    louvain_warm_start_4 = {}
    for c_idx, comm in enumerate(louvain_partition):
        for node in comm:
            louvain_warm_start_4[node] = c_idx
    t0 = time.time()
    general_partition, general_modularity, general_history = qubo_sa_general(
        G, n_communities=4, n_iterations=200000, temperature=10.0, cooling_rate=0.99998,
        warm_start=louvain_warm_start_4
    )
    general_time = time.time() - t0
    print(f"  Time: {general_time:.2f}s | Modularity: {general_modularity:.4f} | Communities: {len(general_partition)}")
    
    # Step 5: Greedy Modularity
    print("\n[5/8] Running Greedy Modularity algorithm...")
    t0 = time.time()
    greedy_comms = nx.community.greedy_modularity_communities(G)
    greedy_partition = [list(c) for c in greedy_comms]
    greedy_time = time.time() - t0
    greedy_modularity = nx.community.modularity(G, greedy_partition)
    print(f"  Time: {greedy_time:.2f}s | Modularity: {greedy_modularity:.4f}")
    
    # Step 6: Visualizations
    print("\n[6/8] Generating visualizations...")
    
    visualize_community_graph(G, true_partitions,
        f"Ground Truth Community Structure\n({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)",
        f"{output_dir}/ground_truth_communities.png")
    
    visualize_community_graph(G, qubo_partition,
        f"QUBO + SA (Two-Community, Paper Eq.6)\nModularity = {qubo_modularity:.4f}  |  Time = {qubo_time:.2f}s",
        f"{output_dir}/qubo_sa_two_community.png")
    
    visualize_community_graph(G, general_partition,
        f"General QUBO + SA (Multi-Community, Paper Eq.5)\nModularity = {general_modularity:.4f}  |  Time = {general_time:.2f}s",
        f"{output_dir}/general_qubo_sa.png")
    
    visualize_community_graph(G, louvain_partition,
        f"Louvain Algorithm\nModularity = {louvain_modularity:.4f}  |  Time = {louvain_time:.2f}s",
        f"{output_dir}/louvain_communities.png")
    
    visualize_community_graph(G, greedy_partition,
        f"Greedy Modularity Algorithm\nModularity = {greedy_modularity:.4f}  |  Time = {greedy_time:.2f}s",
        f"{output_dir}/greedy_modularity.png")
    
    plot_convergence_curves(energy_history, general_history, f"{output_dir}/convergence_comparison.png")
    
    # Step 7: Comparison charts
    print("\n[7/8] Generating comparison charts...")
    plot_modularity_comparison(
        ['QUBO+SA\n(Two-Comm)', 'General QUBO\n+SA (4-Comm)', 'Louvain', 'Greedy\nModularity'],
        [qubo_modularity, general_modularity, louvain_modularity, greedy_modularity],
        f"{output_dir}/modularity_comparison.png"
    )
    plot_combined_comparison(
        [qubo_time, general_time, louvain_time, greedy_time],
        [qubo_modularity, general_modularity, louvain_modularity, greedy_modularity],
        ['QUBO+SA\n(2-Comm)', 'Gen.QUBO\n+SA (4-Comm)', 'Louvain', 'Greedy'],
        f"{output_dir}/algorithm_comparison.png"
    )
    
    # Step 8: Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"True communities: {len(true_partitions)}")
    print(f"Ground truth modularity: {true_mod:.4f}")
    print()
    print(f"{'Method':<30} {'Modularity':<12} {'Time(s)':<10} {'Communities':<10}")
    print("-" * 65)
    print(f"{'QUBO + SA (Two-Community)':<30} {qubo_modularity:<12.4f} {qubo_time:<10.2f} {len(qubo_partition):<10}")
    print(f"{'General QUBO + SA (4-Comm)':<30} {general_modularity:<12.4f} {general_time:<10.2f} {len(general_partition):<10}")
    print(f"{'Louvain':<30} {louvain_modularity:<12.4f} {louvain_time:<10.2f} {len(louvain_partition):<10}")
    print(f"{'Greedy Modularity':<30} {greedy_modularity:<12.4f} {greedy_time:<10.2f} {len(greedy_partition):<10}")
    print("=" * 70)
    print("\nNote: QUBO+SA uses simulated annealing as classical simulation of CIM.")
    print("Actual CIM hardware can explore solution space more efficiently.")
    print("\nAll visualizations saved successfully!")