import numpy as np
import networkx as nx
from collections import defaultdict
from networkx.algorithms.community import louvain_communities
import matplotlib.pyplot as plt

np.random.seed(42)
N = 308
n_fraud = 19
G = nx.random_partition_graph(sizes=[90, 80, 77, 61], p_in=0.22, p_out=0.004, seed=42)
node_list = list(G.nodes())

fraud_labels = np.zeros(N, dtype=int)
for idx in range(n_fraud):
    nid = node_list[idx]
    fraud_labels[nid] = 1

A = np.array(nx.adjacency_matrix(G).todense(), dtype=float)
m = G.number_of_edges()
degrees = np.array([d for _, d in G.degree()])

# ========== 论文公式(6): 二社区，无约束QUBO ==========
Q2 = np.zeros((N,N))
coeff = -1.0/(2*m)
for v in range(N):
    kv = degrees[v]
    for w in range(N):
        kw = degrees[w]
        Mvw = A[v,w] - (kv*kw)/(2.0*m)
        # x_v x_w + (1‑x_v)(1‑x_w) = 2 x_v x_w -x_v -x_w + 1
        Q2[v,w] += coeff * (2 * Mvw)
        if v == w:
            Q2[v,w] += coeff * (-Mvw)

def qubo2_energy(x):
    return float(x @ Q2 @ x)

def sa_qubo2(Qmat, T0=5000, Tmin=0.05, cooling=0.97, steps_per_temp=1500, seed=42):
    rng = np.random.default_rng(seed)
    n = Qmat.shape[0]
    x = rng.integers(0,2,size=n)
    cur_e = qubo2_energy(x)
    best_x = x.copy()
    best_e = cur_e
    T = T0
    while T > Tmin:
        for _ in range(steps_per_temp):
            i = rng.integers(0,n)
            xn = x.copy()
            xn[i] = 1 - xn[i]
            enew = qubo2_energy(xn)
            d = enew - cur_e
            if d < 0 or rng.random() < np.exp(-d / T):
                x = xn
                cur_e = enew
                if cur_e < best_e:
                    best_e = cur_e
                    best_x = x.copy()
        T *= cooling
    return best_x, best_e

best_x2, best_e2 = sa_qubo2(Q2)
# best_x2:0/1代表两个社区
node_comm_2 = {i:int(best_x2[i]) for i in range(N)}

# Louvain
louvain_comms = louvain_communities(G, seed=42)
louvain_node_comm = {}
for cid, comm in enumerate(louvain_comms):
    for n in comm:
        louvain_node_comm[n] = cid

def calc_modularity(graph, node2comm):
    comm_sets = defaultdict(set)
    for n,c in node2comm.items():
        comm_sets[c].add(n)
    comm_list = list(comm_sets.values())
    return nx.algorithms.community.quality.modularity(graph, comm_list)

def calc_fraud_stats(node2comm, fraud_label_arr):
    comm_stats = defaultdict(lambda: {"total":0, "fraud":0})
    N_nodes = len(fraud_label_arr)
    for n in range(N_nodes):
        c = node2comm[n]
        comm_stats[c]["total"] +=1
        if fraud_label_arr[n]==1:
            comm_stats[c]["fraud"] +=1
    total_fraud = np.sum(fraud_label_arr)
    res = {}
    max_fraud_rate = 0.0
    highrisk_recall = 0.0
    for c, st in comm_stats.items():
        fr = st["fraud"] / st["total"] if st["total"]>0 else 0.0
        rc = st["fraud"] / total_fraud if total_fraud>0 else 0.0
        res[c] = {"size":st["total"], "fraud_cnt":st["fraud"], "fraud_rate":fr, "recall":rc}
        if fr>max_fraud_rate:
            max_fraud_rate = fr
            highrisk_recall = rc
    return res, max_fraud_rate, highrisk_recall

sa_stats2, sa_max_fr2, sa_hr_rc2 = calc_fraud_stats(node_comm_2, fraud_labels)
mod_sa2 = calc_modularity(G, node_comm_2)
louv_stats, louv_max_fr, louv_hr_rc = calc_fraud_stats(louvain_node_comm, fraud_labels)
mod_louv = calc_modularity(G, louvain_node_comm)

print("\n===== 二社区QUBO‑SA(论文Eq.6) =====")
print(f"模块度 {mod_sa2:.4f}")
print(f"高危欺诈率 {sa_max_fr2:.3f},召回 {sa_hr_rc2:.3f}")
for c,s in sa_stats2.items():
    print(f"社区{c}: size={s['size']}, fraud={s['fraud_cnt']}, rate={s['fraud_rate']:.3f}")

print("\n===== Louvain =====")
print(f"模块度 {mod_louv:.4f},高危欺诈率 {louv_max_fr:.3f},召回 {louv_hr_rc:.3f}")

# 可视化
def draw_communities(graph, node2comm, title, save_name):
    plt.figure(figsize=(11,8))
    pos = nx.spring_layout(graph, seed=42)
    comm_sets = defaultdict(set)
    for n,c in node2comm.items():
        comm_sets[c].add(n)
    cmap = plt.cm.Set2
    for idx,nodes in enumerate(comm_sets.values()):
        nx.draw_networkx_nodes(graph, pos, nodelist=list(nodes), node_color=cmap(idx), alpha=0.7, node_size=70)
    fraud_nodes = [i for i in range(N) if fraud_labels[i]==1]
    nx.draw_networkx_nodes(graph, pos, nodelist=fraud_nodes, node_color="red", node_size=200)
    nx.draw_networkx_edges(graph, pos, alpha=0.12)
    plt.title(title)
    plt.axis("off")
    plt.savefig(save_name, dpi=140)

draw_communities(G, node_comm_2, "Two‑community QUBO‑SA", "qubo2.png")
draw_communities(G, louvain_node_comm, "Louvain baseline", "louvain.png")
plt.show()
