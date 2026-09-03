"""
Graph Serialization Faithfulness Experiments
=============================================
GroundLM 2026 Workshop Paper

Diagnoses whether LLMs faithfully interpret graph evidence
under graph-preserving transformations of serialized input.

Four diagnostics:
  1. Format Stability (FS)  — same graph, different format
  2. Order Sensitivity (OS)  — same graph+format, different ordering
  3. Isomorphism Consistency (IC) — same structure, relabeled nodes
  4. Distractor Robustness (DR) — same graph + irrelevant structure

NO adaptive acquisition. NO AGEA. NO retrieval. NO fraud datasets.
"""

import os
import json
import random
import itertools
import warnings
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import networkx as nx

warnings.filterwarnings("ignore")

# ─── Configuration ───

SEED = 42
OUTPUT_DIR = "./groundlm_output"
GRAPH_SIZES = [5, 8, 10, 12, 15]
GRAPHS_PER_SIZE = 20  # 5 sizes × 20 = 100 graphs
EDGE_PROB = 0.3

# Graph types for expanded experiments
GRAPH_TYPES = {
    "erdos_renyi": {"sizes": [5, 8, 10, 12, 15], "per_size": 20, "edge_prob": 0.3},
    "barabasi_albert": {"sizes": [5, 8, 10, 12, 15], "per_size": 20, "m_edges": 2},
    "watts_strogatz": {"sizes": [5, 8, 10, 12, 15], "per_size": 20, "k_neighbors": 4, "rewire_prob": 0.3},
    # Real-world graphs (loaded from NetworkX built-ins or small datasets)
    "real_world": {"graphs": ["karate_club", "davis_southern_women", "florentine_families"]},
}

# Stress test: larger graphs (n=20-30)
STRESS_GRAPH_TYPES = {
    "erdos_renyi_large": {"sizes": [20, 25, 30], "per_size": 17, "edge_prob": 0.2},
}

MODELS = [
    "gpt-5.4-mini",
    "gpt-5.4",
    "Qwen/Qwen3-8B",
]

FORMATS = ["edge_list", "adjacency_list", "json", "natural_language"]
ORDERINGS = ["canonical", "random", "bfs"]

TASKS = [
    "connectivity",
    "shortest_path",
    "triangle_detection",
    "common_neighbor",
]

DISTRACTOR_TYPES = [
    "disconnected_component",
    "disconnected_high_degree_star",
    "disconnected_triangle",
]

RELABEL_TYPES = ["neutral", "misleading_semantic"]

NUM_RANDOM_SEEDS = 3  # for stochastic runs


# ─── Graph Generation ───

def generate_graphs(seed: int = SEED) -> List[Dict]:
    """Generate synthetic Erdős–Rényi graphs with ground-truth annotations."""
    random.seed(seed)
    np.random.seed(seed)
    all_graphs = []

    for n in GRAPH_SIZES:
        for gi in range(GRAPHS_PER_SIZE):
            gseed = seed + n * 1000 + gi
            random.seed(gseed)
            np.random.seed(gseed)

            # Generate connected ER graph (retry if disconnected)
            for attempt in range(20):
                p = EDGE_PROB
                G = nx.erdos_renyi_graph(n, p, seed=gseed + attempt)
                if nx.is_connected(G):
                    break
            if not nx.is_connected(G):
                # Force connectivity by adding edges
                components = list(nx.connected_components(G))
                for i in range(len(components) - 1):
                    u = list(components[i])[0]
                    v = list(components[i + 1])[0]
                    G.add_edge(u, v)

            # Assign alphabetical labels
            labels = {i: chr(65 + i) if i < 26 else f"N{i}" for i in range(n)}
            G = nx.relabel_nodes(G, labels)

            graph_info = {
                "graph_id": f"er_n{n}_g{gi}",
                "graph_type": "erdos_renyi",
                "n": n,
                "m": G.number_of_edges(),
                "nodes": sorted(G.nodes()),
                "edges": sorted(G.edges()),
                "graph": G,
                "seed": gseed,
            }
            all_graphs.append(graph_info)

    return all_graphs


def generate_graphs_expanded(seed: int = SEED) -> List[Dict]:
    """Generate graphs from multiple families: ER, Barabási–Albert, Watts–Strogatz, and real-world."""
    random.seed(seed)
    np.random.seed(seed)
    all_graphs = []

    # ─── Erdős–Rényi ───
    cfg = GRAPH_TYPES["erdos_renyi"]
    for n in cfg["sizes"]:
        for gi in range(cfg["per_size"]):
            gseed = seed + n * 1000 + gi
            G = _make_connected(nx.erdos_renyi_graph(n, cfg["edge_prob"], seed=gseed))
            G = _label_graph(G)
            all_graphs.append(_graph_info(G, f"er_n{n}_g{gi}", "erdos_renyi", gseed))

    # ─── Barabási–Albert (scale-free) ───
    cfg = GRAPH_TYPES["barabasi_albert"]
    for n in cfg["sizes"]:
        for gi in range(cfg["per_size"]):
            gseed = seed + 100000 + n * 1000 + gi
            m = min(cfg["m_edges"], n - 1)
            G = nx.barabasi_albert_graph(n, m, seed=gseed)
            G = _label_graph(G)
            all_graphs.append(_graph_info(G, f"ba_n{n}_g{gi}", "barabasi_albert", gseed))

    # ─── Watts–Strogatz (small-world) ───
    cfg = GRAPH_TYPES["watts_strogatz"]
    for n in cfg["sizes"]:
        for gi in range(cfg["per_size"]):
            gseed = seed + 200000 + n * 1000 + gi
            k = min(cfg["k_neighbors"], n - 1)
            if k % 2 != 0:
                k -= 1  # must be even
            if k < 2:
                k = 2
            G = nx.watts_strogatz_graph(n, k, cfg["rewire_prob"], seed=gseed)
            G = _make_connected(G)
            G = _label_graph(G)
            all_graphs.append(_graph_info(G, f"ws_n{n}_g{gi}", "watts_strogatz", gseed))

    # ─── Real-world graphs ───
    rw_graphs = _load_real_world_graphs()
    for name, G in rw_graphs:
        G = _label_graph(G)
        all_graphs.append(_graph_info(G, f"rw_{name}", "real_world", seed))

    return all_graphs


def _make_connected(G: nx.Graph) -> nx.Graph:
    """Force a graph to be connected by adding edges between components."""
    if nx.is_connected(G):
        return G
    components = list(nx.connected_components(G))
    for i in range(len(components) - 1):
        u = list(components[i])[0]
        v = list(components[i + 1])[0]
        G.add_edge(u, v)
    return G


def _label_graph(G: nx.Graph) -> nx.Graph:
    """Assign alphabetical labels to graph nodes."""
    nodes = sorted(G.nodes())
    labels = {}
    for i, node in enumerate(nodes):
        labels[node] = chr(65 + i) if i < 26 else f"N{i}"
    return nx.relabel_nodes(G, labels)


def _graph_info(G: nx.Graph, graph_id: str, graph_type: str, gseed: int) -> Dict:
    """Build a graph info dict."""
    return {
        "graph_id": graph_id,
        "graph_type": graph_type,
        "n": G.number_of_nodes(),
        "m": G.number_of_edges(),
        "nodes": sorted(G.nodes()),
        "edges": sorted(G.edges()),
        "graph": G,
        "seed": gseed,
    }


def _load_real_world_graphs() -> List[tuple]:
    """Load small real-world graphs from NetworkX built-in datasets."""
    graphs = []

    # Zachary's Karate Club (34 nodes, 78 edges)
    G_karate = nx.karate_club_graph()
    graphs.append(("karate_club", G_karate))

    # Davis Southern Women (18 nodes, 89 edges bipartite → projected)
    G_davis = nx.davis_southern_women_graph()
    # Project bipartite graph onto women nodes
    women = [n for n, d in G_davis.nodes(data=True) if d.get("bipartite") == 0]
    G_proj = nx.bipartite.projected_graph(G_davis, women)
    if G_proj.number_of_nodes() > 0:
        graphs.append(("davis_women", G_proj))

    # Florentine Families (15 nodes, 20 edges)
    G_florentine = nx.florentine_families_graph()
    graphs.append(("florentine_families", G_florentine))

    # Les Misérables character co-occurrence (77 nodes, 254 edges)
    try:
        G_lesmis = nx.les_miserables_graph()
        # Take a random connected subgraph of ~20 nodes for consistency
        random.seed(42)
        start = random.choice(list(G_lesmis.nodes()))
        bfs_nodes = list(nx.bfs_tree(G_lesmis, start, depth_limit=2).nodes())[:20]
        G_sub = G_lesmis.subgraph(bfs_nodes).copy()
        if nx.is_connected(G_sub) and G_sub.number_of_nodes() >= 8:
            graphs.append(("lesmis_subgraph", G_sub))
    except Exception:
        pass

    return graphs


# ─── Task Ground Truth ───

def compute_ground_truth(G: nx.Graph, task: str, seed: int = SEED) -> List[Dict]:
    """Compute ground-truth queries for a given task on graph G."""
    random.seed(seed)
    queries = []
    nodes = sorted(G.nodes())

    if task == "connectivity":
        # Sample 3 node pairs
        pairs = []
        for _ in range(3):
            a, b = random.sample(nodes, 2)
            pairs.append((a, b))
        for a, b in pairs:
            has_path = nx.has_path(G, a, b)
            queries.append({
                "query": f"Is there a path from node {a} to node {b}?",
                "answer": "yes" if has_path else "no",
                "answer_bool": has_path,
                "source_nodes": [a, b],
            })

    elif task == "shortest_path":
        pairs = []
        for _ in range(3):
            a, b = random.sample(nodes, 2)
            if nx.has_path(G, a, b):
                pairs.append((a, b))
        for a, b in pairs[:3]:
            path = nx.shortest_path(G, a, b)
            dist = len(path) - 1
            queries.append({
                "query": f"What is the length of the shortest path from node {a} to node {b}?",
                "answer": str(dist),
                "answer_int": dist,
                "source_nodes": [a, b],
            })

    elif task == "triangle_detection":
        # Sample 3 nodes, check if they participate in a triangle
        sample = random.sample(nodes, min(3, len(nodes)))
        for node in sample:
            neighbors = list(G.neighbors(node))
            has_triangle = False
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    if G.has_edge(neighbors[i], neighbors[j]):
                        has_triangle = True
                        break
                if has_triangle:
                    break
            queries.append({
                "query": f"Does node {node} participate in a triangle (a cycle of length 3)?",
                "answer": "yes" if has_triangle else "no",
                "answer_bool": has_triangle,
                "source_nodes": [node],
            })

    elif task == "common_neighbor":
        pairs = []
        for _ in range(3):
            a, b = random.sample(nodes, 2)
            pairs.append((a, b))
        for a, b in pairs:
            common = set(G.neighbors(a)) & set(G.neighbors(b))
            queries.append({
                "query": f"Do nodes {a} and {b} share a common neighbor?",
                "answer": "yes" if common else "no",
                "answer_bool": bool(common),
                "source_nodes": [a, b],
            })

    elif task == "bridge_edge":
        # Sample 3 edges, check if they are bridges
        edges = list(G.edges())
        if len(edges) >= 3:
            sample_edges = random.sample(edges, 3)
        else:
            sample_edges = edges
        for u, v in sample_edges:
            is_bridge = nx.has_path(G, u, v) and (u, v) in nx.bridges(G)
            # nx.bridges returns bridge edges
            queries.append({
                "query": f"Is the edge between {u} and {v} a bridge (removing it disconnects the graph)?",
                "answer": "yes" if is_bridge else "no",
                "answer_bool": is_bridge,
                "source_nodes": [u, v],
            })

    elif task == "cycle_detection":
        has_cycle = len(G.edges()) >= len(G.nodes())
        if has_cycle:
            has_cycle = bool(nx.cycle_basis(G))
        queries.append({
            "query": "Does this graph contain a cycle?",
            "answer": "yes" if has_cycle else "no",
            "answer_bool": has_cycle,
            "source_nodes": [],
        })

    return queries


# ─── Exact-query Verification ───

def compute_answer_for_source_nodes(G: nx.Graph, task: str, source_nodes: List[str]) -> str:
    """Compute the symbolic answer for one already-selected query.

    This verifies a perturbation without resampling query nodes. Adding
    distractor nodes changes the sampling population, so rerunning
    ``compute_ground_truth`` cannot reliably recover the original query even
    with the same random seed.
    """
    if task == "connectivity":
        a, b = source_nodes
        return "yes" if nx.has_path(G, a, b) else "no"
    if task == "shortest_path":
        a, b = source_nodes
        return str(nx.shortest_path_length(G, a, b))
    if task == "triangle_detection":
        (node,) = source_nodes
        neighbors = list(G.neighbors(node))
        has_triangle = any(
            G.has_edge(neighbors[i], neighbors[j])
            for i in range(len(neighbors))
            for j in range(i + 1, len(neighbors))
        )
        return "yes" if has_triangle else "no"
    if task == "common_neighbor":
        a, b = source_nodes
        common = set(G.neighbors(a)) & set(G.neighbors(b))
        return "yes" if common else "no"
    raise ValueError(f"Unsupported task for exact-query verification: {task}")


# ─── Serialization Formats ───

def serialize_edge_list(G: nx.Graph, node_order: List[str], edge_order: List[Tuple]) -> str:
    """Edge list format."""
    lines = ["Edges:"]
    for u, v in edge_order:
        lines.append(f"{u} -- {v}")
    return "\n".join(lines)


def serialize_adjacency_list(G: nx.Graph, node_order: List[str]) -> str:
    """Adjacency list format."""
    lines = []
    for node in node_order:
        neighbors = sorted(G.neighbors(node))
        lines.append(f"{node}: {', '.join(neighbors)}")
    return "\n".join(lines)


def serialize_json(G: nx.Graph, node_order: List[str], edge_order: List[Tuple]) -> str:
    """JSON-like format."""
    data = {
        "nodes": node_order,
        "edges": [[u, v] for u, v in edge_order],
    }
    return json.dumps(data, indent=2)


def serialize_natural_language(G: nx.Graph, node_order: List[str]) -> str:
    """Natural language description."""
    lines = []
    for node in node_order:
        neighbors = sorted(G.neighbors(node))
        if len(neighbors) == 0:
            lines.append(f"Node {node} has no connections.")
        elif len(neighbors) == 1:
            lines.append(f"Node {node} is connected to node {neighbors[0]}.")
        elif len(neighbors) == 2:
            lines.append(f"Node {node} is connected to nodes {neighbors[0]} and {neighbors[1]}.")
        else:
            nl = ", ".join(neighbors[:-1]) + f", and {neighbors[-1]}"
            lines.append(f"Node {node} is connected to nodes {nl}.")
    return " ".join(lines)


SERIALIZERS = {
    "edge_list": serialize_edge_list,
    "adjacency_list": serialize_adjacency_list,
    "json": serialize_json,
    "natural_language": serialize_natural_language,
}


# ─── Node/Edge Orderings ───

def get_node_order(G: nx.Graph, ordering: str, source_nodes: List[str] = None, seed: int = 42) -> List[str]:
    """Return nodes in specified order."""
    nodes = list(G.nodes())
    if ordering == "canonical":
        return sorted(nodes)
    elif ordering == "random":
        rng = random.Random(seed)
        rng.shuffle(nodes)
        return nodes
    elif ordering == "bfs":
        start = source_nodes[0] if source_nodes else nodes[0]
        visited = []
        queue = [start]
        seen = {start}
        while queue:
            n = queue.pop(0)
            visited.append(n)
            for nb in sorted(G.neighbors(n)):
                if nb not in seen:
                    seen.add(nb)
                    queue.append(nb)
        # Add any disconnected nodes
        for n in sorted(nodes):
            if n not in seen:
                visited.append(n)
        return visited
    elif ordering == "degree_desc":
        return sorted(nodes, key=lambda n: G.degree(n), reverse=True)
    elif ordering == "target_centered":
        if source_nodes:
            # Source nodes first, then their neighbors, then rest
            ordered = list(source_nodes)
            for s in source_nodes:
                for nb in sorted(G.neighbors(s)):
                    if nb not in ordered:
                        ordered.append(nb)
            for n in sorted(nodes):
                if n not in ordered:
                    ordered.append(n)
            return ordered
        return sorted(nodes)
    return sorted(nodes)


def get_edge_order(G: nx.Graph, node_order: List[str], ordering: str, seed: int = 42) -> List[Tuple]:
    """Return edges in specified order."""
    edges = list(G.edges())
    if ordering == "canonical":
        return sorted(edges)
    elif ordering == "random":
        rng = random.Random(seed)
        return rng.sample(edges, len(edges))
    elif ordering in ("bfs", "degree_desc", "target_centered"):
        # Order edges by their appearance in node_order
        node_idx = {n: i for i, n in enumerate(node_order)}
        return sorted(edges, key=lambda e: (min(node_idx[e[0]], node_idx[e[1]]), max(node_idx[e[0]], node_idx[e[1]])))
    return sorted(edges)


# ─── Graph Transformations ───

def relabel_neutral(G: nx.Graph) -> Tuple[nx.Graph, Dict]:
    """Relabel nodes with neutral names: A→X1, B→X2, etc."""
    mapping = {}
    for i, node in enumerate(sorted(G.nodes())):
        mapping[node] = f"X{i+1}"
    G_new = nx.relabel_nodes(G, mapping)
    return G_new, mapping


def relabel_misleading(G: nx.Graph) -> Tuple[nx.Graph, Dict]:
    """Apply the recovered degree-congruent semantic-name mapping.

    The historical function name is retained for file compatibility. This is
    not a verified implementation of the paper's role-incongruent intervention.
    """
    # Sort by degree descending
    nodes_by_deg = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)
    semantic_names = ["Hub", "Center", "Bridge", "Anchor", "Core",
                      "Leaf", "Edge", "Tip", "Outlier", "Tail",
                      "Node11", "Node12", "Node13", "Node14", "Node15"]
    mapping = {}
    for i, node in enumerate(nodes_by_deg):
        name = semantic_names[i] if i < len(semantic_names) else f"Node{i+1}"
        mapping[node] = name
    G_new = nx.relabel_nodes(G, mapping)
    return G_new, mapping


def add_disconnected_component(G: nx.Graph, seed: int = 42) -> nx.Graph:
    """Add a disconnected 3-node triangle component."""
    G_new = G.copy()
    rng = random.Random(seed)
    # Find unused node names
    existing = set(G.nodes())
    new_names = []
    i = 1
    while len(new_names) < 3:
        name = f"D{i}"
        if name not in existing:
            new_names.append(name)
        i += 1
    G_new.add_edge(new_names[0], new_names[1])
    G_new.add_edge(new_names[1], new_names[2])
    G_new.add_edge(new_names[0], new_names[2])
    return G_new


def add_disconnected_high_degree_star(G: nx.Graph, seed: int = 42) -> nx.Graph:
    """Add a disconnected high-degree star (1 hub + 5 leaves) as a separate component.

    Label-preserving: the star is fully disconnected from the original graph,
    so it cannot affect any ground-truth answer for connectivity, shortest path,
    triangle detection, or common neighbor queries on the original graph.
    """
    G_new = G.copy()
    existing = set(G.nodes())
    new_names = []
    i = 1
    # Need 6 nodes: 1 hub + 5 leaves
    while len(new_names) < 6:
        name = f"D{i}"
        if name not in existing:
            new_names.append(name)
        i += 1
    hub = new_names[0]
    for leaf in new_names[1:]:
        G_new.add_edge(hub, leaf)
    return G_new


def add_disconnected_triangle(G: nx.Graph, seed: int = 42) -> nx.Graph:
    """Add a disconnected 3-node triangle as a separate component.

    Label-preserving: the triangle is fully disconnected from the original graph,
    so it cannot affect any ground-truth answer.
    """
    G_new = G.copy()
    existing = set(G.nodes())
    new_names = []
    i = 1
    while len(new_names) < 3:
        name = f"D{i}"
        if name not in existing:
            new_names.append(name)
        i += 1
    G_new.add_edge(new_names[0], new_names[1])
    G_new.add_edge(new_names[1], new_names[2])
    G_new.add_edge(new_names[0], new_names[2])
    return G_new


DISTRACTOR_FUNCTIONS = {
    "disconnected_component": add_disconnected_component,
    "disconnected_high_degree_star": add_disconnected_high_degree_star,
    "disconnected_triangle": add_disconnected_triangle,
}


# ─── Prompt Construction ───

def build_prompt(graph_text: str, query: str, task: str) -> str:
    """Build the full prompt for the LLM."""
    return f"""Given the following graph:

{graph_text}

Question: {query}

Think step by step about the graph structure, then give your final answer.
Final answer:"""


def remap_query(query: str, mapping: Dict[str, str]) -> str:
    """Remap node names in a query according to a relabeling mapping."""
    new_query = query
    # Sort by length descending to avoid partial replacements
    for old_name in sorted(mapping.keys(), key=len, reverse=True):
        new_query = new_query.replace(old_name, mapping[old_name])
    return new_query


# ─── Experiment Pipeline ───

def generate_all_prompts(output_dir: str = OUTPUT_DIR, use_expanded: bool = False):
    """Generate all prompt variants for all diagnostics.

    Args:
        use_expanded: If True, use generate_graphs_expanded() which includes
                      Barabási–Albert, Watts–Strogatz, and real-world graphs.
    """
    os.makedirs(output_dir, exist_ok=True)

    if use_expanded:
        graphs = generate_graphs_expanded()
    else:
        graphs = generate_graphs()
    print(f"Generated {len(graphs)} graphs")

    all_prompts = {
        "format_stability": [],    # RQ1
        "order_sensitivity": [],   # RQ2
        "isomorphism_consistency": [],  # RQ3
        "distractor_robustness": [],    # RQ4
    }

    for ginfo in graphs:
        G = ginfo["graph"]
        gid = ginfo["graph_id"]

        for task in TASKS:
            gt_queries = compute_ground_truth(G, task, seed=ginfo["seed"])
            if not gt_queries:
                continue

            for qinfo in gt_queries:
                query = qinfo["query"]
                answer = qinfo["answer"]
                source_nodes = qinfo.get("source_nodes", [])

                # ─── RQ1: Format Stability ───
                instance_key = f"{gid}|{task}|{query}"

                node_order = get_node_order(G, "canonical", source_nodes)
                edge_order = get_edge_order(G, node_order, "canonical")

                for fmt in FORMATS:
                    if fmt == "edge_list":
                        graph_text = serialize_edge_list(G, node_order, edge_order)
                    elif fmt == "adjacency_list":
                        graph_text = serialize_adjacency_list(G, node_order)
                    elif fmt == "json":
                        graph_text = serialize_json(G, node_order, edge_order)
                    elif fmt == "natural_language":
                        graph_text = serialize_natural_language(G, node_order)

                    prompt = build_prompt(graph_text, query, task)
                    all_prompts["format_stability"].append({
                        "graph_id": gid,
                        "graph_type": ginfo.get("graph_type", "erdos_renyi"),
                        "task": task,
                        "format": fmt,
                        "ordering": "canonical",
                        "query": query,
                        "ground_truth": answer,
                        "prompt": prompt,
                        "source_nodes": source_nodes,
                        "perturbation": "none",
                        "group_key": instance_key,
                    })

                # ─── RQ2: Order Sensitivity ───
                fmt = "edge_list"  # fixed format for order test
                for ordering in ORDERINGS:
                    node_order = get_node_order(G, ordering, source_nodes, seed=ginfo["seed"])
                    edge_order = get_edge_order(G, node_order, ordering, seed=ginfo["seed"])
                    graph_text = serialize_edge_list(G, node_order, edge_order)
                    prompt = build_prompt(graph_text, query, task)
                    all_prompts["order_sensitivity"].append({
                        "graph_id": gid,
                        "graph_type": ginfo.get("graph_type", "erdos_renyi"),
                        "task": task,
                        "format": fmt,
                        "ordering": ordering,
                        "query": query,
                        "ground_truth": answer,
                        "prompt": prompt,
                        "source_nodes": source_nodes,
                        "perturbation": "none",
                        "group_key": instance_key,
                    })

                # ─── RQ3: Isomorphism Consistency ───
                fmt = "edge_list"
                ordering = "canonical"
                base_key = instance_key  # shared key for grouping original + relabels

                # Original
                node_order = get_node_order(G, ordering, source_nodes)
                edge_order = get_edge_order(G, node_order, ordering)
                graph_text = serialize_edge_list(G, node_order, edge_order)
                prompt = build_prompt(graph_text, query, task)
                all_prompts["isomorphism_consistency"].append({
                    "graph_id": gid,
                    "task": task,
                    "format": fmt,
                    "ordering": ordering,
                    "query": query,
                    "ground_truth": answer,
                    "prompt": prompt,
                    "source_nodes": source_nodes,
                    "perturbation": "original",
                    "group_key": f"{instance_key}|original",
                    "base_key": base_key,
                })

                # Neutral relabeling
                G_relabeled, mapping = relabel_neutral(G)
                new_query = remap_query(query, mapping)
                new_source = [mapping.get(s, s) for s in source_nodes]
                node_order_r = get_node_order(G_relabeled, ordering, new_source)
                edge_order_r = get_edge_order(G_relabeled, node_order_r, ordering)
                graph_text_r = serialize_edge_list(G_relabeled, node_order_r, edge_order_r)
                prompt_r = build_prompt(graph_text_r, new_query, task)
                all_prompts["isomorphism_consistency"].append({
                    "graph_id": gid,
                    "task": task,
                    "format": fmt,
                    "ordering": ordering,
                    "query": new_query,
                    "ground_truth": answer,
                    "prompt": prompt_r,
                    "source_nodes": new_source,
                    "perturbation": "neutral_relabel",
                    "group_key": f"{instance_key}|neutral_relabel",
                    "base_key": base_key,
                    "relabel_mapping": mapping,
                })

                # Recovered degree-congruent semantic-name relabeling. The
                # historical perturbation key is retained for compatibility.
                G_mis, mapping_mis = relabel_misleading(G)
                new_query_mis = remap_query(query, mapping_mis)
                new_source_mis = [mapping_mis.get(s, s) for s in source_nodes]
                node_order_m = get_node_order(G_mis, ordering, new_source_mis)
                edge_order_m = get_edge_order(G_mis, node_order_m, ordering)
                graph_text_m = serialize_edge_list(G_mis, node_order_m, edge_order_m)
                prompt_m = build_prompt(graph_text_m, new_query_mis, task)
                all_prompts["isomorphism_consistency"].append({
                    "graph_id": gid,
                    "task": task,
                    "format": fmt,
                    "ordering": ordering,
                    "query": new_query_mis,
                    "ground_truth": answer,
                    "prompt": prompt_m,
                    "source_nodes": new_source_mis,
                    "perturbation": "misleading_relabel",
                    "group_key": f"{instance_key}|misleading_relabel",
                    "base_key": base_key,
                    "relabel_mapping": mapping_mis,
                    "intervention_status": "degree_congruent_semantic_names_unverified_against_paper",
                })

                # ─── RQ4: Distractor Robustness ───
                fmt = "edge_list"
                ordering = "canonical"

                # Original (without distractor) — reuse from format_stability
                node_order = get_node_order(G, ordering, source_nodes)
                edge_order = get_edge_order(G, node_order, ordering)
                graph_text = serialize_edge_list(G, node_order, edge_order)
                prompt = build_prompt(graph_text, query, task)
                all_prompts["distractor_robustness"].append({
                    "graph_id": gid,
                    "task": task,
                    "format": fmt,
                    "ordering": ordering,
                    "query": query,
                    "ground_truth": answer,
                    "prompt": prompt,
                    "source_nodes": source_nodes,
                    "perturbation": "no_distractor",
                    "group_key": instance_key,
                })

                # With each distractor type
                for dtype in DISTRACTOR_TYPES:
                    G_dist = DISTRACTOR_FUNCTIONS[dtype](G, seed=ginfo["seed"])
                    # Verify label-preservation: distractor must not change ground truth.
                    # Check the exact sampled query; do not rerun the query sampler
                    # after changing the graph's node population.
                    answer_dist = compute_answer_for_source_nodes(G_dist, task, source_nodes)
                    if answer_dist != answer:
                        print(f"WARNING: distractor {dtype} changed ground truth for {gid}/{task}! Skipping.")
                        continue
                    node_order_d = get_node_order(G_dist, ordering, source_nodes)
                    edge_order_d = get_edge_order(G_dist, node_order_d, ordering)
                    graph_text_d = serialize_edge_list(G_dist, node_order_d, edge_order_d)
                    prompt_d = build_prompt(graph_text_d, query, task)
                    all_prompts["distractor_robustness"].append({
                        "graph_id": gid,
                        "graph_type": ginfo.get("graph_type", "erdos_renyi"),
                        "task": task,
                        "format": fmt,
                        "ordering": ordering,
                        "query": query,
                        "ground_truth": answer,
                        "prompt": prompt_d,
                        "source_nodes": source_nodes,
                        "perturbation": dtype,
                        "group_key": instance_key,
                    })

    # Save prompts
    for rq_name, prompts in all_prompts.items():
        save_path = os.path.join(output_dir, f"prompts_{rq_name}.jsonl")
        with open(save_path, "w") as f:
            for p in prompts:
                # Don't save the full prompt text in the JSONL — too large
                p_save = {k: v for k, v in p.items() if k != "prompt"}
                f.write(json.dumps(p_save) + "\n")
        print(f"  {rq_name}: {len(prompts)} prompts saved to {save_path}")

    # Save full prompts separately (for LLM inference)
    full_prompts_path = os.path.join(output_dir, "all_prompts_full.json")
    # Convert to serializable format
    saveable = {}
    for rq_name, prompts in all_prompts.items():
        saveable[rq_name] = []
        for p in prompts:
            sp = dict(p)
            # Convert non-serializable types
            if "relabel_mapping" in sp:
                sp["relabel_mapping"] = {str(k): str(v) for k, v in sp["relabel_mapping"].items()}
            saveable[rq_name].append(sp)

    with open(full_prompts_path, "w") as f:
        json.dump(saveable, f, indent=2)
    print(f"Full prompts saved to {full_prompts_path}")

    # Summary statistics
    print(f"\n=== Prompt Generation Summary ===")
    for rq_name, prompts in all_prompts.items():
        groups = set(p["group_key"].split("|")[0] + "|" + p["task"] for p in prompts)
        print(f"  {rq_name}: {len(prompts)} prompts, {len(groups)} graph-task pairs")

    return all_prompts


# ─── LLM Inference ───

def query_openai(prompts: List[Dict], model: str = "gpt-4o-mini",
                 output_file: str = None, max_concurrent: int = 10) -> List[Dict]:
    """Query OpenAI API for a list of prompts. Returns results with answers."""
    try:
        from openai import OpenAI
    except ImportError:
        print("openai package not installed. Run: pip install openai")
        return []

    client = OpenAI()  # Uses OPENAI_API_KEY env var
    results = []

    for i, p in enumerate(prompts):
        if (i + 1) % 100 == 0:
            print(f"  Querying {i+1}/{len(prompts)}...", flush=True)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a precise graph reasoning assistant. Answer graph questions based only on the provided graph description. Give your final answer clearly."},
                    {"role": "user", "content": p["prompt"]},
                ],
                temperature=0.0,
                max_tokens=256,
            )
            answer_text = response.choices[0].message.content.strip()
            # Extract token usage
            usage = response.usage
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
        except Exception as e:
            answer_text = f"ERROR: {str(e)}"
            prompt_tokens = 0
            completion_tokens = 0

        result = dict(p)
        result["model_answer"] = answer_text
        result["model"] = model
        result["prompt_tokens"] = prompt_tokens
        result["completion_tokens"] = completion_tokens
        results.append(result)

        # Save incrementally
        if output_file and (i + 1) % 50 == 0:
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)

    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Results saved to {output_file}")

    return results


def query_local_model(prompts: List[Dict], model_name: str,
                      output_file: str = None) -> List[Dict]:
    """Query a local model via transformers."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except ImportError:
        print("transformers/torch not installed.")
        return []

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if not torch.cuda.is_available():
        model = model.to("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    model.eval()

    results = []
    for i, p in enumerate(prompts):
        if (i + 1) % 50 == 0:
            print(f"  Querying {i+1}/{len(prompts)}...", flush=True)

        messages = [
            {"role": "system", "content": "You are a precise graph reasoning assistant. Answer graph questions based only on the provided graph description. Give your final answer clearly. /no_think"},
            {"role": "user", "content": p["prompt"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.0,
                do_sample=False,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        answer_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        result = dict(p)
        result["model_answer"] = answer_text
        result["model"] = model_name
        result["prompt_tokens"] = inputs["input_ids"].shape[1]
        result["completion_tokens"] = len(new_tokens)
        results.append(result)

        if output_file and (i + 1) % 50 == 0:
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)

    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

    return results


# ─── Answer Extraction ───

def extract_answer(model_text: str, task: str) -> Optional[str]:
    """Extract the final answer from model output.

    Strategy:
    1. Look for explicit answer markers ("Final answer: yes", "Answer: 3", etc.)
    2. Look for the last clear answer statement
    3. For binary tasks, look for yes/no patterns
    4. For shortest path, look for a number

    Returns "yes", "no", a number string, or "unclear".
    """
    import re

    text = model_text.strip()
    text_lower = text.lower()

    # ── Step 1: Find explicit answer section ──
    answer_section = text
    for marker in ["final answer:", "answer:", "result:", "conclusion:"]:
        # Case-insensitive split, keep everything after the LAST occurrence
        idx = text_lower.rfind(marker)
        if idx >= 0:
            answer_section = text[idx + len(marker):].strip()
            break

    answer_lower = answer_section.lower()

    # ── Step 2: Task-specific extraction ──
    if task in ("connectivity", "triangle_detection", "common_neighbor"):
        # Binary yes/no task

        # Pattern 1: "yes" or "no" as the first word after the marker
        first_word = answer_lower.split()[0] if answer_lower.split() else ""
        if first_word.rstrip(".,;!") == "yes":
            return "yes"
        if first_word.rstrip(".,;!") == "no":
            return "no"

        # Pattern 2: "the answer is yes/no"
        if re.search(r'\banswer is\s+yes\b', answer_lower):
            return "yes"
        if re.search(r'\banswer is\s+no\b', answer_lower):
            return "no"

        # Pattern 3: "there is/is no path/triangle/common neighbor"
        if task == "connectivity":
            if re.search(r'\bthere is (?:a |no )?path\b', answer_lower):
                if re.search(r'\bthere is no path\b', answer_lower):
                    return "no"
                return "yes"
        elif task == "triangle_detection":
            if re.search(r'\bdoes (?:not |n\'t )?participate\b', answer_lower):
                if re.search(r'\bdoes not participate\b|\bdoesn\'t participate\b', answer_lower):
                    return "no"
                return "yes"
        elif task == "common_neighbor":
            if re.search(r'\bdo(?:es)? (?:not |n\'t )?share\b', answer_lower):
                if re.search(r'\bdo not share\b|\bdo not have\b|\bno common\b', answer_lower):
                    return "no"
                return "yes"

        # Pattern 4: Last occurrence of yes/no in the answer section
        # Weight towards the end of the text (model's conclusion)
        last_yes = answer_lower.rfind("yes")
        last_no = answer_lower.rfind("no")

        # Only count if they appear as standalone words
        if last_yes >= 0 and re.search(r'\byes\b', answer_lower[last_yes:]):
            yes_pos = last_yes
        else:
            yes_pos = -1
        if last_no >= 0 and re.search(r'\bno\b', answer_lower[last_no:]):
            no_pos = last_no
        else:
            no_pos = -1

        if yes_pos >= 0 and no_pos < 0:
            return "yes"
        if no_pos >= 0 and yes_pos < 0:
            return "no"
        if yes_pos >= 0 and no_pos >= 0:
            # Both present — use whichever appears last
            return "yes" if yes_pos > no_pos else "no"

        return "unclear"

    elif task == "shortest_path":
        # Numeric answer

        # Pattern 1: "shortest path length is N" or "distance is N"
        m = re.search(r'(?:length|distance|steps|edges)\s+(?:is|of|=)\s+(\d+)', answer_lower)
        if m:
            return m.group(1)

        # Pattern 2: "N edges" or "N steps"
        m = re.search(r'\b(\d+)\s+(?:edges?|steps?|hops?)\b', answer_lower)
        if m:
            return m.group(1)

        # Pattern 3: Just a number at the start of the answer section
        m = re.match(r'^(\d+)', answer_section.strip())
        if m:
            return m.group(1)

        # Pattern 4: Last number in the answer section
        numbers = re.findall(r'\b(\d+)\b', answer_section)
        if numbers:
            return numbers[-1]

        return "unclear"

    return "unclear"


# ─── Bootstrap Confidence Intervals ───

def bootstrap_ci(values: List[float], n_bootstrap: int = 1000, ci: float = 0.95, seed: int = 42) -> Tuple[float, float, float]:
    """Compute mean and bootstrap confidence interval for a list of 0/1 values.
    Returns (mean, ci_low, ci_high).
    """
    arr = np.array(values)
    mean = arr.mean()
    rng = np.random.RandomState(seed)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_means.append(sample.mean())
    alpha = (1 - ci) / 2
    ci_low = np.percentile(boot_means, 100 * alpha)
    ci_high = np.percentile(boot_means, 100 * (1 - alpha))
    return float(mean), float(ci_low), float(ci_high)


# ─── Metrics Computation ───

def compute_format_stability(results: List[Dict]) -> Dict:
    """Compute Format Stability (FS) from RQ1 results."""
    # Group by graph-task pair
    groups = defaultdict(list)
    for r in results:
        key = r["group_key"]
        groups[key].append(r)

    agreements = []
    per_task = defaultdict(list)
    per_format_pair = defaultdict(list)
    n_total_pairs = 0    # all comparable format pairs (incl. those with unclear)
    n_unclear_pairs = 0  # pairs where >=1 side was "unclear" (excluded from FS)

    format_pairs = list(itertools.combinations(FORMATS, 2))

    for key, group in groups.items():
        answers = {}
        for r in group:
            extracted = extract_answer(r["model_answer"], r["task"])
            answers[r["format"]] = extracted

        # Pairwise agreement. Convention: "unclear" is treated as MISSING
        # (excluded from the denominator), NOT as a value that agrees with
        # itself. Two "unclear" answers must not count as faithfulness.
        for fmt_a, fmt_b in format_pairs:
            if fmt_a in answers and fmt_b in answers:
                n_total_pairs += 1
                a_ans, b_ans = answers[fmt_a], answers[fmt_b]
                if a_ans == "unclear" or b_ans == "unclear":
                    n_unclear_pairs += 1  # skipped, not counted as agree
                    continue
                agree = 1.0 if a_ans == b_ans else 0.0
                agreements.append(agree)
                per_format_pair[(fmt_a, fmt_b)].append(agree)

        # Overall agreement across all non-unclear formats for this instance
        vals = [v for v in answers.values() if v != "unclear"]
        if len(vals) >= 2:
            all_same = 1.0 if len(set(vals)) == 1 else 0.0
            for r in group:
                per_task[r["task"]].append(all_same)

    fs, fs_lo, fs_hi = bootstrap_ci(agreements) if agreements else (0.0, 0.0, 0.0)
    fs_per_task = {t: np.mean(v) for t, v in per_task.items() if not t.startswith("__")}
    fs_per_pair = {f"{a}|{b}": np.mean(v) for (a, b), v in per_format_pair.items()}
    fs_unclear_rate = (n_unclear_pairs / n_total_pairs) if n_total_pairs else 0.0

    return {
        "FS": fs,
        "FS_ci": (fs_lo, fs_hi),
        # Fraction of pairwise comparisons where >=1 side was "unclear"
        # (these were excluded from the FS denominator). High => the model is
        # often too incoherent to extract an answer, which FS alone hides.
        "FS_unclear_rate": fs_unclear_rate,
        "FS_per_task": fs_per_task,
        "FS_per_pair": fs_per_pair,
    }


def compute_order_sensitivity(results: List[Dict]) -> Dict:
    """Compute Order Sensitivity (OS) from RQ2 results."""
    groups = defaultdict(list)
    for r in results:
        key = r["group_key"]
        groups[key].append(r)

    flip_rates = []
    per_task = defaultdict(list)
    per_ordering = defaultdict(list)
    n_total = 0      # comparisons with a usable canonical answer
    n_unclear = 0    # comparisons skipped because canonical or extracted was unclear

    for key, group in groups.items():
        canonical_answer = None
        for r in group:
            extracted = extract_answer(r["model_answer"], r["task"])
            if r["ordering"] == "canonical":
                canonical_answer = extracted

        # Skip the whole instance if the canonical ordering was unclear —
        # there is no stable reference to compare against, and counting every
        # real answer as a "flip" would inflate OS.
        if canonical_answer is None or canonical_answer == "unclear":
            continue

        for r in group:
            if r["ordering"] == "canonical":
                continue
            extracted = extract_answer(r["model_answer"], r["task"])
            n_total += 1
            # Convention: "unclear" on the perturbed side is MISSING, not a flip.
            if extracted == "unclear":
                n_unclear += 1
                continue
            flipped = 1.0 if extracted != canonical_answer else 0.0
            flip_rates.append(flipped)
            per_task[r["task"]].append(flipped)
            per_ordering[r["ordering"]].append(flipped)

    os_score, os_lo, os_hi = bootstrap_ci(flip_rates) if flip_rates else (0.0, 0.0, 0.0)
    os_per_task = {t: np.mean(v) for t, v in per_task.items()}
    os_per_ordering = {o: np.mean(v) for o, v in per_ordering.items()}
    os_unclear_rate = (n_unclear / n_total) if n_total else 0.0

    return {
        "OS": os_score,
        "OS_ci": (os_lo, os_hi),
        "OS_unclear_rate": os_unclear_rate,
        "OS_per_task": os_per_task,
        "OS_per_ordering": os_per_ordering,
    }


def compute_isomorphism_consistency(results: List[Dict]) -> Dict:
    """Compute Isomorphism Consistency (IC) from RQ3 results."""
    groups = defaultdict(list)
    for r in results:
        # Group by base_key (graph+task+query, without perturbation type)
        key = r.get("base_key", "|".join(r["group_key"].split("|")[:3]))
        groups[key].append(r)

    ic_neutral = []
    ic_misleading = []
    per_task_neutral = defaultdict(list)
    per_task_misleading = defaultdict(list)
    n_total_n = n_unclear_n = 0  # neutral comparisons: total / unclear (excluded)
    n_total_m = n_unclear_m = 0  # misleading comparisons

    for key, group in groups.items():
        original_answer = None
        for r in group:
            extracted = extract_answer(r["model_answer"], r["task"])
            if r["perturbation"] == "original":
                original_answer = extracted

        # No stable reference if the original was unclear.
        if original_answer is None or original_answer == "unclear":
            continue

        for r in group:
            extracted = extract_answer(r["model_answer"], r["task"])
            # Convention: "unclear" on the perturbed side is MISSING, not agreement.
            if extracted == "unclear":
                if r["perturbation"] == "neutral_relabel":
                    n_total_n += 1; n_unclear_n += 1
                elif r["perturbation"] == "misleading_relabel":
                    n_total_m += 1; n_unclear_m += 1
                continue
            agree = 1.0 if extracted == original_answer else 0.0
            if r["perturbation"] == "neutral_relabel":
                ic_neutral.append(agree)
                per_task_neutral[r["task"]].append(agree)
                n_total_n += 1
            elif r["perturbation"] == "misleading_relabel":
                ic_misleading.append(agree)
                per_task_misleading[r["task"]].append(agree)
                n_total_m += 1

    ic_n, ic_n_lo, ic_n_hi = bootstrap_ci(ic_neutral) if ic_neutral else (0.0, 0.0, 0.0)
    ic_m, ic_m_lo, ic_m_hi = bootstrap_ci(ic_misleading) if ic_misleading else (0.0, 0.0, 0.0)

    return {
        "IC_neutral": ic_n,
        "IC_neutral_ci": (ic_n_lo, ic_n_hi),
        "IC_neutral_unclear_rate": (n_unclear_n / n_total_n) if n_total_n else 0.0,
        "IC_misleading": ic_m,
        "IC_misleading_ci": (ic_m_lo, ic_m_hi),
        "IC_misleading_unclear_rate": (n_unclear_m / n_total_m) if n_total_m else 0.0,
        "IC_neutral_per_task": {t: np.mean(v) for t, v in per_task_neutral.items()},
        "IC_misleading_per_task": {t: np.mean(v) for t, v in per_task_misleading.items()},
    }


def compute_distractor_robustness(results: List[Dict]) -> Dict:
    """Compute Distractor Robustness (DR) from RQ4 results."""
    groups = defaultdict(list)
    for r in results:
        key = r["group_key"]
        groups[key].append(r)

    dr_scores = []
    per_task = defaultdict(list)
    per_distractor = defaultdict(list)
    n_total = 0    # comparisons with a usable baseline
    n_unclear = 0  # comparisons skipped because baseline or extracted was unclear

    for key, group in groups.items():
        baseline_answer = None
        for r in group:
            extracted = extract_answer(r["model_answer"], r["task"])
            if r["perturbation"] == "no_distractor":
                baseline_answer = extracted

        # No stable reference if the no-distractor baseline was unclear.
        if baseline_answer is None or baseline_answer == "unclear":
            continue

        for r in group:
            if r["perturbation"] == "no_distractor":
                continue
            extracted = extract_answer(r["model_answer"], r["task"])
            n_total += 1
            # Convention: "unclear" on the distractor side is MISSING, not agreement.
            if extracted == "unclear":
                n_unclear += 1
                continue
            same = 1.0 if extracted == baseline_answer else 0.0
            dr_scores.append(same)
            per_task[r["task"]].append(same)
            per_distractor[r["perturbation"]].append(same)

    dr, dr_lo, dr_hi = bootstrap_ci(dr_scores) if dr_scores else (0.0, 0.0, 0.0)
    dr_per_task = {t: np.mean(v) for t, v in per_task.items()}
    dr_per_distractor = {d: np.mean(v) for d, v in per_distractor.items()}
    dr_unclear_rate = (n_unclear / n_total) if n_total else 0.0

    return {
        "DR": dr,
        "DR_ci": (dr_lo, dr_hi),
        "DR_unclear_rate": dr_unclear_rate,
        "DR_per_task": dr_per_task,
        "DR_per_distractor": dr_per_distractor,
    }


# ─── Main Runner ───

def run_experiments(output_dir: str = OUTPUT_DIR):
    """Run the full experiment pipeline."""
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Generate prompts
    print("=" * 60)
    print("Step 1: Generating prompts...")
    print("=" * 60)
    all_prompts = generate_all_prompts(output_dir)

    # Step 2: Run inference for each model × RQ
    print("\n" + "=" * 60)
    print("Step 2: Running LLM inference...")
    print("=" * 60)

    all_results = {}

    for model_name in MODELS:
        print(f"\n--- Model: {model_name} ---")
        model_results = {}

        for rq_name, prompts in all_prompts.items():
            print(f"\n  Running {rq_name} ({len(prompts)} prompts)...")

            results_file = os.path.join(output_dir, f"results_{model_name}_{rq_name}.json")

            if os.path.exists(results_file):
                print(f"  Loading existing results from {results_file}")
                with open(results_file) as f:
                    results = json.load(f)
            else:
                if "gpt" in model_name.lower():
                    results = query_openai(prompts, model=model_name, output_file=results_file)
                else:
                    results = query_local_model(prompts, model_name=model_name, output_file=results_file)

            model_results[rq_name] = results

        all_results[model_name] = model_results

    # Step 3: Compute metrics
    print("\n" + "=" * 60)
    print("Step 3: Computing metrics...")
    print("=" * 60)

    summary = {}
    for model_name, model_results in all_results.items():
        print(f"\n--- {model_name} ---")
        metrics = {}

        if "format_stability" in model_results and model_results["format_stability"]:
            fs = compute_format_stability(model_results["format_stability"])
            metrics["FS"] = fs
            print(f"  Format Stability (FS): {fs['FS']:.4f}")

        if "order_sensitivity" in model_results and model_results["order_sensitivity"]:
            os_m = compute_order_sensitivity(model_results["order_sensitivity"])
            metrics["OS"] = os_m
            print(f"  Order Sensitivity (OS): {os_m['OS']:.4f}")

        if "isomorphism_consistency" in model_results and model_results["isomorphism_consistency"]:
            ic = compute_isomorphism_consistency(model_results["isomorphism_consistency"])
            metrics["IC"] = ic
            print(f"  Isomorphism Consistency (IC-neutral): {ic['IC_neutral']:.4f}")
            print(f"  Isomorphism Consistency (IC-misleading): {ic['IC_misleading']:.4f}")

        if "distractor_robustness" in model_results and model_results["distractor_robustness"]:
            dr = compute_distractor_robustness(model_results["distractor_robustness"])
            metrics["DR"] = dr
            print(f"  Distractor Robustness (DR): {dr['DR']:.4f}")

        summary[model_name] = metrics

    # Step 4: Print main result table
    print("\n" + "=" * 80)
    print("MAIN RESULT TABLE")
    print("=" * 80)
    print(f"  {'Model':<25} {'Accuracy':>10} {'FS':>8} {'OS':>8} {'IC-N':>8} {'IC-M':>8} {'DR':>8}")
    print(f"  {'-'*25} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for model_name, metrics in summary.items():
        fs = metrics.get("FS", {}).get("FS", 0.0)
        os_m = metrics.get("OS", {}).get("OS", 0.0)
        ic_n = metrics.get("IC", {}).get("IC_neutral", 0.0)
        ic_m = metrics.get("IC", {}).get("IC_misleading", 0.0)
        dr = metrics.get("DR", {}).get("DR", 0.0)
        print(f"  {model_name:<25} {'—':>10} {fs:>8.4f} {os_m:>8.4f} {ic_n:>8.4f} {ic_m:>8.4f} {dr:>8.4f}")

    # Save summary
    summary_path = os.path.join(output_dir, "metrics_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nMetrics saved to {summary_path}")

    return summary


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "generate":
        # Only generate prompts, don't run inference
        generate_all_prompts()
    else:
        raise SystemExit(
            "Direct inference from groundlm_serialization.py is disabled in the "
            "public artifact because it lacks complete run provenance. Use "
            "run_api_models.py or run_qwen_local.py after reading RELEASE_STATUS.md. "
            "Prompt-only generation remains available with: "
            "python groundlm_serialization.py generate"
        )
