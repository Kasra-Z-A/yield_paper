"""Network visualization for Yearn vault update_debt token flows."""

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

# pygraphviz gives the best hierarchical layouts; fall back to pydot, then spring
try:
    from networkx.drawing.nx_pydot import graphviz_layout
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    try:
        from networkx.drawing.nx_agraph import graphviz_layout
        GRAPHVIZ_AVAILABLE = True
    except ImportError:
        GRAPHVIZ_AVAILABLE = False
        graphviz_layout = None


def load_address_labels(labels_path: Optional[str] = None) -> Dict[str, str]:
    """
    Load address labels from JSON file.

    If no path is given, looks for data/address_labels.json relative to the
    package root. Returns an empty dict if the file is not found.
    """
    if labels_path is None:
        current_dir = Path(__file__).parent
        labels_path = current_dir.parent.parent / "data" / "address_labels.json"

    labels_path = Path(labels_path)

    if not labels_path.exists():
        print(f"Warning: Address labels file not found at {labels_path}")
        return {}

    with open(labels_path, 'r') as f:
        data = json.load(f)

    return {
        address.lower(): info["name"]
        for address, info in data.get("addresses", {}).items()
    }


def create_update_debt_network(
    transactions_df: pd.DataFrame,
    transfers_df: pd.DataFrame,
    vault_address: str,
    function_signatures: Dict[str, list]
) -> nx.DiGraph:
    """
    Build a weighted directed graph from update_debt token transfers.

    Only includes transactions where the first transfer originates from the vault
    (i.e. the vault is deploying funds to strategies, not receiving them back).
    Edge weights represent the total token volume across all transactions.
    """
    vault_address = vault_address.lower()
    update_debt_sigs = function_signatures.get('strategy_investment', [])

    tx_df = transactions_df.copy()
    tx_df['func_sig'] = tx_df['input'].str[:10]

    update_debt_txs = tx_df[
        (tx_df['func_sig'].isin(update_debt_sigs)) &
        (tx_df['to_address'].str.lower() == vault_address)
    ]

    update_debt_tx_hashes = set(update_debt_txs['tx_hash'])
    transfers = transfers_df[transfers_df['tx_hash'].isin(update_debt_tx_hashes)].copy()

    transfers['from_address'] = transfers['from_address'].str.lower()
    transfers['to_address'] = transfers['to_address'].str.lower()

    # Keep only transactions where the vault initiates the first transfer
    transfers_sorted = transfers.sort_values(['tx_hash', 'log_index'])
    first_transfers = transfers_sorted.groupby('tx_hash').first().reset_index()
    outgoing_tx_hashes = first_transfers[
        first_transfers['from_address'] == vault_address
    ]['tx_hash']

    outgoing_transfers = transfers[transfers['tx_hash'].isin(outgoing_tx_hashes)].copy()

    G = nx.DiGraph()

    for _, row in outgoing_transfers.iterrows():
        from_addr = row['from_address']
        to_addr = row['to_address']
        value = row['value']

        if G.has_edge(from_addr, to_addr):
            G[from_addr][to_addr]['weight'] += value
        else:
            G.add_edge(from_addr, to_addr, weight=value)

    return G


def visualize_update_debt_network(
    G: nx.DiGraph,
    vault_address: str,
    title: str = "Update Debt Token Transfer Network",
    figsize: Tuple[int, int] = (16, 12),
    contract_labels: Optional[Dict[str, str]] = None,
    labels_path: Optional[str] = None,
    save_path: Optional[str] = None,
    colormap: str = 'viridis',
    node_size: int = 800,
    label_y_offset_fraction: float = 0.03,
):
    """
    Visualize the update_debt network with a hierarchical dot layout.

    The vault node is drawn in red. All other nodes are colored by weighted
    in-degree (in-strength) using the specified colormap. Edge widths are
    scaled proportionally to token volume. Edge labels show total volume in
    K/M notation. Node labels are placed below each node. Falls back to
    spring layout if Graphviz is not installed.

    Returns the matplotlib Figure object.
    """
    if contract_labels is None:
        contract_labels = load_address_labels(labels_path)

    vault_address = vault_address.lower()

    fig, ax = plt.subplots(figsize=figsize)

    if GRAPHVIZ_AVAILABLE:
        try:
            pos = graphviz_layout(G, prog='dot', root=vault_address)
        except Exception as e:
            print(f"Note: Graphviz layout failed ({e}), using spring layout")
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    else:
        print("Note: For hierarchical layout, install: pip install pydot")
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    labels = {}
    for node in G.nodes():
        if node in contract_labels:
            labels[node] = contract_labels[node]
        else:
            labels[node] = f"{node[:6]}...{node[-4:]}"

    edges = list(G.edges())
    weights = [G[u][v]['weight'] for u, v in edges]

    if weights:
        max_weight = max(weights)
        min_weight = min(weights)
        if max_weight > min_weight:
            # Scale linearly from 0.6 to 5.0
            edge_widths = [0.6 + 4.4 * (w - min_weight) / (max_weight - min_weight) for w in weights]
        else:
            edge_widths = [2.8] * len(weights)
    else:
        edge_widths = [2.8]

    nx.draw_networkx_edges(
        G, pos,
        width=edge_widths,
        edge_color='gray',
        alpha=0.6,
        arrows=True,
        arrowsize=20,
        arrowstyle='->',
        ax=ax
    )

    all_nodes = list(G.nodes())
    vault_nodes = [n for n in all_nodes if n.lower() == vault_address]
    regular_nodes = [n for n in all_nodes if n.lower() != vault_address]

    if regular_nodes:
        in_strength = dict(G.in_degree(weight='weight'))
        node_values = [in_strength.get(n, 0) for n in regular_nodes]
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=regular_nodes,
            node_color=node_values,
            node_size=node_size,
            cmap=plt.cm.get_cmap(colormap),
            alpha=0.9,
            ax=ax
        )

    if vault_nodes:
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=vault_nodes,
            node_color='red',
            node_size=node_size,
            alpha=0.9,
            ax=ax
        )

    # Offset label positions below each node
    if pos:
        y_values = [y for _, y in pos.values()]
        y_span = max(y_values) - min(y_values) if len(y_values) > 1 else 1.0
        y_offset = -label_y_offset_fraction * (y_span if y_span > 0 else 1.0)
    else:
        y_offset = 0.0

    label_pos = {node: (x, y + y_offset) for node, (x, y) in pos.items()}

    nx.draw_networkx_labels(
        G, label_pos,
        labels=labels,
        font_size=8,
        font_weight='bold',
        ax=ax,
        verticalalignment='top',
    )

    edge_labels = {}
    for u, v in G.edges():
        weight = G[u][v]['weight']
        if weight >= 1_000_000:
            edge_labels[(u, v)] = f"${weight/1_000_000:.1f}M"
        elif weight >= 1_000:
            edge_labels[(u, v)] = f"${weight/1_000:.1f}K"
        else:
            edge_labels[(u, v)] = f"${weight:.0f}"

    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_size=8,
        ax=ax
    )

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Network saved to: {save_path}")

    return fig


def export_network_data(
    G: nx.DiGraph,
    output_path: str,
    contract_labels: Optional[Dict[str, str]] = None,
    labels_path: Optional[str] = None
):
    """
    Export network nodes and edges to CSV files for further analysis.

    Writes two files: {output_path}_nodes.csv and {output_path}_edges.csv.
    """
    if contract_labels is None:
        contract_labels = load_address_labels(labels_path)

    nodes_data = []
    for node in G.nodes():
        label = contract_labels.get(node, f"{node[:10]}...").replace('\n', ' ')
        nodes_data.append({
            'address': node,
            'label': label,
            'in_degree': G.in_degree(node),
            'out_degree': G.out_degree(node),
            'in_strength': sum(G[u][node]['weight'] for u in G.predecessors(node)),
            'out_strength': sum(G[node][v]['weight'] for v in G.successors(node))
        })

    nodes_df = pd.DataFrame(nodes_data)
    nodes_path = f"{output_path}_nodes.csv"
    nodes_df.to_csv(nodes_path, index=False)
    print(f"Nodes exported to: {nodes_path}")

    edges_data = []
    for u, v in G.edges():
        edges_data.append({
            'from_address': u,
            'to_address': v,
            'from_label': contract_labels.get(u, f"{u[:10]}...").replace('\n', ' '),
            'to_label': contract_labels.get(v, f"{v[:10]}...").replace('\n', ' '),
            'weight': G[u][v]['weight']
        })

    edges_df = pd.DataFrame(edges_data)
    edges_path = f"{output_path}_edges.csv"
    edges_df.to_csv(edges_path, index=False)
    print(f"Edges exported to: {edges_path}")

    print(f"\nNetwork Summary:")
    print(f"  Nodes: {len(G.nodes())}")
    print(f"  Edges: {len(G.edges())}")
    print(f"  Total token flow: ${sum(G[u][v]['weight'] for u, v in G.edges()):,.2f}")
