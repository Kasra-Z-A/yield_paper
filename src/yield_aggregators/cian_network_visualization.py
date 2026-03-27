"""Network visualization for Cian vault token transfer flows."""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import json
from typing import Dict, List, Optional, Set
from pathlib import Path


def load_address_labels(labels_path: str = None) -> Dict[str, str]:
    """
    Load address labels from JSON file.

    If no path is given, looks for data/address_labels.json relative to the
    package root. Returns an empty dict if the file is not found.
    """
    if labels_path is None:
        labels_path = Path(__file__).parent.parent.parent / 'data' / 'address_labels.json'

    try:
        with open(labels_path, 'r') as f:
            data = json.load(f)
            return {
                addr.lower(): info.get('name', addr)
                for addr, info in data.get('addresses', {}).items()
            }
    except FileNotFoundError:
        print(f"Warning: Address labels file not found at {labels_path}")
        return {}
    except Exception as e:
        print(f"Warning: Error loading address labels: {e}")
        return {}


def get_transaction_hashes_by_function(
    vault_transactions: pd.DataFrame,
    target_address: str,
    function_selectors: List[str]
) -> List[str]:
    """Get transaction hashes matching specific function selectors sent to a given address."""
    if isinstance(function_selectors, str):
        function_selectors = [function_selectors]

    mask = (
        vault_transactions['input'].str.startswith(tuple(function_selectors), na=False) &
        (vault_transactions['to_address'].str.lower() == target_address.lower())
    )

    return vault_transactions[mask]['tx_hash'].unique().tolist()


def create_transaction_network(
    transaction_data: pd.DataFrame,
    from_col: str = 'from_address',
    to_col: str = 'to_address',
    weight_col: str = 'value',
    vault_address: str = None
) -> nx.DiGraph:
    """Build a weighted directed graph from a DataFrame of token transfers."""
    G = nx.DiGraph()

    edge_data = {}

    for _, row in transaction_data.iterrows():
        from_addr = row[from_col]
        to_addr = row[to_col]

        if from_addr == to_addr:
            continue

        edge = (from_addr, to_addr)

        if edge not in edge_data:
            edge_data[edge] = {'weight': 0, 'count': 0}

        if weight_col and weight_col in row:
            edge_data[edge]['weight'] += row[weight_col]
        edge_data[edge]['count'] += 1

    for (from_addr, to_addr), data in edge_data.items():
        G.add_edge(from_addr, to_addr, **data)

    return G


def get_hierarchical_layout(G: nx.DiGraph, root: str = None) -> Dict:
    """
    Compute a hierarchical layout with the root node at the top.

    Tries pygraphviz first; falls back to a custom BFS-based layout if
    pygraphviz is not installed.
    """
    try:
        from networkx.drawing.nx_agraph import graphviz_layout
        return graphviz_layout(G, prog='dot', root=root)
    except (ImportError, Exception):
        pass

    if root is None or root not in G.nodes():
        root = max(G.nodes(), key=lambda n: G.out_degree(n))

    levels = {root: 0}
    queue = [root]
    visited = {root}

    while queue:
        node = queue.pop(0)
        current_level = levels[node]

        for successor in G.successors(node):
            if successor not in visited:
                visited.add(successor)
                levels[successor] = current_level + 1
                queue.append(successor)

        for predecessor in G.predecessors(node):
            if predecessor not in visited:
                visited.add(predecessor)
                levels[predecessor] = current_level - 1
                queue.append(predecessor)

    for node in G.nodes():
        if node not in levels:
            levels[node] = 0

    level_nodes = {}
    for node, level in levels.items():
        level_nodes.setdefault(level, []).append(node)

    pos = {}
    y_spacing = 1.0
    x_spacing = 1.5

    for level, nodes in level_nodes.items():
        y = -level * y_spacing  # negative so root appears at top
        for i, node in enumerate(nodes):
            x = (i - (len(nodes) - 1) / 2) * x_spacing
            pos[node] = (x, y)

    return pos


def visualize_network(
    G: nx.DiGraph,
    vault_address: str,
    layout: str = 'hierarchical_dot',
    root: str = None,
    title: str = "Token Transfer Network",
    figsize: tuple = (12, 10),
    node_size: int = 300,
    show_node_labels: bool = True,
    show_edge_labels: bool = True,
    node_label_font_size: int = 8,
    edge_label_font_size: int = 8,
    colormap: str = 'viridis',
    color_eoas: bool = False,
    color_contracts: bool = False,
    contract_addresses: Set[str] = None,
    scale_edge_width: bool = True,
    min_edge_width: float = 0.5,
    max_edge_width: float = 5.0,
    address_labels: Dict[str, str] = None,
    label_y_offset_fraction: float = 0.03,
):
    """
    Visualize a token transfer network.

    The vault node is drawn in red. Other nodes are colored by weighted in-degree
    using the specified colormap. Edge widths are optionally scaled by transfer volume.

    Returns the matplotlib Figure object.
    """
    if G.number_of_nodes() == 0:
        print("Error: Graph has no nodes")
        return None

    if layout == 'hierarchical_dot':
        pos = get_hierarchical_layout(G, root=root)
    elif layout == 'spring':
        pos = nx.spring_layout(G, seed=42)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)

    fig, ax = plt.subplots(figsize=figsize)

    vault_address_lower = vault_address.lower() if vault_address else None
    nodes = list(G.nodes())

    vault_nodes, eoa_nodes, contract_nodes, regular_nodes = [], [], [], []

    for node in nodes:
        node_lower = node.lower()
        if vault_address_lower and node_lower == vault_address_lower:
            vault_nodes.append(node)
        elif color_eoas or color_contracts:
            if contract_addresses:
                is_contract = node_lower in {addr.lower() for addr in contract_addresses}
                if color_contracts and is_contract:
                    contract_nodes.append(node)
                elif color_eoas and not is_contract:
                    eoa_nodes.append(node)
                else:
                    regular_nodes.append(node)
            else:
                regular_nodes.append(node)
        else:
            regular_nodes.append(node)

    if regular_nodes:
        node_values = dict(G.in_degree(weight='weight'))
        values = [node_values.get(node, 0) for node in regular_nodes]
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=regular_nodes,
            node_color=values,
            node_size=node_size,
            cmap=plt.cm.get_cmap(colormap),
            ax=ax
        )

    if vault_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=vault_nodes, node_color='red', node_size=node_size, ax=ax)

    if eoa_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=eoa_nodes, node_color='pink', node_size=node_size, ax=ax)

    if contract_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=contract_nodes, node_color='lightblue', node_size=node_size, ax=ax)

    if scale_edge_width and G.number_of_edges() > 0:
        weights = [data.get('weight', 1) for _, _, data in G.edges(data=True)]
        max_weight = max(weights) if weights else 1
        min_weight = min(weights) if weights else 1

        if max_weight > min_weight:
            edge_widths = [
                min_edge_width + (max_edge_width - min_edge_width) *
                ((data.get('weight', 1) - min_weight) / (max_weight - min_weight))
                for _, _, data in G.edges(data=True)
            ]
        else:
            edge_widths = [min_edge_width] * len(weights)
    else:
        edge_widths = 1.0

    nx.draw_networkx_edges(
        G, pos,
        edge_color='gray',
        width=edge_widths,
        arrows=True,
        arrowsize=10,
        ax=ax,
        alpha=0.5
    )

    if show_node_labels:
        if address_labels:
            labels = {
                node: address_labels.get(node.lower(), f"{node[:6]}...{node[-4:]}")
                for node in G.nodes()
            }
        else:
            labels = {node: f"{node[:6]}...{node[-4:]}" for node in G.nodes()}

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
            font_size=node_label_font_size,
            ax=ax,
            verticalalignment='top',
        )

    if show_edge_labels:
        edge_labels = {}
        for u, v, data in G.edges(data=True):
            weight = data.get('weight', 0)
            if weight >= 1_000_000:
                edge_labels[(u, v)] = f"${weight/1e6:.1f}M"
            elif weight >= 1_000:
                edge_labels[(u, v)] = f"${weight/1e3:.1f}K"
            elif weight >= 1:
                edge_labels[(u, v)] = f"${weight:,.0f}"
            else:
                edge_labels[(u, v)] = f"${weight:.2f}"

        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=edge_label_font_size, ax=ax)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()

    return fig


def convert_steth_to_usd(
    steth_transfers: pd.DataFrame,
    exchange_rates_dir: str = None
) -> pd.DataFrame:
    """
    Convert stETH transfer values to USD.

    stETH tracks ETH very closely, so ETH/USD rates are used for conversion.
    Falls back to using raw values if the exchange rate file is not found.
    """
    if exchange_rates_dir is None:
        exchange_rates_dir = Path(__file__).parent.parent.parent / 'data' / 'exchange_rates'

    eth_usd_path = Path(exchange_rates_dir) / 'eth_usd.csv'

    if not eth_usd_path.exists():
        print(f"Warning: Exchange rate file not found at {eth_usd_path}")
        steth_transfers = steth_transfers.copy()
        steth_transfers['value_usd'] = steth_transfers['value']
        return steth_transfers

    exchange_df = pd.read_csv(eth_usd_path)
    exchange_df['date'] = pd.to_datetime(exchange_df['snapped_at']).dt.date
    exchange_df = exchange_df.set_index('date')

    steth_transfers = steth_transfers.copy()
    steth_transfers['datetime'] = pd.to_datetime(steth_transfers['datetime'])

    usd_values = []
    for _, row in steth_transfers.iterrows():
        date = row['datetime'].date()
        if date in exchange_df.index:
            eth_price = exchange_df.loc[date, 'price']
        else:
            available_dates = [d for d in exchange_df.index if d <= date]
            eth_price = exchange_df.loc[available_dates[-1], 'price'] if available_dates else exchange_df.iloc[0]['price']
        usd_values.append(row['value'] * eth_price)

    steth_transfers['value_usd'] = usd_values

    return steth_transfers


def plot_transaction_network(
    transfers_df: pd.DataFrame,
    transaction_hash: str,
    token_name: str,
    vault_address: str,
    title: str = None,
    figsize: tuple = (12, 10),
    show_node_labels: bool = True,
    show_edge_labels: bool = True,
    address_labels_path: str = None,
    convert_to_usd: bool = True,
    exchange_rates_dir: str = None,
    exchange_rate_file: str = None
):
    """
    Plot the token transfer network for a single transaction.

    Returns the matplotlib Figure object, or None if the transaction is not found.
    """
    address_labels = load_address_labels(address_labels_path)

    filtered_transfers = transfers_df[transfers_df['tx_hash'] == transaction_hash].copy()

    if len(filtered_transfers) == 0:
        return None

    weight_col = 'value'

    if convert_to_usd and exchange_rate_file:
        if exchange_rates_dir is None:
            exchange_rates_dir = Path(__file__).parent.parent.parent / 'data' / 'exchange_rates'

        exchange_rate_path = Path(exchange_rates_dir) / exchange_rate_file

        if exchange_rate_path.exists():
            exchange_df = pd.read_csv(exchange_rate_path)
            exchange_df['date'] = pd.to_datetime(exchange_df['snapped_at']).dt.date
            exchange_df = exchange_df.set_index('date')

            filtered_transfers['datetime'] = pd.to_datetime(filtered_transfers['datetime'])

            usd_values = []
            for _, row in filtered_transfers.iterrows():
                date = row['datetime'].date()
                if date in exchange_df.index:
                    price = exchange_df.loc[date, 'price']
                else:
                    available_dates = [d for d in exchange_df.index if d <= date]
                    price = exchange_df.loc[available_dates[-1], 'price'] if available_dates else exchange_df.iloc[0]['price']
                usd_values.append(row['value'] * price)

            filtered_transfers['value_usd'] = usd_values
            weight_col = 'value_usd'
        else:
            print(f"Warning: Exchange rate file not found at {exchange_rate_path}, using native values")

    G = create_transaction_network(
        filtered_transfers,
        from_col='from_address',
        to_col='to_address',
        weight_col=weight_col,
        vault_address=vault_address
    )

    if title is None:
        unit = 'USD' if weight_col == 'value_usd' else token_name
        title = f'{token_name} Transfers ({unit}) - Tx {transaction_hash[:10]}...'

    return visualize_network(
        G,
        vault_address=vault_address,
        layout='hierarchical_dot',
        root=vault_address,
        title=title,
        figsize=figsize,
        node_size=300,
        show_node_labels=show_node_labels,
        show_edge_labels=show_edge_labels,
        node_label_font_size=8,
        edge_label_font_size=8,
        colormap='viridis',
        scale_edge_width=True,
        min_edge_width=0.5,
        max_edge_width=5.0,
        address_labels=address_labels
    )


def plot_steth_transfer_to_strategy_network(
    steth_transfers: pd.DataFrame,
    vault_transactions: pd.DataFrame,
    vault_address: str,
    title: str = None,
    figsize: tuple = (12, 10),
    show_node_labels: bool = True,
    show_edge_labels: bool = True,
    address_labels_path: str = None,
    convert_to_usd: bool = True,
    exchange_rates_dir: str = None
):
    """
    Plot the aggregated stETH transfer network for transfer_to_strategy operations.

    All transfer_to_strategy transactions are combined into a single graph with
    edges weighted by total volume. Returns the matplotlib Figure object.
    """
    address_labels = load_address_labels(address_labels_path)

    tx_hashes = get_transaction_hashes_by_function(
        vault_transactions,
        vault_address,
        ['0xba8bfa2a']
    )

    filtered_transfers = steth_transfers[steth_transfers['tx_hash'].isin(tx_hashes)].copy()

    if len(filtered_transfers) == 0:
        print("No transfers found for transfer_to_strategy function")
        return None

    if convert_to_usd:
        filtered_transfers = convert_steth_to_usd(filtered_transfers, exchange_rates_dir)
        weight_col = 'value_usd'
    else:
        weight_col = 'value'

    G = create_transaction_network(
        filtered_transfers,
        from_col='from_address',
        to_col='to_address',
        weight_col=weight_col,
        vault_address=vault_address
    )

    if title is None:
        unit = 'USD' if convert_to_usd else 'stETH'
        title = f'Cian Vault - stETH Transfers (transfer_to_strategy) in {unit}'

    return visualize_network(
        G,
        vault_address=vault_address,
        layout='hierarchical_dot',
        root=vault_address,
        title=title,
        figsize=figsize,
        node_size=300,
        show_node_labels=show_node_labels,
        show_edge_labels=show_edge_labels,
        node_label_font_size=8,
        edge_label_font_size=8,
        colormap='viridis',
        scale_edge_width=True,
        min_edge_width=0.5,
        max_edge_width=5.0,
        address_labels=address_labels
    )
