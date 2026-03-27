"""Yearn vault analysis functions."""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, List, Optional


def _count_token_transfer_volumes_over_time(
    transactions_df: pd.DataFrame,
    transfers_df: pd.DataFrame,
    function_signatures: Dict[str, List[str]],
    vault_address: str,
    time_column: str = 'datetime',
    freq: str = 'D',
    direction_map: Dict[str, str] = None
) -> pd.DataFrame:
    """
    Calculate token transfer volumes by transaction type over time.

    Helper used internally by the higher-level flow calculation functions.
    """
    if direction_map is None:
        direction_map = {
            'deposit': 'to_vault',
            'redeem': 'from_vault',
            'strategy_investment': 'both'
        }

    vault_address = vault_address.lower()

    tx_df_copy = transactions_df.copy()
    tx_df_copy['function_signature'] = tx_df_copy['input'].astype(str).str[:10]

    transfers_copy = transfers_df.copy()
    transfers_copy['from_normalized'] = transfers_copy['from_address'].str.lower()
    transfers_copy['to_normalized'] = transfers_copy['to_address'].str.lower()
    transfers_copy[time_column] = pd.to_datetime(transfers_copy[time_column])

    time_series_data = {}

    for tx_type, signatures in function_signatures.items():
        matching_txs = tx_df_copy[
            (tx_df_copy['function_signature'].isin(signatures)) &
            (tx_df_copy['to_address'].str.lower() == vault_address)
        ]
        tx_hashes = set(matching_txs['tx_hash'])

        type_transfers = transfers_copy[transfers_copy['tx_hash'].isin(tx_hashes)].copy()

        if len(type_transfers) == 0:
            continue

        type_transfers = type_transfers.set_index(time_column)

        direction = direction_map.get(tx_type, 'both')

        if direction == 'to_vault':
            to_vault_transfers = type_transfers[type_transfers['to_normalized'] == vault_address]
            volumes = to_vault_transfers['value'].resample(freq).sum()
            time_series_data[f"{tx_type}_to_vault"] = volumes

        elif direction == 'from_vault':
            from_vault_transfers = type_transfers[type_transfers['from_normalized'] == vault_address]
            volumes = from_vault_transfers['value'].resample(freq).sum()
            time_series_data[f"{tx_type}_from_vault"] = volumes

        else:  # 'both'
            to_vault_transfers = type_transfers[type_transfers['to_normalized'] == vault_address]
            from_vault_transfers = type_transfers[type_transfers['from_normalized'] == vault_address]

            time_series_data[f"{tx_type}_to_vault"] = to_vault_transfers['value'].resample(freq).sum()
            time_series_data[f"{tx_type}_from_vault"] = from_vault_transfers['value'].resample(freq).sum()

    result_df = pd.DataFrame(time_series_data).fillna(0)

    if not result_df.empty:
        result_df = result_df.sort_index()

    return result_df


def calculate_deposit_redeem_statistics(
    transactions_df: pd.DataFrame,
    transfers_df: pd.DataFrame,
    function_signatures: Dict[str, List[str]],
    vault_address: str,
    token_symbol: str = 'USDC'
) -> Dict[str, float]:
    """
    Calculate summary statistics for deposits and redeems.

    Counts only transfers directly to/from the vault address in transactions
    matching the deposit/redeem function signatures.
    """
    vault_address = vault_address.lower()

    tx_df = transactions_df.copy()
    tx_df['function_signature'] = tx_df['input'].astype(str).str[:10]

    transfers = transfers_df.copy()
    transfers['from_normalized'] = transfers['from_address'].str.lower()
    transfers['to_normalized'] = transfers['to_address'].str.lower()

    deposit_sigs = function_signatures.get('deposit', [])
    deposit_txs = tx_df[
        (tx_df['function_signature'].isin(deposit_sigs)) &
        (tx_df['to_address'].str.lower() == vault_address)
    ]
    deposit_tx_hashes = set(deposit_txs['tx_hash'])

    redeem_sigs = function_signatures.get('redeem', [])
    redeem_txs = tx_df[
        (tx_df['function_signature'].isin(redeem_sigs)) &
        (tx_df['to_address'].str.lower() == vault_address)
    ]
    redeem_tx_hashes = set(redeem_txs['tx_hash'])

    deposit_transfers = transfers[
        (transfers['tx_hash'].isin(deposit_tx_hashes)) &
        (transfers['to_normalized'] == vault_address)
    ]

    redeem_transfers = transfers[
        (transfers['tx_hash'].isin(redeem_tx_hashes)) &
        (transfers['from_normalized'] == vault_address)
    ]

    stats = {}

    if len(deposit_transfers) > 0:
        stats['total_deposits'] = deposit_transfers['value'].sum()
        stats['mean_deposit'] = deposit_transfers['value'].mean()
        stats['median_deposit'] = deposit_transfers['value'].median()
        stats['max_deposit'] = deposit_transfers['value'].max()
        stats['deposit_count'] = len(deposit_transfers)
    else:
        stats.update({
            'total_deposits': 0, 'mean_deposit': 0, 'median_deposit': 0,
            'max_deposit': 0, 'deposit_count': 0
        })

    if len(redeem_transfers) > 0:
        stats['total_redeems'] = redeem_transfers['value'].sum()
        stats['mean_redeem'] = redeem_transfers['value'].mean()
        stats['median_redeem'] = redeem_transfers['value'].median()
        stats['max_redeem'] = redeem_transfers['value'].max()
        stats['redeem_count'] = len(redeem_transfers)
    else:
        stats.update({
            'total_redeems': 0, 'mean_redeem': 0, 'median_redeem': 0,
            'max_redeem': 0, 'redeem_count': 0
        })

    stats['net_flow'] = stats['total_deposits'] - stats['total_redeems']
    stats['token_symbol'] = token_symbol

    return stats


def print_deposit_redeem_statistics(stats: Dict[str, float]):
    """Print deposit and redeem statistics in a formatted table."""
    token = stats.get('token_symbol', 'Token')

    print("\n" + "="*70)
    print("DEPOSIT & REDEEM SUMMARY STATISTICS")
    print("="*70)
    print(f"\nTotal Deposits:        ${stats['total_deposits']:>20,.2f} {token}")
    print(f"Total Redeems:         ${stats['total_redeems']:>20,.2f} {token}")
    print(f"Net Flow:              ${stats['net_flow']:>20,.2f} {token}")
    print("-"*70)

    print(f"\nDeposit Statistics:")
    print(f"  Mean:                ${stats['mean_deposit']:>20,.2f} {token}")
    print(f"  Median:              ${stats['median_deposit']:>20,.2f} {token}")
    print(f"  Max:                 ${stats['max_deposit']:>20,.2f} {token}")
    print(f"  Count:               {stats['deposit_count']:>21,} transfers")

    print(f"\nRedeem Statistics:")
    print(f"  Mean:                ${stats['mean_redeem']:>20,.2f} {token}")
    print(f"  Median:              ${stats['median_redeem']:>20,.2f} {token}")
    print(f"  Max:                 ${stats['max_redeem']:>20,.2f} {token}")
    print(f"  Count:               {stats['redeem_count']:>21,} transfers")

    print("="*70 + "\n")


def calculate_transaction_depth_statistics(
    transactions_df: pd.DataFrame,
    transfers_df: pd.DataFrame,
    function_signatures: Dict[str, List[str]],
    vault_address: str,
) -> Dict[str, Dict[str, float]]:
    """
    Calculate statistics on the number of token transfers (internal transactions)
    per transaction hash, broken down by function type.

    For each transaction type, counts how many rows in transfers_df share the same
    tx_hash, then summarises the distribution of those counts.
    """
    vault_address = vault_address.lower()

    tx_df = transactions_df.copy()
    tx_df['function_signature'] = tx_df['input'].astype(str).str[:10]

    stats = {}

    for tx_type, signatures in function_signatures.items():
        type_txs = tx_df[
            (tx_df['function_signature'].isin(signatures)) &
            (tx_df['to_address'].str.lower() == vault_address)
        ]
        tx_hashes = set(type_txs['tx_hash'])

        type_transfers = transfers_df[transfers_df['tx_hash'].isin(tx_hashes)]
        depth_series = type_transfers.groupby('tx_hash').size()

        if len(depth_series) > 0:
            stats[tx_type] = {
                'tx_count': len(depth_series),
                'mean': depth_series.mean(),
                'median': depth_series.median(),
                'min': int(depth_series.min()),
                'max': int(depth_series.max()),
            }
        else:
            stats[tx_type] = {
                'tx_count': 0, 'mean': 0.0, 'median': 0.0, 'min': 0, 'max': 0
            }

    return stats


def print_transaction_depth_statistics(stats: Dict[str, Dict[str, float]]):
    """Print transaction depth statistics in a formatted table."""
    print("\n" + "="*70)
    print("TRANSACTION DEPTH STATISTICS (token transfers per tx hash)")
    print("="*70)

    for tx_type, s in stats.items():
        print(f"\n{tx_type.upper().replace('_', ' ')}")
        print(f"  Transactions:  {s['tx_count']:>10,}")
        print(f"  Mean:          {s['mean']:>10.2f}")
        print(f"  Median:        {s['median']:>10.2f}")
        print(f"  Min:           {s['min']:>10,}")
        print(f"  Max:           {s['max']:>10,}")

    print("\n" + "="*70 + "\n")


def export_deposit_redeem_transfers(
    transactions_df: pd.DataFrame,
    transfers_df: pd.DataFrame,
    function_signatures: Dict[str, List[str]],
    vault_address: str,
    output_dir: str,
    token_symbol: str = 'USDC'
) -> tuple:
    """
    Extract and export individual deposit and redeem transfers to CSV files.

    Returns a tuple of (deposit_transfers, redeem_transfers) DataFrames.
    """
    vault_address = vault_address.lower()

    tx_df = transactions_df.copy()
    tx_df['function_signature'] = tx_df['input'].astype(str).str[:10]

    transfers = transfers_df.copy()
    transfers['from_normalized'] = transfers['from_address'].str.lower()
    transfers['to_normalized'] = transfers['to_address'].str.lower()

    deposit_sigs = function_signatures.get('deposit', [])
    deposit_txs = tx_df[
        (tx_df['function_signature'].isin(deposit_sigs)) &
        (tx_df['to_address'].str.lower() == vault_address)
    ]
    deposit_tx_hashes = set(deposit_txs['tx_hash'])

    redeem_sigs = function_signatures.get('redeem', [])
    redeem_txs = tx_df[
        (tx_df['function_signature'].isin(redeem_sigs)) &
        (tx_df['to_address'].str.lower() == vault_address)
    ]
    redeem_tx_hashes = set(redeem_txs['tx_hash'])

    deposit_transfers = transfers[
        (transfers['tx_hash'].isin(deposit_tx_hashes)) &
        (transfers['to_normalized'] == vault_address)
    ].copy()

    redeem_transfers = transfers[
        (transfers['tx_hash'].isin(redeem_tx_hashes)) &
        (transfers['from_normalized'] == vault_address)
    ].copy()

    essential_columns = ['datetime', 'tx_hash', 'from_address', 'to_address', 'value', 'token_address']

    deposit_export = deposit_transfers[essential_columns].copy()
    redeem_export = redeem_transfers[essential_columns].copy()

    deposit_path = f'{output_dir}/yearn_deposits.csv'
    redeem_path = f'{output_dir}/yearn_withdrawals.csv'

    deposit_export.to_csv(deposit_path, index=False)
    redeem_export.to_csv(redeem_path, index=False)

    print("="*70)
    print("EXPORTED DEPOSIT AND REDEEM DATA")
    print("="*70)
    print(f"Deposits exported:     {len(deposit_export):>6,} records -> {deposit_path}")
    print(f"Redeems exported:      {len(redeem_export):>6,} records -> {redeem_path}")
    print("="*70)

    return deposit_transfers, redeem_transfers


def calculate_cumulative_token_flow(
    transactions_df: pd.DataFrame,
    transfers_df: pd.DataFrame,
    function_signatures: Dict[str, List[str]],
    vault_address: str,
    time_column: str = 'datetime',
    freq: str = 'D',
    deposit_types: List[str] = None,
    redeem_types: List[str] = None
) -> pd.DataFrame:
    """
    Calculate cumulative token flow (deposits minus redeems) over time.

    Returns a DataFrame with columns: Deposits, Redeems, Net_Flow, Cumulative_Flow.
    """
    if deposit_types is None:
        deposit_types = ['deposit']
    if redeem_types is None:
        redeem_types = ['redeem']

    transfers_df = transfers_df.copy()
    transfers_df[time_column] = pd.to_datetime(transfers_df[time_column])

    transactions_df = transactions_df.copy()
    if 'func_sig' not in transactions_df.columns:
        transactions_df['func_sig'] = transactions_df['input'].str[:10]

    vault_address = vault_address.lower()
    transfers_df['from_address'] = transfers_df['from_address'].str.lower()
    transfers_df['to_address'] = transfers_df['to_address'].str.lower()

    transactions_df['to_address'] = transactions_df['to_address'].str.lower()
    transactions_df['func_sig'] = transactions_df['func_sig'].str.lower()

    tx_hashes_by_type: Dict[str, set] = {}
    for tx_type, sigs in function_signatures.items():
        sigs_lower = [s.lower() for s in sigs]
        matching = transactions_df[
            (transactions_df['func_sig'].isin(sigs_lower)) &
            (transactions_df['to_address'] == vault_address)
        ]
        tx_hashes_by_type[tx_type] = set(matching['tx_hash'])

    deposit_tx_hashes = set().union(*(tx_hashes_by_type.get(t, set()) for t in deposit_types))
    deposits = transfers_df[
        (transfers_df['tx_hash'].isin(deposit_tx_hashes)) &
        (transfers_df['to_address'] == vault_address)
    ].copy()

    redeem_tx_hashes = set().union(*(tx_hashes_by_type.get(t, set()) for t in redeem_types))
    redeems = transfers_df[
        (transfers_df['tx_hash'].isin(redeem_tx_hashes)) &
        (transfers_df['from_address'] == vault_address)
    ].copy()

    deposits[time_column] = pd.to_datetime(deposits[time_column])
    redeems[time_column] = pd.to_datetime(redeems[time_column])

    deposits_grouped = deposits.groupby(pd.Grouper(key=time_column, freq=freq))['value'].sum()
    redeems_grouped = redeems.groupby(pd.Grouper(key=time_column, freq=freq))['value'].sum()

    result_df = pd.DataFrame({
        'Deposits': deposits_grouped,
        'Redeems': redeems_grouped
    }).fillna(0)

    result_df['Net_Flow'] = result_df['Deposits'] - result_df['Redeems']
    result_df['Cumulative_Flow'] = result_df['Net_Flow'].cumsum()

    return result_df


def plot_net_deposits_vs_withdrawals(
    transactions_df: pd.DataFrame,
    transfers_df: pd.DataFrame,
    function_signatures: Dict[str, List[str]],
    vault_address: str,
    time_column: str = 'datetime',
    freq: str = 'D',
    title: str = None,
    figsize: tuple = (12, 6),
    token_symbol: str = 'USDC',
    start_date: str = None,
    end_date: str = None
):
    """
    Plot cumulative net deposits vs. withdrawals over time.

    Returns the matplotlib Figure object.
    """
    flow_data = calculate_cumulative_token_flow(
        transactions_df, transfers_df, function_signatures, vault_address,
        time_column, freq
    )

    if len(flow_data) == 0:
        raise ValueError("No data found for the specified parameters")

    if start_date is not None:
        flow_data = flow_data[flow_data.index >= pd.to_datetime(start_date)]
    if end_date is not None:
        flow_data = flow_data[flow_data.index <= pd.to_datetime(end_date)]

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.plot(flow_data.index, flow_data['Cumulative_Flow'],
            color='#1f77b4', linewidth=2.5, label='Net Deposits vs Withdrawals',
            marker='o', markersize=3, alpha=0.9)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_ylabel(f'{token_symbol}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

    ax.set_title(title or 'Net Deposits vs Withdrawals', fontsize=14, fontweight='bold')

    plt.tight_layout()

    print("\n" + "="*70)
    print(f"NET DEPOSITS VS WITHDRAWALS SUMMARY ({token_symbol})")
    print("="*70)
    print(f"\nTotal Deposits:              {flow_data['Deposits'].sum():>15,.2f} {token_symbol}")
    print(f"Total Withdrawals:           {flow_data['Redeems'].sum():>15,.2f} {token_symbol}")
    print(f"Net Flow:                    {flow_data['Net_Flow'].sum():>15,.2f} {token_symbol}")
    print(f"Final Cumulative Balance:    {flow_data['Cumulative_Flow'].iloc[-1]:>15,.2f} {token_symbol}")
    print("="*70)

    return fig


def export_cumulative_flow_to_csv(
    transactions_df: pd.DataFrame,
    transfers_df: pd.DataFrame,
    function_signatures: Dict[str, List[str]],
    vault_address: str,
    output_path: str,
    time_column: str = 'datetime',
    freq: str = 'D',
    token_symbol: str = 'USDC'
):
    """Calculate and export cumulative flow data to CSV."""
    flow_data = calculate_cumulative_token_flow(
        transactions_df, transfers_df, function_signatures, vault_address,
        time_column, freq
    )

    export_data = pd.DataFrame({
        'datetime': flow_data.index,
        f'{token_symbol}': flow_data['Cumulative_Flow']
    })

    export_data.to_csv(output_path, index=False)
    print(f"Data exported to: {output_path}")


def calculate_strategy_investment_flow(
    transactions_df: pd.DataFrame,
    transfers_df: pd.DataFrame,
    function_signatures: Dict[str, List[str]],
    vault_address: str,
    time_column: str = 'datetime',
    freq: str = 'D'
) -> pd.DataFrame:
    """
    Calculate net flow to strategies over time.

    Tracks tokens entering and leaving the vault during update_debt operations,
    plus tokens entering during redeem operations (strategy withdrawals triggered
    by user redemptions).

    Returns a DataFrame with columns:
        Strategy_Inflows, Strategy_Outflows, Redeem_Inflows,
        Net_Total_Flow, Cumulative_Net_Total_Flow
    """
    combined_signatures = {
        'strategy_investment': function_signatures['strategy_investment'],
        'redeem': function_signatures['redeem']
    }

    # For strategy investment, track both directions; for redeem, track only inflows
    direction_map = {
        'strategy_investment': 'both',
        'redeem': 'to_vault'
    }

    transfer_volumes = _count_token_transfer_volumes_over_time(
        transactions_df, transfers_df, combined_signatures, vault_address,
        time_column, freq, direction_map=direction_map
    )

    if len(transfer_volumes) == 0:
        return pd.DataFrame()

    result_df = pd.DataFrame(index=transfer_volumes.index)

    strategy_inflows = transfer_volumes.get('strategy_investment_to_vault', pd.Series(0, index=transfer_volumes.index))
    strategy_outflows = transfer_volumes.get('strategy_investment_from_vault', pd.Series(0, index=transfer_volumes.index))
    redeem_inflows = transfer_volumes.get('redeem_to_vault', pd.Series(0, index=transfer_volumes.index))

    net_total_flow = (strategy_inflows + redeem_inflows) - strategy_outflows

    result_df['Strategy_Inflows'] = strategy_inflows
    result_df['Strategy_Outflows'] = strategy_outflows
    result_df['Redeem_Inflows'] = redeem_inflows
    result_df['Net_Total_Flow'] = net_total_flow
    result_df['Cumulative_Net_Total_Flow'] = net_total_flow.cumsum()

    return result_df


def plot_strategy_investment_flow(
    transactions_df: pd.DataFrame,
    transfers_df: pd.DataFrame,
    function_signatures: Dict[str, List[str]],
    vault_address: str,
    time_column: str = 'datetime',
    freq: str = 'D',
    title: str = None,
    figsize: tuple = (12, 6),
    token_symbol: str = 'USDC',
    start_date: str = None,
    end_date: str = None
):
    """
    Plot cumulative net flow to strategies over time.

    Positive values indicate more tokens are currently deployed to strategies
    than have been withdrawn. Returns the matplotlib Figure object.
    """
    flow_data = calculate_strategy_investment_flow(
        transactions_df, transfers_df, function_signatures, vault_address,
        time_column, freq
    )

    if len(flow_data) == 0:
        raise ValueError("No strategy investment data found for the specified parameters")

    if start_date is not None:
        flow_data = flow_data[flow_data.index >= pd.to_datetime(start_date)]
    if end_date is not None:
        flow_data = flow_data[flow_data.index <= pd.to_datetime(end_date)]

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Negate so that positive values = tokens deployed to strategies
    ax.plot(flow_data.index, -flow_data['Cumulative_Net_Total_Flow'],
            color='#9B59B6', linewidth=2.5, label='Net Flow to Strategies',
            marker='o', markersize=3, alpha=0.9)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_ylabel(f'{token_symbol}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

    ax.set_title(title or 'Net Flow to Strategies', fontsize=14, fontweight='bold')

    plt.tight_layout()

    print("\n" + "="*70)
    print(f"STRATEGY INVESTMENT FLOW SUMMARY ({token_symbol})")
    print("="*70)
    print(f"\nTotal Strategy Inflows:      {flow_data['Strategy_Inflows'].sum():>15,.2f} {token_symbol}")
    print(f"Total Strategy Outflows:     {flow_data['Strategy_Outflows'].sum():>15,.2f} {token_symbol}")
    print(f"Total Redeem Inflows:        {flow_data['Redeem_Inflows'].sum():>15,.2f} {token_symbol}")
    print(f"Net Cumulative Flow:         {-flow_data['Cumulative_Net_Total_Flow'].iloc[-1]:>15,.2f} {token_symbol}")
    print("="*70)

    return fig


def calculate_strategy_investment_protocol_flows(
    transactions_df: pd.DataFrame,
    transfers_df: pd.DataFrame,
    function_signatures: Dict[str, List[str]],
    vault_address: str,
    protocol_addresses: Dict[str, str],
    time_column: str = 'datetime',
    freq: str = 'D'
) -> pd.DataFrame:
    """
    Calculate net token flows to specific protocol addresses during strategy operations.

    Covers both update_debt and redeem transactions. Multiple addresses can map
    to the same protocol name for aggregated analysis.

    Returns a DataFrame with one column per protocol showing cumulative net flows.
    """
    combined_signatures = {
        'strategy_investment': function_signatures['strategy_investment'],
        'redeem': function_signatures['redeem']
    }

    vault_address = vault_address.lower()
    protocol_map = {addr.lower(): name for addr, name in protocol_addresses.items()}

    tx_df_copy = transactions_df.copy()
    tx_df_copy['function_signature'] = tx_df_copy['input'].astype(str).str[:10]

    all_signatures = combined_signatures['strategy_investment'] + combined_signatures['redeem']
    matching_txs = tx_df_copy[
        (tx_df_copy['function_signature'].isin(all_signatures)) &
        (tx_df_copy['to_address'].str.lower() == vault_address)
    ]
    tx_hashes = set(matching_txs['tx_hash'])

    transfers_copy = transfers_df.copy()
    transfers_copy['from_normalized'] = transfers_copy['from_address'].str.lower()
    transfers_copy['to_normalized'] = transfers_copy['to_address'].str.lower()
    transfers_copy[time_column] = pd.to_datetime(transfers_copy[time_column])

    type_transfers = transfers_copy[transfers_copy['tx_hash'].isin(tx_hashes)].copy()

    if len(type_transfers) == 0:
        return pd.DataFrame()

    type_transfers = type_transfers.set_index(time_column)

    protocol_flows = {}

    for protocol_addr, protocol_name in protocol_map.items():
        inflows = type_transfers[type_transfers['to_normalized'] == protocol_addr]
        inflow_volumes = inflows['value'].resample(freq).sum()

        outflows = type_transfers[type_transfers['from_normalized'] == protocol_addr]
        outflow_volumes = outflows['value'].resample(freq).sum()

        net_flow = inflow_volumes.subtract(outflow_volumes, fill_value=0)

        # Aggregate in case multiple addresses map to the same protocol
        if protocol_name in protocol_flows:
            protocol_flows[protocol_name] = protocol_flows[protocol_name].add(net_flow, fill_value=0)
        else:
            protocol_flows[protocol_name] = net_flow

    result_df = pd.DataFrame(protocol_flows).fillna(0)

    if not result_df.empty:
        result_df = result_df.sort_index()

    return result_df


def plot_strategy_investment_protocol_flows(
    transactions_df: pd.DataFrame,
    transfers_df: pd.DataFrame,
    function_signatures: Dict[str, List[str]],
    vault_address: str,
    protocol_addresses: Dict[str, str],
    time_column: str = 'datetime',
    freq: str = 'D',
    title: str = None,
    figsize: tuple = (12, 6),
    token_symbol: str = 'USDC',
    start_date: str = None,
    end_date: str = None
):
    """
    Plot cumulative net token flows to each protocol during strategy operations.

    Positive values indicate net tokens deployed to a protocol (vault → protocol),
    negative values indicate net withdrawals (protocol → vault).
    Returns the matplotlib Figure object.
    """
    protocol_flows = calculate_strategy_investment_protocol_flows(
        transactions_df, transfers_df, function_signatures, vault_address,
        protocol_addresses, time_column, freq
    )

    if len(protocol_flows) == 0:
        raise ValueError("No strategy investment protocol flow data found")

    if start_date is not None:
        protocol_flows = protocol_flows[protocol_flows.index >= pd.to_datetime(start_date)]
    if end_date is not None:
        protocol_flows = protocol_flows[protocol_flows.index <= pd.to_datetime(end_date)]

    cumulative_flows = protocol_flows.cumsum()

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    colors = ['#9467bd', '#FFD700', '#e377c2', '#17becf', '#8c564b', '#7f7f7f']

    for i, protocol in enumerate(cumulative_flows.columns):
        color = colors[i % len(colors)]
        ax.plot(cumulative_flows.index, cumulative_flows[protocol],
               label=protocol, linewidth=2.5, marker='o', markersize=3, color=color, alpha=0.9)

    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_ylabel(f'{token_symbol}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

    ax.set_title(title or 'Net Token Flows to Protocols', fontsize=14, fontweight='bold')

    plt.tight_layout()

    print("\n" + "="*70)
    print(f"STRATEGY INVESTMENT PROTOCOL FLOWS SUMMARY ({token_symbol})")
    print("="*70)

    if len(cumulative_flows) > 0:
        final_flows = cumulative_flows.iloc[-1]
        total_net_flow = final_flows.sum()

        print(f"\nFinal Cumulative Net Flows by Protocol:")
        for protocol, amount in final_flows.sort_values(ascending=False).items():
            percentage = (abs(amount) / final_flows.abs().sum() * 100) if final_flows.abs().sum() > 0 else 0
            direction = "↑" if amount >= 0 else "↓"
            print(f"  {direction} {protocol:20s}: {amount:>15,.2f} {token_symbol} ({percentage:>5.1f}%)")

        print(f"\nTotal Net Flow to Protocols: {total_net_flow:>15,.2f} {token_symbol}")
    print("="*70)

    return fig


def export_strategy_investment_flow_to_csv(
    transactions_df: pd.DataFrame,
    transfers_df: pd.DataFrame,
    function_signatures: Dict[str, List[str]],
    vault_address: str,
    output_path: str,
    time_column: str = 'datetime',
    freq: str = 'D',
    token_symbol: str = 'USDC'
):
    """Calculate and export strategy investment flow data to CSV."""
    flow_data = calculate_strategy_investment_flow(
        transactions_df, transfers_df, function_signatures, vault_address,
        time_column, freq
    )

    # Negate so that positive values = tokens currently deployed to strategies
    export_data = pd.DataFrame({
        'datetime': flow_data.index,
        f'Net_Flow_to_Strategies_{token_symbol}': -flow_data['Cumulative_Net_Total_Flow']
    })

    export_data.to_csv(output_path, index=False)
    print(f"Strategy investment flow data exported to: {output_path}")


def export_protocol_flows_to_csv(
    transactions_df: pd.DataFrame,
    transfers_df: pd.DataFrame,
    function_signatures: Dict[str, List[str]],
    vault_address: str,
    protocol_addresses: Dict[str, str],
    output_path: str,
    time_column: str = 'datetime',
    freq: str = 'D'
):
    """Calculate and export protocol flows data to CSV."""
    protocol_flows = calculate_strategy_investment_protocol_flows(
        transactions_df, transfers_df, function_signatures, vault_address,
        protocol_addresses, time_column, freq
    )

    cumulative_flows = protocol_flows.cumsum()
    export_data = cumulative_flows.copy()
    export_data.insert(0, 'datetime', export_data.index)

    export_data.to_csv(output_path, index=False)
    print(f"Protocol flows data exported to: {output_path}")


def _standardize_x_axis_format(ax, data_index, freq='D', padding_percent=0.02):
    """
    Apply consistent x-axis date formatting based on the data's time range.

    Selects tick density to avoid label crowding at all common frequencies.
    """
    if len(data_index) == 0:
        return

    date_range = (data_index.max() - data_index.min()).days

    if freq == 'D':
        if date_range > 180:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator())
        elif date_range > 90:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        elif date_range > 30:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        else:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    elif freq == 'W':
        if date_range > 180:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator())
        elif date_range > 90:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        else:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    elif freq == 'M':
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
    elif freq == 'H':
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))

    ax.tick_params(axis='x', rotation=0)

    padding = pd.Timedelta(days=max(1, date_range * padding_percent))
    ax.set_xlim(data_index.min() - padding, data_index.max() + padding)


def plot_vault_exchange_rate_over_time(
    vault_events_df: pd.DataFrame,
    time_column: str = 'datetime',
    title: str = 'Vault Exchange Rate Over Time',
    figsize: tuple = (12, 6),
    token_symbol: str = 'USDC',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """
    Plot vault exchange rate (assets/shares) from deposit and withdraw events.

    Each point represents one transaction. Rising exchange rate over time indicates
    positive yield generation for vault shareholders.
    Returns the matplotlib Figure object.
    """
    if len(vault_events_df) == 0:
        print("No vault events data available for plotting.")
        return None

    df = vault_events_df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
        df[time_column] = pd.to_datetime(df[time_column])

    if start_date:
        df = df[df[time_column] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df[time_column] <= pd.to_datetime(end_date)]

    # Drop events where shares or assets are zero to avoid division errors
    df = df[(df['shares'] > 0) & (df['assets'] > 0)]

    if len(df) == 0:
        print("No valid exchange rate data available for plotting.")
        return None

    df['exchange_rate'] = df['assets'] / df['shares']
    df = df.sort_values(time_column)

    fig, ax1 = plt.subplots(figsize=figsize)

    ax1.plot(df[time_column], df['exchange_rate'],
             label='Exchange Rate', color='#2E8B57', linewidth=1, marker='o', markersize=2, alpha=0.7)

    ax1.set_xlabel('Date')
    ax1.set_ylabel(f'Exchange Rate ({token_symbol} per Share)', color='#2E8B57')
    ax1.tick_params(axis='y', labelcolor='#2E8B57')
    ax1.grid(True, alpha=0.3)

    _standardize_x_axis_format(ax1, df[time_column], freq='D')

    ax1.legend(loc='upper left', bbox_to_anchor=(0.02, 0.85))
    ax1.set_title(title, fontsize=14, fontweight='bold', pad=20)

    exchange_rate_change = (
        (df['exchange_rate'].iloc[-1] - df['exchange_rate'].iloc[0]) /
        df['exchange_rate'].iloc[0] * 100
    ) if len(df) > 1 else 0

    plt.tight_layout()

    print("\n" + "="*70)
    print("VAULT EXCHANGE RATE ANALYSIS")
    print("="*70)
    print(f"Starting rate: {df['exchange_rate'].iloc[0]:.6f} {token_symbol} per share")
    print(f"Ending rate:   {df['exchange_rate'].iloc[-1]:.6f} {token_symbol} per share")
    print(f"Total change:  {exchange_rate_change:+.2f}%")
    print("="*70)

    return fig


def export_exchange_rate_to_csv(
    vault_events_df: pd.DataFrame,
    output_path: str,
    time_column: str = 'datetime',
    token_symbol: str = 'USDC'
):
    """Calculate and export vault exchange rate data to CSV."""
    if len(vault_events_df) == 0:
        print("No vault events data available for export.")
        return

    df = vault_events_df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
        df[time_column] = pd.to_datetime(df[time_column])

    df = df[(df['shares'] > 0) & (df['assets'] > 0)]

    if len(df) == 0:
        print("No valid exchange rate data available for export.")
        return

    df['exchange_rate'] = df['assets'] / df['shares']
    df = df.sort_values(time_column)

    export_data = pd.DataFrame({
        'datetime': df[time_column],
        f'Exchange_Rate_{token_symbol}_per_Share': df['exchange_rate'],
        'Assets': df['assets'],
        'Shares': df['shares']
    })

    export_data.to_csv(output_path, index=False)
    print(f"Exchange rate data exported to: {output_path}")


def calculate_internal_tx_depth_statistics(
    transactions_df: pd.DataFrame,
    function_signatures: Dict[str, List[str]],
    vault_address: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Calculate statistics on the number of internal transactions (traces) per
    transaction hash, broken down by function type.

    For each transaction type, identifies the matching tx_hashes by function
    signature, then counts all rows in transactions_df sharing each tx_hash.
    """
    tx_df = transactions_df.copy()
    tx_df['function_signature'] = tx_df['input'].astype(str).str[:10]

    all_depth_series = tx_df.groupby('tx_hash').size()

    stats = {}

    for tx_type, signatures in function_signatures.items():
        type_txs = tx_df[tx_df['function_signature'].isin(signatures)]
        tx_hashes = set(type_txs['tx_hash'])

        depth_series = all_depth_series[all_depth_series.index.isin(tx_hashes)]

        if len(depth_series) > 0:
            stats[tx_type] = {
                'tx_count': len(depth_series),
                'mean': depth_series.mean(),
                'median': depth_series.median(),
                'min': int(depth_series.min()),
                'max': int(depth_series.max()),
            }
        else:
            stats[tx_type] = {
                'tx_count': 0, 'mean': 0.0, 'median': 0.0, 'min': 0, 'max': 0
            }

    return stats


def print_internal_tx_depth_statistics(stats: Dict[str, Dict[str, float]]):
    """Print internal transaction depth statistics in a formatted table."""
    print("\n" + "=" * 70)
    print("INTERNAL TRANSACTION DEPTH STATISTICS (traces per tx hash)")
    print("=" * 70)

    for tx_type, s in stats.items():
        print(f"\n{tx_type.upper().replace('_', ' ')}")
        print(f"  Transactions:  {s['tx_count']:>10,}")
        print(f"  Mean:          {s['mean']:>10.2f}")
        print(f"  Median:        {s['median']:>10.2f}")
        print(f"  Min:           {s['min']:>10,}")
        print(f"  Max:           {s['max']:>10,}")

    print("\n" + "=" * 70 + "\n")
