"""Cian vault analysis functions."""

import pandas as pd
import numpy as np
from typing import Dict, List
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


WSTETH_ADDRESS = '0x7f39c581f595b53c5cb19bd0b3f8da6c935e2ca0'
STETH_ADDRESS = '0xae7ab96520de3a18e5e111b5eaab095312d7fe84'


def _deduplicate_wsteth_unwrap(
    transfers: pd.DataFrame,
    wsteth_address: str = WSTETH_ADDRESS,
    steth_address: str = STETH_ADDRESS,
) -> pd.DataFrame:
    """
    Remove wstETH transfer rows only for transactions that also contain a stETH
    transfer. In those cases the wstETH was unwrapped to stETH mid-transaction,
    so the stETH row already captures the full deposit/withdrawal value.

    Transactions that carry only wstETH, only stETH, or neither token are
    returned completely unchanged.
    """
    if transfers.empty:
        return transfers

    token_lower = transfers['token_address'].str.lower()
    wsteth_txs = set(transfers.loc[token_lower == wsteth_address.lower(), 'tx_hash'])
    steth_txs = set(transfers.loc[token_lower == steth_address.lower(), 'tx_hash'])
    both_txs = wsteth_txs & steth_txs

    if not both_txs:
        return transfers

    mask_drop = transfers['tx_hash'].isin(both_txs) & (token_lower == wsteth_address.lower())
    return transfers[~mask_drop].copy()


def calculate_cumulative_token_flow(
    transactions_df: pd.DataFrame,
    transfers_df_list: List[pd.DataFrame],
    function_signatures: Dict[str, List[str]],
    vault_address: str,
    time_column: str = 'datetime',
    freq: str = 'D',
    deduplicate_by_tx_hash: bool = True
) -> pd.DataFrame:
    """
    Calculate cumulative token flow for the Cian vault.

    When deduplicate_by_tx_hash is True (default), wstETH transfer rows are
    dropped for any transaction that also contains a stETH transfer. This
    avoids double-counting when wstETH is unwrapped mid-transaction, creating
    both a wstETH and a stETH transfer event for a single user deposit.
    All other transfers are kept as-is.
    """
    vault_address = vault_address.lower()

    all_transfers = pd.concat(transfers_df_list, ignore_index=True)

    tx_df = transactions_df.copy()
    tx_df['function_signature'] = tx_df['input'].astype(str).str[:10]

    deposit_sigs = function_signatures.get('deposit', [])
    redeem_sigs = function_signatures.get('redeem', [])

    deposit_txs = set(tx_df[tx_df['function_signature'].isin(deposit_sigs)]['tx_hash'])
    redeem_txs = set(tx_df[tx_df['function_signature'].isin(redeem_sigs)]['tx_hash'])

    all_transfers['to_lower'] = all_transfers['to_address'].str.lower()
    all_transfers['from_lower'] = all_transfers['from_address'].str.lower()

    deposits = all_transfers[
        (all_transfers['tx_hash'].isin(deposit_txs)) &
        (all_transfers['to_lower'] == vault_address)
    ].copy()

    if deduplicate_by_tx_hash and not deposits.empty:
        deposits = _deduplicate_wsteth_unwrap(deposits)

    redeems = all_transfers[
        (all_transfers['tx_hash'].isin(redeem_txs)) &
        (all_transfers['from_lower'] == vault_address)
    ].copy()

    deposits['period'] = pd.to_datetime(deposits[time_column]).dt.to_period(freq)
    redeems['period'] = pd.to_datetime(redeems[time_column]).dt.to_period(freq)

    deposit_flow = deposits.groupby('period')['value'].sum()
    redeem_flow = redeems.groupby('period')['value'].sum()

    if len(deposit_flow) > 0 and len(redeem_flow) > 0:
        period_start = min(deposit_flow.index.min(), redeem_flow.index.min())
        period_end = max(deposit_flow.index.max(), redeem_flow.index.max())
    elif len(deposit_flow) > 0:
        period_start, period_end = deposit_flow.index.min(), deposit_flow.index.max()
    else:
        period_start, period_end = redeem_flow.index.min(), redeem_flow.index.max()

    all_periods = pd.period_range(start=period_start, end=period_end, freq=freq)

    result = pd.DataFrame(index=all_periods)
    result['Deposits'] = deposit_flow.reindex(all_periods, fill_value=0)
    result['Redeems'] = redeem_flow.reindex(all_periods, fill_value=0)
    result['Net_Flow'] = result['Deposits'] - result['Redeems']
    result['Cumulative_Flow'] = result['Net_Flow'].cumsum()

    result.index = result.index.to_timestamp()

    return result


def calculate_address_cumulative_token_flow(
    transfers_df_list: List[pd.DataFrame],
    address: str,
    token_index: int,
    time_column: str = 'datetime',
    freq: str = 'D'
) -> pd.DataFrame:
    """Calculate cumulative token flow for a specific address."""
    df = transfers_df_list[token_index].copy()
    address_normalized = address.lower()

    df['from_normalized'] = df['from_address'].str.lower()
    df['to_normalized'] = df['to_address'].str.lower()
    df[time_column] = pd.to_datetime(df[time_column])

    outgoing = df[df['from_normalized'] == address_normalized].copy()
    incoming = df[df['to_normalized'] == address_normalized].copy()

    outgoing_grouped = outgoing.set_index(time_column).resample(freq)['value'].sum()
    incoming_grouped = incoming.set_index(time_column).resample(freq)['value'].sum()

    all_periods_set = set(outgoing_grouped.index) | set(incoming_grouped.index)
    if len(all_periods_set) == 0:
        return pd.DataFrame(columns=['Outgoing', 'Incoming', 'Net_Flow', 'Cumulative_Flow'])

    all_periods = sorted(list(all_periods_set))

    result_df = pd.DataFrame(index=all_periods)
    result_df['Outgoing'] = outgoing_grouped.reindex(all_periods, fill_value=0)
    result_df['Incoming'] = incoming_grouped.reindex(all_periods, fill_value=0)
    result_df['Net_Flow'] = result_df['Incoming'] - result_df['Outgoing']
    result_df['Cumulative_Flow'] = result_df['Net_Flow'].cumsum()

    return result_df


def load_daily_exchange_rates(base_token, exchange_rates_dir):
    """Load daily exchange rates from CSV files."""
    filepath = Path(exchange_rates_dir) / f"{base_token.lower()}_usd.csv"

    if not filepath.exists():
        raise FileNotFoundError(f"Exchange rate file not found: {filepath}")

    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['snapped_at']).dt.date
    df = df[['date', 'price']].copy()
    df = df.drop_duplicates(subset=['date'], keep='first')
    df = df.set_index('date')

    return df


def get_daily_cross_rate(date, base_token, quote_token, cache):
    """
    Get the exchange rate between two tokens using USD as a bridge.

    Falls back to forward-filling with the most recent available price.
    """
    if isinstance(date, pd.Timestamp):
        date = date.date()

    base_key = f"{base_token}_USD"
    quote_key = f"{quote_token}_USD"

    base_df = cache[base_key]
    quote_df = cache[quote_key]

    try:
        if date in base_df.index:
            base_price = base_df.loc[date, 'price']
        else:
            available_dates = [d for d in base_df.index if d <= date]
            base_price = base_df.loc[available_dates[-1], 'price'] if available_dates else base_df.iloc[0]['price']

        if date in quote_df.index:
            quote_price = quote_df.loc[date, 'price']
        else:
            available_dates = [d for d in quote_df.index if d <= date]
            quote_price = quote_df.loc[available_dates[-1], 'price'] if available_dates else quote_df.iloc[0]['price']

        return base_price / quote_price if quote_price > 0 else 1.0
    except Exception as e:
        print(f"Warning: Error getting exchange rate for {date}: {e}. Using rate 1.0")
        return 1.0


def calculate_combined_aave_flows(
    transfers_df_list: List[pd.DataFrame],
    time_column: str = 'datetime',
    freq: str = 'D',
    exchange_rates_dir: str = None
) -> pd.DataFrame:
    """
    Calculate combined cumulative flows for all three Aave positions in ETH equivalent.

    Covers the wstETH, WETH, and ezETH positions. wstETH and ezETH are converted
    to ETH using daily cross-rates; WETH is treated as 1:1 with ETH.
    """
    aave_addresses = [
        {
            'address': '0xc035a7cf15375ce2706766804551791ad035e0c2',
            'token_index': 2,  # wstETH
            'token_symbol': 'wstETH',
            'needs_conversion': True
        },
        {
            'address': '0xfa1fdbbd71b0aa16162d76914d69cd8cb3ef92da',
            'token_index': 1,  # WETH
            'token_symbol': 'WETH',
            'needs_conversion': False  # WETH = ETH 1:1
        },
        {
            'address': '0x74e5664394998f13b07af42446380acef637969f',
            'token_index': 3,  # ezETH
            'token_symbol': 'ezETH',
            'needs_conversion': True
        }
    ]

    exchange_rate_cache = {}
    for token in ['wstETH', 'ezETH', 'ETH']:
        try:
            exchange_rate_cache[f"{token}_USD"] = load_daily_exchange_rates(token, exchange_rates_dir)
        except FileNotFoundError:
            print(f"Warning: Exchange rate file for {token}/USD not found")

    flows = []
    for aave_addr in aave_addresses:
        flow_df = calculate_address_cumulative_token_flow(
            transfers_df_list,
            aave_addr['address'],
            aave_addr['token_index'],
            time_column,
            freq
        )

        if aave_addr['needs_conversion']:
            for idx in flow_df.index:
                rate = get_daily_cross_rate(
                    idx,
                    base_token=aave_addr['token_symbol'],
                    quote_token='ETH',
                    cache=exchange_rate_cache
                )
                flow_df.loc[idx, 'Outgoing'] *= rate
                flow_df.loc[idx, 'Incoming'] *= rate

            flow_df['Net_Flow'] = flow_df['Incoming'] - flow_df['Outgoing']
            flow_df['Cumulative_Flow'] = flow_df['Net_Flow'].cumsum()

        flows.append(flow_df)

    all_indices = sorted(set().union(*[set(f.index) for f in flows]))

    result_df = pd.DataFrame(index=all_indices)
    result_df['Aave_wstETH_Cumulative'] = flows[0]['Cumulative_Flow'].reindex(all_indices, fill_value=0)
    result_df['Aave_WETH_Cumulative'] = flows[1]['Cumulative_Flow'].reindex(all_indices, fill_value=0)
    result_df['Aave_ezETH_Cumulative'] = flows[2]['Cumulative_Flow'].reindex(all_indices, fill_value=0)

    result_df['Combined_Cumulative_Flow'] = (
        result_df['Aave_wstETH_Cumulative'] +
        result_df['Aave_WETH_Cumulative'] +
        result_df['Aave_ezETH_Cumulative']
    )

    return result_df


def _lookup_eth_price(date, exchange_df):
    """Look up the ETH/USD price for a given date, forward-filling if needed."""
    date_only = date.date() if hasattr(date, 'date') else date
    if date_only in exchange_df.index:
        return exchange_df.loc[date_only, 'price']
    nearest_date = min(exchange_df.index, key=lambda x: abs((pd.Timestamp(x) - pd.Timestamp(date)).days))
    return exchange_df.loc[nearest_date, 'price']


def convert_eth_cumulative_balance_to_usd(
    eth_flow_df: pd.DataFrame,
    eth_usd_csv_path: str
) -> pd.DataFrame:
    """Convert ETH cumulative balance to USD using daily exchange rates."""
    if eth_flow_df.empty:
        return pd.DataFrame()

    exchange_df = pd.read_csv(eth_usd_csv_path)
    exchange_df['date'] = pd.to_datetime(exchange_df['snapped_at']).dt.date
    exchange_df = exchange_df.set_index('date')

    result = eth_flow_df.copy()

    result['Cumulative_Flow'] = [
        result.loc[date, 'Cumulative_Flow'] * _lookup_eth_price(date, exchange_df)
        for date in result.index
    ]

    if 'Deposits' in result.columns:
        result['Deposits'] = result['Deposits'] * result.index.map(
            lambda d: _lookup_eth_price(d, exchange_df)
        )
    if 'Redeems' in result.columns:
        result['Redeems'] = result['Redeems'] * result.index.map(
            lambda d: _lookup_eth_price(d, exchange_df)
        )
    if 'Net_Flow' in result.columns:
        result['Net_Flow'] = result['Deposits'] - result['Redeems']

    return result


def convert_combined_aave_flows_to_usd(
    eth_flow_df: pd.DataFrame,
    eth_usd_csv_path: str
) -> pd.DataFrame:
    """Convert combined Aave flows from ETH equivalent to USD."""
    if eth_flow_df.empty:
        return pd.DataFrame()

    exchange_df = pd.read_csv(eth_usd_csv_path)
    exchange_df['date'] = pd.to_datetime(exchange_df['snapped_at']).dt.date
    exchange_df = exchange_df.set_index('date')

    result = eth_flow_df.copy()

    for col in result.columns:
        result[col] = [
            result.loc[date, col] * _lookup_eth_price(date, exchange_df)
            for date in result.index
        ]

    return result


def calculate_deposit_withdrawal_statistics(
    transactions_df: pd.DataFrame,
    transfers_df_list: List[pd.DataFrame],
    function_signatures: Dict[str, List[str]],
    vault_address: str,
    eth_usd_csv_path: str,
    start_date: str = None,
    end_date: str = None,
    deduplicate_by_tx_hash: bool = True
) -> None:
    """
    Print statistics for individual deposit and withdrawal transfers in USD.

    When deduplicate_by_tx_hash is True (default), wstETH transfer rows are
    dropped for any transaction that also contains a stETH transfer, to avoid
    double-counting from wstETH unwrapping events. All other transfers are
    kept as-is.
    """
    all_transfers = pd.concat(transfers_df_list, ignore_index=True)

    transactions_df = transactions_df.copy()
    transactions_df['function_signature'] = transactions_df['input'].astype(str).str[:10]

    deposit_sigs = function_signatures.get('deposit', [])
    redeem_sigs = function_signatures.get('redeem', [])

    deposit_txs = transactions_df[transactions_df['function_signature'].isin(deposit_sigs)]['tx_hash'].unique()
    redeem_txs = transactions_df[transactions_df['function_signature'].isin(redeem_sigs)]['tx_hash'].unique()

    vault_addr_lower = vault_address.lower()

    deposit_transfers = all_transfers[
        (all_transfers['tx_hash'].isin(deposit_txs)) &
        (all_transfers['to_address'].str.lower() == vault_addr_lower)
    ].copy()

    if deduplicate_by_tx_hash and len(deposit_transfers) > 0:
        deposit_transfers = _deduplicate_wsteth_unwrap(deposit_transfers)

    withdrawal_transfers = all_transfers[
        (all_transfers['tx_hash'].isin(redeem_txs)) &
        (all_transfers['from_address'].str.lower() == vault_addr_lower)
    ].copy()

    if deduplicate_by_tx_hash and len(withdrawal_transfers) > 0:
        withdrawal_transfers = _deduplicate_wsteth_unwrap(withdrawal_transfers)

    deposit_transfers['datetime'] = pd.to_datetime(deposit_transfers['datetime'])
    withdrawal_transfers['datetime'] = pd.to_datetime(withdrawal_transfers['datetime'])

    if start_date:
        deposit_transfers = deposit_transfers[deposit_transfers['datetime'] >= start_date]
        withdrawal_transfers = withdrawal_transfers[withdrawal_transfers['datetime'] >= start_date]
    if end_date:
        deposit_transfers = deposit_transfers[deposit_transfers['datetime'] <= end_date]
        withdrawal_transfers = withdrawal_transfers[withdrawal_transfers['datetime'] <= end_date]

    eth_usd_df = pd.read_csv(eth_usd_csv_path)
    eth_usd_df['date'] = pd.to_datetime(eth_usd_df['snapped_at']).dt.date
    eth_usd_df = eth_usd_df.set_index('date')

    def convert_to_usd(df, exchange_df):
        if len(df) == 0:
            return pd.Series(dtype=float)

        usd_values = []
        for _, row in df.iterrows():
            date = row['datetime'].date()
            if date in exchange_df.index:
                eth_price = exchange_df.loc[date, 'price']
            else:
                available_dates = [d for d in exchange_df.index if d <= date]
                eth_price = exchange_df.loc[available_dates[-1], 'price'] if available_dates else exchange_df.iloc[0]['price']
            usd_values.append(row['value'] * eth_price)

        return pd.Series(usd_values)

    deposit_usd = convert_to_usd(deposit_transfers, eth_usd_df)
    withdrawal_usd = convert_to_usd(withdrawal_transfers, eth_usd_df)

    print("="*70)
    print("DEPOSIT AND WITHDRAWAL STATISTICS (USD)")
    print("="*70)

    print("\n" + "-"*70)
    print("OVERALL SUMMARY")
    print("-"*70)
    print(f"Total Deposits:     ${deposit_usd.sum():>15,.2f}")
    print(f"Total Withdrawals:  ${withdrawal_usd.sum():>15,.2f}")
    print(f"Net Flow:           ${deposit_usd.sum() - withdrawal_usd.sum():>15,.2f}")

    if len(deposit_usd) > 0:
        print("\n" + "-"*70)
        print("DEPOSIT STATISTICS (individual transfers)")
        print("-"*70)
        print(f"Count:              {len(deposit_usd):>15,}")
        print(f"Mean:               ${deposit_usd.mean():>15,.2f}")
        print(f"Median:             ${deposit_usd.median():>15,.2f}")
        print(f"Std Dev:            ${deposit_usd.std():>15,.2f}")
        print(f"Min:                ${deposit_usd.min():>15,.2f}")
        print(f"25th Percentile:    ${deposit_usd.quantile(0.25):>15,.2f}")
        print(f"75th Percentile:    ${deposit_usd.quantile(0.75):>15,.2f}")
        print(f"Max:                ${deposit_usd.max():>15,.2f}")
    else:
        print("\nNo deposit transfers found in the specified period.")

    if len(withdrawal_usd) > 0:
        print("\n" + "-"*70)
        print("WITHDRAWAL STATISTICS (individual transfers)")
        print("-"*70)
        print(f"Count:              {len(withdrawal_usd):>15,}")
        print(f"Mean:               ${withdrawal_usd.mean():>15,.2f}")
        print(f"Median:             ${withdrawal_usd.median():>15,.2f}")
        print(f"Std Dev:            ${withdrawal_usd.std():>15,.2f}")
        print(f"Min:                ${withdrawal_usd.min():>15,.2f}")
        print(f"25th Percentile:    ${withdrawal_usd.quantile(0.25):>15,.2f}")
        print(f"75th Percentile:    ${withdrawal_usd.quantile(0.75):>15,.2f}")
        print(f"Max:                ${withdrawal_usd.max():>15,.2f}")
    else:
        print("\nNo withdrawal transfers found in the specified period.")

    print("\n" + "="*70)


def calculate_transaction_depth_statistics(
    transactions_df: pd.DataFrame,
    transfers_df_list: List[pd.DataFrame],
    function_signatures: Dict[str, List[str]],
) -> Dict[str, Dict[str, float]]:
    """
    Calculate statistics on the number of token transfers per transaction hash,
    broken down by function type.

    For each transaction type, counts how many rows across all transfer DataFrames
    share the same tx_hash, then summarises the distribution of those counts.
    """
    tx_df = transactions_df.copy()
    tx_df['function_signature'] = tx_df['input'].astype(str).str[:10]

    all_transfers = pd.concat(transfers_df_list, ignore_index=True)

    stats = {}

    for tx_type, signatures in function_signatures.items():
        type_txs = tx_df[tx_df['function_signature'].isin(signatures)]
        tx_hashes = set(type_txs['tx_hash'])

        type_transfers = all_transfers[all_transfers['tx_hash'].isin(tx_hashes)]
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


def calculate_internal_tx_depth_statistics(
    transactions_df: pd.DataFrame,
    function_signatures: Dict[str, List[str]],
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
    print("\n" + "="*70)
    print("INTERNAL TRANSACTION DEPTH STATISTICS (traces per tx hash)")
    print("="*70)

    for tx_type, s in stats.items():
        print(f"\n{tx_type.upper().replace('_', ' ')}")
        print(f"  Transactions:  {s['tx_count']:>10,}")
        print(f"  Mean:          {s['mean']:>10.2f}")
        print(f"  Median:        {s['median']:>10.2f}")
        print(f"  Min:           {s['min']:>10,}")
        print(f"  Max:           {s['max']:>10,}")

    print("\n" + "="*70 + "\n")


def export_deposit_withdrawal_transfers(
    transactions_df: pd.DataFrame,
    transfers_df_list: List[pd.DataFrame],
    function_signatures: Dict[str, List[str]],
    vault_address: str,
    eth_usd_csv_path: str,
    output_dir: str,
    start_date: str = None,
    end_date: str = None,
    deduplicate_by_tx_hash: bool = True
) -> tuple:
    """
    Extract and export individual deposit and withdrawal transfers to CSV files.

    Returns a tuple of (deposit_transfers, withdrawal_transfers) DataFrames
    with USD values added as a value_usd column.
    """
    all_transfers = pd.concat(transfers_df_list, ignore_index=True)

    transactions_df = transactions_df.copy()
    transactions_df['function_signature'] = transactions_df['input'].astype(str).str[:10]

    deposit_sigs = function_signatures.get('deposit', [])
    redeem_sigs = function_signatures.get('redeem', [])

    deposit_txs = transactions_df[transactions_df['function_signature'].isin(deposit_sigs)]['tx_hash'].unique()
    redeem_txs = transactions_df[transactions_df['function_signature'].isin(redeem_sigs)]['tx_hash'].unique()

    vault_addr_lower = vault_address.lower()

    deposit_transfers = all_transfers[
        (all_transfers['tx_hash'].isin(deposit_txs)) &
        (all_transfers['to_address'].str.lower() == vault_addr_lower)
    ].copy()

    if deduplicate_by_tx_hash and len(deposit_transfers) > 0:
        deposit_transfers = _deduplicate_wsteth_unwrap(deposit_transfers)

    withdrawal_transfers = all_transfers[
        (all_transfers['tx_hash'].isin(redeem_txs)) &
        (all_transfers['from_address'].str.lower() == vault_addr_lower)
    ].copy()

    if deduplicate_by_tx_hash and len(withdrawal_transfers) > 0:
        withdrawal_transfers = _deduplicate_wsteth_unwrap(withdrawal_transfers)

    deposit_transfers['datetime'] = pd.to_datetime(deposit_transfers['datetime'])
    withdrawal_transfers['datetime'] = pd.to_datetime(withdrawal_transfers['datetime'])

    if start_date:
        deposit_transfers = deposit_transfers[deposit_transfers['datetime'] >= start_date]
        withdrawal_transfers = withdrawal_transfers[withdrawal_transfers['datetime'] >= start_date]
    if end_date:
        deposit_transfers = deposit_transfers[deposit_transfers['datetime'] <= end_date]
        withdrawal_transfers = withdrawal_transfers[withdrawal_transfers['datetime'] <= end_date]

    eth_usd_df = pd.read_csv(eth_usd_csv_path)
    eth_usd_df['date'] = pd.to_datetime(eth_usd_df['snapped_at']).dt.date
    eth_usd_df = eth_usd_df.set_index('date')

    def convert_to_usd(df, exchange_df):
        if len(df) == 0:
            return pd.Series(dtype=float)

        usd_values = []
        for _, row in df.iterrows():
            date = row['datetime'].date()
            if date in exchange_df.index:
                eth_price = exchange_df.loc[date, 'price']
            else:
                closest_date = min(exchange_df.index, key=lambda x: abs(x - date))
                eth_price = exchange_df.loc[closest_date, 'price']
            usd_values.append(row['value'] * eth_price)

        return pd.Series(usd_values, index=df.index)

    deposit_transfers['value_usd'] = convert_to_usd(deposit_transfers, eth_usd_df)
    withdrawal_transfers['value_usd'] = convert_to_usd(withdrawal_transfers, eth_usd_df)

    essential_columns = ['datetime', 'tx_hash', 'from_address', 'to_address', 'value', 'value_usd', 'token_address']

    deposit_export = deposit_transfers[essential_columns].copy()
    withdrawal_export = withdrawal_transfers[essential_columns].copy()

    deposit_path = f'{output_dir}/cian_deposits.csv'
    withdrawal_path = f'{output_dir}/cian_withdrawals.csv'

    deposit_export.to_csv(deposit_path, index=False)
    withdrawal_export.to_csv(withdrawal_path, index=False)

    print("="*70)
    print("EXPORTED DEPOSIT AND WITHDRAWAL DATA")
    print("="*70)
    print(f"Deposits exported:     {len(deposit_transfers):>6,} records -> {deposit_path}")
    print(f"Withdrawals exported:  {len(withdrawal_transfers):>6,} records -> {withdrawal_path}")
    print("="*70)

    return deposit_transfers, withdrawal_transfers


def export_cumulative_flow_to_csv(
    transactions_df: pd.DataFrame,
    transfers_df_list: List[pd.DataFrame],
    function_signatures: Dict[str, List[str]],
    vault_address: str,
    eth_usd_csv_path: str,
    output_path: str,
    start_date: str = None,
    end_date: str = None,
    freq: str = 'D'
):
    """Export net deposits vs. withdrawals cumulative flow to CSV with USD and ETH columns."""
    flow_eth = calculate_cumulative_token_flow(
        transactions_df, transfers_df_list, function_signatures,
        vault_address, 'datetime', freq, deduplicate_by_tx_hash=True
    )

    if start_date:
        flow_eth = flow_eth[flow_eth.index >= start_date]
    if end_date:
        flow_eth = flow_eth[flow_eth.index <= end_date]

    flow_usd = convert_eth_cumulative_balance_to_usd(flow_eth, eth_usd_csv_path)

    export_data = pd.DataFrame({
        'datetime': flow_usd.index,
        'USD': flow_usd['Cumulative_Flow'],
        'ETH': flow_eth['Cumulative_Flow']
    })

    export_data.to_csv(output_path, index=False)
    print(f"Data exported to: {output_path}")


def export_deleveraged_assets_to_csv(
    transactions_df: pd.DataFrame,
    transfers_df_list: List[pd.DataFrame],
    function_signatures: Dict[str, List[str]],
    vault_address: str,
    eth_usd_csv_path: str,
    output_path: str,
    exchange_rates_dir: str = None,
    start_date: str = None,
    end_date: str = None,
    freq: str = 'D'
):
    """Export deleveraged Cian assets and net vault deposits to CSV in USD."""
    if exchange_rates_dir is None:
        exchange_rates_dir = str(Path(eth_usd_csv_path).parent)

    aave_eth = calculate_combined_aave_flows(
        transfers_df_list, 'datetime', freq, exchange_rates_dir
    )

    if start_date:
        aave_eth = aave_eth[aave_eth.index >= start_date]
    if end_date:
        aave_eth = aave_eth[aave_eth.index <= end_date]

    aave_usd = convert_combined_aave_flows_to_usd(aave_eth, eth_usd_csv_path)

    net_deposits_eth = calculate_cumulative_token_flow(
        transactions_df, transfers_df_list, function_signatures,
        vault_address, 'datetime', freq, deduplicate_by_tx_hash=True
    )

    if start_date:
        net_deposits_eth = net_deposits_eth[net_deposits_eth.index >= start_date]
    if end_date:
        net_deposits_eth = net_deposits_eth[net_deposits_eth.index <= end_date]

    net_deposits_usd = convert_eth_cumulative_balance_to_usd(net_deposits_eth, eth_usd_csv_path)

    all_dates = aave_usd.index.union(net_deposits_usd.index)

    export_data = pd.DataFrame({
        'datetime': all_dates,
        'Deleveraged_Cian_Assets': aave_usd['Combined_Cumulative_Flow'].reindex(all_dates),
        'Net_Vault_Deposits': net_deposits_usd['Cumulative_Flow'].reindex(all_dates)
    })

    export_data.to_csv(output_path, index=False)
    print(f"Data exported to: {output_path}")


def plot_cumulative_token_flow_usd(
    transactions_df: pd.DataFrame,
    transfers_df_list: List[pd.DataFrame],
    function_signatures: Dict[str, List[str]],
    vault_address: str,
    eth_usd_csv_path: str,
    start_date: str = None,
    end_date: str = None,
    freq: str = 'D',
    title: str = None,
    figsize: tuple = (12, 6)
):
    """
    Plot cumulative net deposits vs. withdrawals in both USD and ETH.

    USD is shown on the left y-axis, ETH on the right.
    Returns the matplotlib Figure object.
    """
    flow_eth = calculate_cumulative_token_flow(
        transactions_df, transfers_df_list, function_signatures,
        vault_address, 'datetime', freq, deduplicate_by_tx_hash=True
    )

    if start_date:
        flow_eth = flow_eth[flow_eth.index >= start_date]
    if end_date:
        flow_eth = flow_eth[flow_eth.index <= end_date]

    flow_usd = convert_eth_cumulative_balance_to_usd(flow_eth, eth_usd_csv_path)

    fig, ax = plt.subplots(figsize=figsize)

    line1 = ax.plot(flow_usd.index, flow_usd['Cumulative_Flow'],
                    color='#1f77b4', linewidth=3,
                    label='Net Balance (USD)', marker='o', markersize=3)

    ax_right = ax.twinx()
    line2 = ax_right.plot(flow_eth.index, flow_eth['Cumulative_Flow'],
                          color='#ff7f0e', linewidth=3,
                          label='Net Balance (ETH)', marker='o', markersize=3)

    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)

    ax.set_title(title or 'Net Deposits vs. Withdrawals (USD & ETH)', fontsize=14, fontweight='bold')
    ax.set_ylabel('USD', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax_right.set_ylabel('ETH', fontsize=12)
    ax.grid(True, alpha=0.3)

    lines = line1 + line2
    ax.legend(lines, [l.get_label() for l in lines], loc='best')

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax_right.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

    plt.tight_layout()

    return fig


def plot_combined_aave_flows_usd(
    transactions_df: pd.DataFrame,
    transfers_df_list: List[pd.DataFrame],
    function_signatures: Dict[str, List[str]],
    vault_address: str,
    eth_usd_csv_path: str,
    exchange_rates_dir: str = None,
    start_date: str = None,
    end_date: str = None,
    freq: str = 'D',
    title: str = None,
    figsize: tuple = (12, 6)
):
    """
    Plot combined Aave position flows in USD with net vault deposits as an overlay.

    Returns the matplotlib Figure object.
    """
    if exchange_rates_dir is None:
        exchange_rates_dir = str(Path(eth_usd_csv_path).parent)

    aave_eth = calculate_combined_aave_flows(
        transfers_df_list, 'datetime', freq, exchange_rates_dir
    )

    if start_date:
        aave_eth = aave_eth[aave_eth.index >= start_date]
    if end_date:
        aave_eth = aave_eth[aave_eth.index <= end_date]

    aave_usd = convert_combined_aave_flows_to_usd(aave_eth, eth_usd_csv_path)

    net_deposits_eth = calculate_cumulative_token_flow(
        transactions_df, transfers_df_list, function_signatures,
        vault_address, 'datetime', freq, deduplicate_by_tx_hash=True
    )

    if start_date:
        net_deposits_eth = net_deposits_eth[net_deposits_eth.index >= start_date]
    if end_date:
        net_deposits_eth = net_deposits_eth[net_deposits_eth.index <= end_date]

    net_deposits_usd = convert_eth_cumulative_balance_to_usd(net_deposits_eth, eth_usd_csv_path)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(net_deposits_usd.index, net_deposits_usd['Cumulative_Flow'],
            linewidth=2.5, color='#E8A030', alpha=0.5, linestyle='-',
            label='Net Vault Deposits and Withdrawals', zorder=5)

    ax.plot(aave_usd.index, aave_usd['Combined_Cumulative_Flow'],
            linewidth=3, color='#2E86AB', label='Deleveraged Cian Assets', zorder=10)

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    ax.grid(True, alpha=0.3)
    ax.set_ylabel('USD', fontsize=12)
    ax.set_xlabel('Time', fontsize=12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    y_increment = 10_000_000  # 10M USD steps
    cum_min = min(aave_usd['Combined_Cumulative_Flow'].min(), net_deposits_usd['Cumulative_Flow'].min())
    cum_max = max(aave_usd['Combined_Cumulative_Flow'].max(), net_deposits_usd['Cumulative_Flow'].max())

    y_min = np.floor(cum_min / y_increment) * y_increment - y_increment
    y_max = np.ceil(cum_max / y_increment) * y_increment + y_increment

    ax.set_yticks(np.arange(y_min, y_max + y_increment, y_increment))
    ax.set_ylim(y_min, y_max)

    ax.set_title(title or 'Deleveraged Cian Assets (USD)', fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='best', framealpha=0.9)

    plt.tight_layout()

    print("\n" + "="*70)
    print("DELEVERAGED CIAN ASSETS (USD)")
    print("="*70)
    print(f"Final flow:    ${aave_usd['Combined_Cumulative_Flow'].iloc[-1]:>12,.2f} USD")
    print(f"Maximum flow:  ${aave_usd['Combined_Cumulative_Flow'].max():>12,.2f} USD")
    print(f"Minimum flow:  ${aave_usd['Combined_Cumulative_Flow'].min():>12,.2f} USD")
    print("="*70)

    return fig


def get_raw_cian_vault_exchange_rate_data(
    cian_exchange_df: pd.DataFrame,
    time_column: str = 'datetime'
) -> pd.DataFrame:
    """
    Extract per-event exchange rate data from Cian vault price update events.

    Returns a DataFrame with columns: datetime, exchange_rate, revenue_eth, tx_hash.
    """
    if len(cian_exchange_df) == 0:
        return pd.DataFrame(columns=[time_column, 'exchange_rate', 'revenue_eth', 'tx_hash'])

    required_cols = {'exchange_price_eth', time_column}
    if not required_cols.issubset(cian_exchange_df.columns):
        missing = required_cols - set(cian_exchange_df.columns)
        raise ValueError(f"Cian exchange DataFrame missing required columns: {missing}")

    df = cian_exchange_df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
        df[time_column] = pd.to_datetime(df[time_column])

    df = df[df['exchange_price_eth'] > 0]

    if len(df) == 0:
        return pd.DataFrame(columns=[time_column, 'exchange_rate', 'revenue_eth', 'tx_hash'])

    df['exchange_rate'] = df['exchange_price_eth']
    df = df.sort_values(time_column)

    return df


def plot_cian_vault_exchange_rate_over_time(
    cian_exchange_df: pd.DataFrame,
    time_column: str = 'datetime',
    freq: str = 'D',
    title: str = 'Cian Vault Exchange Rate Over Time',
    figsize: tuple = (12, 6),
    start_date: str = None,
    end_date: str = None
):
    """
    Plot Cian vault exchange rate from dedicated price update events.

    Returns the matplotlib Figure object.
    """
    df = get_raw_cian_vault_exchange_rate_data(cian_exchange_df, time_column)

    if len(df) == 0:
        print("No valid Cian exchange rate data available for plotting.")
        return None

    if start_date:
        df = df[df[time_column] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df[time_column] <= pd.Timestamp(end_date)]

    if len(df) == 0:
        print("No data in the specified date range.")
        return None

    fig, ax1 = plt.subplots(figsize=figsize)

    ax1.plot(df[time_column], df['exchange_rate'],
             label='Exchange Rate (ETH)', color='#2E8B57', linewidth=1.5, marker='o', markersize=3, alpha=0.8)

    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Exchange Rate (ETH per Share)', color='#2E8B57', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='#2E8B57')
    ax1.grid(True, alpha=0.3)

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    ax1.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax1.legend(loc='upper left')

    plt.tight_layout()

    start_rate = df['exchange_rate'].iloc[0]
    end_rate = df['exchange_rate'].iloc[-1]
    total_change = end_rate - start_rate
    pct_change = (total_change / start_rate * 100) if start_rate != 0 else 0

    print("\n" + "="*70)
    print("CIAN VAULT EXCHANGE RATE ANALYSIS")
    print("="*70)
    print(f"Starting rate: {start_rate:.6f} ETH per share")
    print(f"Ending rate:   {end_rate:.6f} ETH per share")
    print(f"Total change:  {pct_change:+.2f}%")

    # Annualized rate projected from data until 2025-04-11 (observation window end)
    cutoff_date = pd.Timestamp('2025-04-11')
    df_filtered = df[df[time_column] <= cutoff_date].copy()

    if len(df_filtered) >= 2:
        first_rate = df_filtered['exchange_rate'].iloc[0]
        last_rate = df_filtered['exchange_rate'].iloc[-1]
        first_date = df_filtered[time_column].iloc[0]
        last_date = df_filtered[time_column].iloc[-1]

        days_elapsed = (last_date - first_date).total_seconds() / (24 * 3600)

        if days_elapsed > 0:
            daily_rate = (last_rate - first_rate) / days_elapsed
            annual_rate_pct = (daily_rate * 365 / first_rate * 100) if first_rate != 0 else 0
            print(f"\nAnnualized rate (projected from data until 2025-04-11): {annual_rate_pct:+.2f}% per year")

    print("="*70)

    return fig
