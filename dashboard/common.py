"""Shared helpers for the FinRL dashboard (no heavy imports here).

Both the Streamlit app and the background worker import this module, so it
must stay light: no torch / stable-baselines3 / finrl imports.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

DASHBOARD_DIR = Path(__file__).resolve().parent
REPO_ROOT = DASHBOARD_DIR.parent
RUNS_DIR = DASHBOARD_DIR / "runs"
HISTORY_FILE = RUNS_DIR / "history.csv"

# Same list as finrl/config.py INDICATORS (copied so the UI does not import finrl).
DEFAULT_INDICATORS = [
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "cci_30",
    "dx_30",
    "close_30_sma",
    "close_60_sma",
]
EXTRA_INDICATORS = ["rsi_14", "cci_14", "close_20_sma", "close_200_sma", "atr", "kdjk"]

ALL_MODELS = ["a2c", "ddpg", "ppo", "td3", "sac", "tqc", "crossq"]
SCRIPT_MODELS = ["a2c", "ddpg", "ppo", "td3", "sac"]  # what the example train script uses

# Hyperparameters used by examples/FinRL_StockTrading_2026_2_train.py.
# `None` means "use finrl/config.py <MODEL>_PARAMS", exactly like the script.
DEFAULT_MODEL_PARAMS: dict[str, dict | None] = {
    "a2c": None,
    "ddpg": None,
    "ppo": {"n_steps": 2048, "ent_coef": 0.01, "learning_rate": 0.00025, "batch_size": 128},
    "td3": {"batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.001},
    "sac": {
        "batch_size": 128,
        "buffer_size": 100000,
        "learning_rate": 0.0001,
        "learning_starts": 100,
        "ent_coef": "auto_0.1",
    },
    "tqc": None,
    "crossq": None,
}

# Values of finrl/config.py <MODEL>_PARAMS, shown in the UI as editable defaults.
CONFIG_MODEL_PARAMS: dict[str, dict] = {
    "a2c": {"n_steps": 5, "ent_coef": 0.01, "learning_rate": 0.0007},
    "ddpg": {"batch_size": 128, "buffer_size": 50000, "learning_rate": 0.001},
    "tqc": {
        "batch_size": 64,
        "buffer_size": 100000,
        "learning_rate": 0.0001,
        "learning_starts": 100,
        "ent_coef": "auto_0.1",
        "top_quantiles_to_drop_per_net": 2,
    },
    "crossq": {
        "batch_size": 256,
        "buffer_size": 1000000,
        "learning_rate": 0.001,
        "learning_starts": 100,
        "ent_coef": "auto",
    },
}

SYMBOL_PRESETS = {
    "Indian large caps (NSE)": [
        "RELIANCE.NS",
        "TCS.NS",
        "HDFCBANK.NS",
        "INFY.NS",
        "ICICIBANK.NS",
        "HINDUNILVR.NS",
        "ITC.NS",
        "SBIN.NS",
        "BHARTIARTL.NS",
        "LT.NS",
    ],
    "Gold & Silver (futures)": ["GC=F", "SI=F"],
    "Gold & Silver (India ETFs)": ["GOLDBEES.NS", "SILVERBEES.NS"],
    "Commodities": ["GC=F", "SI=F", "CL=F", "HG=F"],
    "US mega caps": ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "JPM", "XOM"],
    "Crypto": ["BTC-USD", "ETH-USD"],
}

BENCHMARKS = {
    "NIFTY 50 (^NSEI)": "^NSEI",
    "SENSEX (^BSESN)": "^BSESN",
    "Dow Jones (^DJI)": "^DJI",
    "S&P 500 (^GSPC)": "^GSPC",
    "Gold futures (GC=F)": "GC=F",
    "None": "",
}


def default_config() -> dict:
    return {
        "tickers": SYMBOL_PRESETS["Indian large caps (NSE)"][:5],
        "train_start": "2015-01-01",
        "train_end": "2024-12-31",
        "trade_start": "2025-01-01",
        "trade_end": "2025-12-31",
        "indicators": list(DEFAULT_INDICATORS),
        "use_vix": False,
        "use_turbulence": True,
        "models": list(SCRIPT_MODELS),
        "total_timesteps": 20000,
        "initial_amount": 1000000,
        "hmax": 100,
        "buy_cost_pct": 0.001,
        "sell_cost_pct": 0.001,
        "reward_scaling": 1e-4,
        "turbulence_threshold": 70,
        "device": "auto",
        "seed": None,
        "benchmark": "^NSEI",
        "backtest_checkpoint": "final",
        "model_params": {
            m: (DEFAULT_MODEL_PARAMS[m] or CONFIG_MODEL_PARAMS.get(m, {})) for m in ALL_MODELS
        },
    }


# ---------------------------------------------------------------- run files


def run_dir(name: str) -> Path:
    return RUNS_DIR / name


def list_runs() -> list[str]:
    if not RUNS_DIR.exists():
        return []
    runs = [p for p in RUNS_DIR.iterdir() if p.is_dir() and (p / "config.json").exists()]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.name for p in runs]


def read_json(path: Path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path: Path, data) -> None:
    """Atomic write so the UI never reads a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    for _ in range(20):  # Windows can briefly lock the target while it is being read
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.05)
    os.replace(tmp, path)


def read_csv_safe(path: Path, **kwargs) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, **kwargs)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return None


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    if os.name == "nt":
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        code = ctypes.c_ulong()
        ok = ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(handle)
        return bool(ok) and code.value == STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


# ------------------------------------------------------------------ metrics


def compute_metrics(values: pd.Series, initial_amount: float | None = None) -> dict:
    """Standard backtest metrics from a daily portfolio-value series."""
    values = pd.Series(values).dropna().astype(float)
    if len(values) < 2:
        return {}
    start = float(initial_amount or values.iloc[0])
    end = float(values.iloc[-1])
    rets = values.pct_change().dropna()
    n_days = len(rets)
    years = n_days / 252 if n_days else 0
    total_return = end / start - 1
    cagr = (end / start) ** (1 / years) - 1 if years > 0 and end > 0 else np.nan
    vol = rets.std() * np.sqrt(252)
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else np.nan
    downside = rets[rets < 0].std()
    sortino = rets.mean() / downside * np.sqrt(252) if downside and downside > 0 else np.nan
    drawdown = values / values.cummax() - 1
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if max_dd < 0 and not np.isnan(cagr) else np.nan
    return {
        "Final value": end,
        "Total return %": total_return * 100,
        "CAGR %": cagr * 100,
        "Volatility %": vol * 100,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max drawdown %": max_dd * 100,
        "Calmar": calmar,
        "Win days %": (rets > 0).mean() * 100,
    }


# Higher is better for every metric except volatility (lower is better).
# Max drawdown is negative, so "higher" (closer to 0) is better there too.
LOWER_IS_BETTER = {"Volatility %"}
