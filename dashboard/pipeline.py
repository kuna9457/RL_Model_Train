"""Background worker for the FinRL dashboard.

Runs the same three steps as the example scripts
(examples/FinRL_StockTrading_2026_{1_data,2_train,3_Backtest}.py), but for the
symbols / dates / parameters chosen in the dashboard. It only *uses* FinRL
classes (YahooDownloader, FeatureEngineer, StockTradingEnv, DRLAgent); it does
not change any of them.

Usage (the dashboard launches this for you):
    python dashboard/pipeline.py --run <run_name> --stage download|train|backtest|all
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO_ROOT, HISTORY_FILE, compute_metrics, read_json, run_dir, write_json  # noqa: E402

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# --------------------------------------------------------------- status file


class Status:
    def __init__(self, rdir: Path, stage: str):
        self.path = rdir / "status.json"
        self.data = {
            "state": "running",
            "stage": stage,
            "pid": None,
            "started": datetime.now().isoformat(timespec="seconds"),
            "message": "",
            "models": {},
        }
        import os

        self.data["pid"] = os.getpid()
        self.flush()

    def update(self, **kw):
        self.data.update(kw)
        self.data["updated"] = datetime.now().isoformat(timespec="seconds")
        self.flush()

    def model(self, name: str, **kw):
        self.data["models"].setdefault(name, {}).update(kw)
        self.update()

    def flush(self):
        write_json(self.path, self.data)


def log(msg: str):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


# --------------------------------------------------------------- stage 1: data


def stage_download(cfg: dict, rdir: Path, status: Status):
    from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
    from finrl.meta.preprocessor.yahoodownloader import YahooDownloader

    tickers = [t.strip() for t in cfg["tickers"] if t.strip()]
    if not tickers:
        raise ValueError("No symbols selected.")

    status.update(stage="download", message=f"Downloading {len(tickers)} symbols from Yahoo")
    log(f"Downloading {tickers} {cfg['train_start']} -> {cfg['trade_end']}")
    df_raw = YahooDownloader(
        start_date=cfg["train_start"], end_date=cfg["trade_end"], ticker_list=tickers
    ).fetch_data()
    got = sorted(df_raw["tic"].unique())
    missing = sorted(set(tickers) - set(got))
    if missing:
        log(f"WARNING: no data for {missing}; they are skipped.")
    df_raw.to_csv(rdir / "raw_data.csv", index=False)

    status.update(stage="preprocess", message="Adding technical indicators")
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=cfg["indicators"],
        use_vix=cfg["use_vix"],
        use_turbulence=cfg["use_turbulence"],
        user_defined_feature=False,
        keep_incomplete_tickers=True,
    )
    processed = fe.preprocess_data(df_raw)

    # The trading env needs every symbol on every date. Different markets have
    # different holidays (e.g. NSE vs COMEX), so build a full date x symbol grid,
    # carry each symbol's last value over its holidays, and start from the first
    # date on which every symbol has data.
    first_dates = processed.groupby("tic")["date"].min()
    common_start = first_dates.max()
    if common_start > processed["date"].min():
        log(
            f"Symbols start on different dates; using data from {common_start} "
            f"(latest start: {first_dates.idxmax()})."
        )
    dates = sorted(processed["date"].unique())
    grid = pd.DataFrame(list(itertools.product(dates, got)), columns=["date", "tic"])
    full = grid.merge(processed, on=["date", "tic"], how="left")
    full = full.sort_values(["tic", "date"])
    value_cols = [c for c in full.columns if c not in ("date", "tic")]
    full[value_cols] = full.groupby("tic", sort=False)[value_cols].ffill()
    full = full[full["date"] >= common_start]
    full[value_cols] = full[value_cols].fillna(0)
    full = full.sort_values(["date", "tic"]).reset_index(drop=True)

    train = data_split(full, cfg["train_start"], cfg["train_end"])
    trade = data_split(full, cfg["trade_start"], cfg["trade_end"])
    if len(train) == 0 or len(trade) == 0:
        raise ValueError(
            f"Empty split (train rows={len(train)}, trade rows={len(trade)}). Check the dates."
        )
    train.to_csv(rdir / "train_data.csv")
    trade.to_csv(rdir / "trade_data.csv")

    summary = (
        df_raw.groupby("tic")
        .agg(first_date=("date", "min"), last_date=("date", "max"), rows=("date", "count"))
        .reset_index()
    )
    write_json(
        rdir / "data_info.json",
        {
            "symbols": got,
            "missing": missing,
            "common_start": str(common_start),
            "train_days": int(train["date"].nunique()),
            "trade_days": int(trade["date"].nunique()),
            "summary": summary.to_dict(orient="records"),
        },
    )
    log(
        f"Data ready: {len(got)} symbols, {train['date'].nunique()} train days, "
        f"{trade['date'].nunique()} trade days."
    )


# --------------------------------------------------------------- environment


def load_split(rdir: Path, name: str) -> pd.DataFrame:
    path = rdir / f"{name}_data.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found. Run 'Download data' first.")
    df = pd.read_csv(path)
    df = df.set_index(df.columns[0])
    df.index.names = [""]
    return df


def env_kwargs(cfg: dict, df: pd.DataFrame) -> dict:
    # Same structure as the example train / backtest scripts.
    stock_dimension = len(df.tic.unique())
    indicators = cfg["indicators"]
    state_space = 1 + 2 * stock_dimension + len(indicators) * stock_dimension
    return {
        "hmax": int(cfg["hmax"]),
        "initial_amount": float(cfg["initial_amount"]),
        "num_stock_shares": [0] * stock_dimension,
        "buy_cost_pct": [float(cfg["buy_cost_pct"])] * stock_dimension,
        "sell_cost_pct": [float(cfg["sell_cost_pct"])] * stock_dimension,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": indicators,
        "action_space": stock_dimension,
        "reward_scaling": float(cfg["reward_scaling"]),
    }


# -------------------------------------------------------------- stage 2: train


def make_progress_callback(name, total, rdir, cfg, status):
    from stable_baselines3.common.callbacks import BaseCallback

    prog_dir = rdir / "progress"
    prog_dir.mkdir(exist_ok=True)
    steps_file = prog_dir / f"{name}_steps.csv"
    eps_file = prog_dir / f"{name}_episodes.csv"
    steps_file.write_text("timesteps,mean_reward\n")
    eps_file.write_text("episode,timesteps,return_pct,final_value,improved,saved_best\n")
    initial = float(cfg["initial_amount"])
    scaling = float(cfg["reward_scaling"])
    best_path = rdir / "models" / f"agent_{name}_best"

    class ProgressCallback(BaseCallback):
        def __init__(self):
            super().__init__(0)
            self.ep_sum = 0.0
            self.recent = []
            self.episode = 0
            self.best = -np.inf
            self.prev = None
            self.every = max(total // 200, 50)
            self.t0 = time.time()

        def _on_step(self) -> bool:
            r = float(np.asarray(self.locals.get("rewards", [0]))[0])
            self.ep_sum += r
            self.recent.append(r)
            if self.num_timesteps % self.every == 0:
                mean_r = float(np.mean(self.recent)) if self.recent else 0.0
                self.recent = []
                with open(steps_file, "a") as f:
                    f.write(f"{self.num_timesteps},{mean_r}\n")
                elapsed = time.time() - self.t0
                eta = elapsed / max(self.num_timesteps, 1) * (total - self.num_timesteps)
                status.model(
                    name, state="training", timesteps=int(self.num_timesteps), total=total,
                    eta_sec=int(eta),
                )
            dones = self.locals.get("dones")
            if dones is not None and bool(np.asarray(dones)[0]):
                # Reward = change in portfolio value * reward_scaling, so the
                # episode sum gives the profit over the training period.
                profit = self.ep_sum / scaling
                ret = profit / initial * 100
                self.episode += 1
                improved = self.prev is None or ret > self.prev
                saved = ret > self.best
                if saved:
                    self.best = ret
                    self.model.save(str(best_path))
                with open(eps_file, "a") as f:
                    f.write(
                        f"{self.episode},{self.num_timesteps},{ret:.4f},{initial + profit:.2f},"
                        f"{int(improved)},{int(saved)}\n"
                    )
                log(
                    f"{name} episode {self.episode}: train return {ret:+.2f}% "
                    f"({'improved' if improved else 'worse'}{', saved best' if saved else ''})"
                )
                status.model(name, episodes=self.episode, last_return=ret, best_return=self.best)
                self.prev = ret
                self.ep_sum = 0.0
            return True

    return ProgressCallback()


def stage_train(cfg: dict, rdir: Path, status: Status):
    from stable_baselines3.common.logger import configure

    from finrl.agents.stablebaselines3.models import DRLAgent
    from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

    train = load_split(rdir, "train")
    kw = env_kwargs(cfg, train)
    log(f"Stock dimension {kw['stock_dim']}, state space {kw['state_space']}")
    (rdir / "models").mkdir(exist_ok=True)
    total = int(cfg["total_timesteps"])
    models = cfg["models"]
    for m in models:
        status.model(m, state="queued", timesteps=0, total=total)

    for m in models:
        status.update(stage="train", message=f"Training {m.upper()}")
        status.model(m, state="training", started=datetime.now().isoformat(timespec="seconds"))
        log(f"===== Training {m.upper()} for {total} timesteps =====")
        e_train_gym = StockTradingEnv(df=train, **kw)
        env_train, _ = e_train_gym.get_sb_env()
        agent = DRLAgent(env=env_train)
        params = dict(cfg["model_params"].get(m) or {})
        if cfg.get("device") and cfg["device"] != "auto":
            params["device"] = cfg["device"]
        model = agent.get_model(
            m, model_kwargs=params or None, seed=cfg.get("seed"), verbose=0
        )
        model.set_logger(configure(str(rdir / "sb3_logs" / m), ["csv"]))
        t0 = time.time()
        cb = make_progress_callback(m, total, rdir, cfg, status)
        trained = DRLAgent.train_model(
            model=model, tb_log_name=m, total_timesteps=total, callbacks=[cb]
        )
        trained.save(str(rdir / "models" / f"agent_{m}"))
        status.model(
            m, state="done", timesteps=total, eta_sec=0, seconds=int(time.time() - t0)
        )
        log(f"{m.upper()} saved ({time.time() - t0:.0f}s).")


# ----------------------------------------------------------- stage 3: backtest


def stage_backtest(cfg: dict, rdir: Path, status: Status):
    import yfinance as yf

    from finrl.agents.stablebaselines3.models import MODELS, DRLAgent
    from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

    trade = load_split(rdir, "trade")
    kw = env_kwargs(cfg, trade)
    out = rdir / "backtest"
    out.mkdir(exist_ok=True)
    initial = float(cfg["initial_amount"])
    use_best = cfg.get("backtest_checkpoint") == "best"

    curves = {}
    for m in cfg["models"]:
        path = rdir / "models" / (f"agent_{m}_best.zip" if use_best else f"agent_{m}.zip")
        if use_best and not path.exists():
            path = rdir / "models" / f"agent_{m}.zip"
        if not path.exists():
            log(f"No trained model for {m}, skipping.")
            continue
        status.update(stage="backtest", message=f"Backtesting {m.upper()}")
        log(f"Backtesting {m.upper()} ({path.name})")
        model = MODELS[m].load(str(path))
        if cfg["use_vix"]:
            env = StockTradingEnv(
                df=trade,
                turbulence_threshold=cfg.get("turbulence_threshold"),
                risk_indicator_col="vix",
                **kw,
            )
        elif cfg["use_turbulence"]:
            env = StockTradingEnv(
                df=trade, turbulence_threshold=cfg.get("turbulence_threshold"), **kw
            )
        else:
            env = StockTradingEnv(df=trade, **kw)
        acc, actions = DRLAgent.DRL_prediction(model=model, environment=env)
        acc.to_csv(out / f"account_{m}.csv", index=False)
        actions.to_csv(out / f"actions_{m}.csv")
        curves[m] = acc.set_index("date")["account_value"]

    # Baseline 1: equal-weight basket of the selected symbols (daily rebalanced),
    # same idea as the script's equal-weight Dow baseline.
    prices = trade.pivot(index="date", columns="tic", values="close").sort_index()
    ew = (1 + prices.pct_change().mean(axis=1).fillna(0)).cumprod() * initial
    curves["equal_weight"] = ew

    # Baseline 2: benchmark index.
    bench = cfg.get("benchmark")
    if bench:
        try:
            b = yf.download(
                bench, start=cfg["trade_start"], end=cfg["trade_end"], progress=False,
                auto_adjust=False,
            )
            close = b["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close.index = close.index.strftime("%Y-%m-%d")
            curves[f"benchmark ({bench})"] = close / close.iloc[0] * initial
        except Exception as e:  # noqa: BLE001
            log(f"Benchmark {bench} download failed: {e}")

    result = pd.DataFrame(curves).sort_index()
    result.index.name = "date"
    result.to_csv(out / "portfolio_values.csv")

    metrics = {name: compute_metrics(s, initial) for name, s in curves.items()}
    write_json(out / "metrics.json", metrics)

    # Append to the cross-run history so the UI can show improved / worse.
    stamp = datetime.now().isoformat(timespec="seconds")
    rows = [
        {"run": rdir.name, "time": stamp, "model": k, "symbols": " ".join(cfg["tickers"]),
         "timesteps": cfg["total_timesteps"], **v}
        for k, v in metrics.items()
    ]
    hist = pd.DataFrame(rows)
    hist.to_csv(HISTORY_FILE, mode="a", header=not HISTORY_FILE.exists(), index=False)
    log("Backtest done.")
    print(pd.DataFrame(metrics).T.round(3).to_string(), flush=True)


# ------------------------------------------------------------------- main


STAGES = {"download": stage_download, "train": stage_train, "backtest": stage_backtest}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--stage", required=True, choices=[*STAGES, "all"])
    args = ap.parse_args()

    rdir = run_dir(args.run)
    cfg = read_json(rdir / "config.json")
    if cfg is None:
        raise SystemExit(f"Missing {rdir / 'config.json'}")

    status = Status(rdir, args.stage)
    stages = list(STAGES) if args.stage == "all" else [args.stage]
    try:
        for s in stages:
            STAGES[s](cfg, rdir, status)
        status.update(state="done", message=f"Finished: {', '.join(stages)}")
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        status.update(state="error", message=f"{type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
