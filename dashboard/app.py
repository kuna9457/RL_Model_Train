"""FinRL training dashboard.

Run from the repo root (with the venv active):
    streamlit run dashboard/app.py

Pick symbols -> download data -> train models (live progress) -> backtest and
compare. All heavy work runs in a background process (dashboard/pipeline.py),
so the page stays responsive and you can close / reopen the browser.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    ALL_MODELS,
    BENCHMARKS,
    DEFAULT_INDICATORS,
    EXTRA_INDICATORS,
    HISTORY_FILE,
    LOWER_IS_BETTER,
    REPO_ROOT,
    RUNS_DIR,
    SYMBOL_PRESETS,
    default_config,
    list_runs,
    pid_alive,
    read_csv_safe,
    read_json,
    run_dir,
    write_json,
)

st.set_page_config(page_title="FinRL Dashboard", page_icon="📈", layout="wide")

GREEN = "#d4f4dd"
RED = "#fbd5d5"
GREEN_TXT = "#137333"
RED_TXT = "#c5221f"


# ------------------------------------------------------------------ helpers


def job_status(rdir: Path) -> dict:
    s = read_json(rdir / "status.json", {}) or {}
    if s.get("state") == "running" and not pid_alive(s.get("pid")):
        s["state"] = "stopped"
        s["message"] = (s.get("message") or "") + " (process ended unexpectedly - see log)"
    return s


def is_running(rdir: Path) -> bool:
    return job_status(rdir).get("state") == "running"


def worker_python() -> str:
    """Python used for training. Prefer the repo's venv so it works even if
    streamlit itself was started from a different (global) Python."""
    for rel in ("venv/Scripts/python.exe", "venv/bin/python", ".venv/Scripts/python.exe", ".venv/bin/python"):
        p = REPO_ROOT / rel
        if p.exists():
            return str(p)
    return sys.executable


def launch(run: str, stage: str):
    rdir = run_dir(run)
    log_file = open(rdir / "log.txt", "a", encoding="utf-8")
    log_file.write(f"\n\n========== {datetime.now():%Y-%m-%d %H:%M:%S}  stage={stage} ==========\n")
    log_file.flush()
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    proc = subprocess.Popen(
        [worker_python(), str(Path(__file__).parent / "pipeline.py"), "--run", run, "--stage", stage],
        cwd=str(REPO_ROOT),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
        creationflags=flags,
    )
    # Write a provisional status so the UI flips to "running" immediately.
    write_json(
        rdir / "status.json",
        {"state": "running", "stage": stage, "pid": proc.pid, "message": "Starting...", "models": {}},
    )


def stop(rdir: Path):
    s = read_json(rdir / "status.json", {}) or {}
    pid = s.get("pid")
    if pid and pid_alive(pid):
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
        else:
            os.kill(pid, 9)
    s["state"] = "stopped"
    s["message"] = "Stopped by user"
    write_json(rdir / "status.json", s)


def tail(path: Path, n: int = 40) -> str:
    if not path.exists():
        return ""
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    # Collapse yfinance progress-bar noise.
    lines = [ln for ln in lines if "%***" not in ln]
    return "".join(lines[-n:])


def fmt_secs(s) -> str:
    if s is None:
        return "-"
    s = int(s)
    return f"{s // 3600}h {s % 3600 // 60}m" if s >= 3600 else f"{s // 60}m {s % 60}s"


def color_metric_table(df: pd.DataFrame, prev: pd.DataFrame | None):
    """Green if better than the previous run of the same model, red if worse."""

    def style(col: pd.Series):
        out = []
        for model, val in col.items():
            css = ""
            if prev is not None and model in prev.index and col.name in prev.columns:
                old = prev.at[model, col.name]
                if pd.notna(old) and pd.notna(val) and val != old:
                    better = val < old if col.name in LOWER_IS_BETTER else val > old
                    css = f"background-color: {GREEN if better else RED}"
            out.append(css)
        return out

    def best_bold(col: pd.Series):
        if col.dropna().empty:
            return [""] * len(col)
        target = col.min() if col.name in LOWER_IS_BETTER else col.max()
        return ["font-weight: bold" if v == target else "" for v in col]

    return (
        df.style.apply(style, axis=0)
        .apply(best_bold, axis=0)
        .format("{:,.2f}", na_rep="-")
    )


def previous_metrics(run: str) -> pd.DataFrame | None:
    """Latest metrics per model from any *earlier* backtest (other runs or older backtests)."""
    hist = read_csv_safe(HISTORY_FILE)
    if hist is None or hist.empty:
        return None
    hist = hist.sort_values("time")
    this_run = hist[hist["run"] == run]
    if not this_run.empty:
        last_time = this_run["time"].max()
        hist = hist[hist["time"] < last_time]
    if hist.empty:
        return None
    return hist.groupby("model").last()


# ------------------------------------------------------------------ sidebar

RUNS_DIR.mkdir(parents=True, exist_ok=True)
st.sidebar.title("📈 FinRL Dashboard")

runs = list_runs()
mode = st.sidebar.radio("Experiment", ["Open existing", "Create new"], index=0 if runs else 1)
if mode == "Create new" or not runs:
    new_name = st.sidebar.text_input("New run name", value=f"run_{datetime.now():%Y%m%d_%H%M}")
    if st.sidebar.button("Create run", type="primary", width="stretch"):
        clean = "".join(c for c in new_name if c.isalnum() or c in "-_") or "run"
        rd = run_dir(clean)
        rd.mkdir(parents=True, exist_ok=True)
        if not (rd / "config.json").exists():
            base = read_json(run_dir(runs[0]) / "config.json") if runs else None
            write_json(rd / "config.json", base or default_config())
        st.session_state["run"] = clean
        st.rerun()
    if not runs:
        st.info("Create a run in the sidebar to start.")
        st.stop()
    run = st.session_state.get("run", runs[0])
else:
    default_idx = runs.index(st.session_state["run"]) if st.session_state.get("run") in runs else 0
    run = st.sidebar.selectbox("Run", runs, index=default_idx)
    st.session_state["run"] = run

RDIR = run_dir(run)
cfg = read_json(RDIR / "config.json", default_config())
for k, v in default_config().items():  # fill keys added in newer versions
    cfg.setdefault(k, v)

status = job_status(RDIR)
state = status.get("state", "idle")
icon = {"running": "🟡", "done": "🟢", "error": "🔴", "stopped": "⚪"}.get(state, "⚪")
st.sidebar.markdown(f"**Status:** {icon} {state}  \n{status.get('stage', '')} {status.get('message', '')}")
if state == "running":
    if st.sidebar.button("⏹ Stop job", width="stretch"):
        stop(RDIR)
        st.rerun()

st.sidebar.divider()
st.sidebar.caption("One-click: download → train → backtest with the current settings.")
if st.sidebar.button("▶ Run full pipeline", disabled=state == "running", width="stretch"):
    launch(run, "all")
    st.rerun()

try:
    import torch  # noqa: F401

    gpu = torch.cuda.is_available()
    st.sidebar.caption(f"GPU: {'✅ ' + torch.cuda.get_device_name(0) if gpu else '❌ CPU only'}")
except Exception:  # noqa: BLE001
    gpu = False


def save_cfg():
    write_json(RDIR / "config.json", cfg)


tab_data, tab_train, tab_results, tab_symbol, tab_history = st.tabs(
    ["1️⃣ Data", "2️⃣ Train", "3️⃣ Results", "🔍 Symbol view", "📜 History"]
)

# ---------------------------------------------------------------- 1. data

with tab_data:
    st.subheader("Symbols & dates")
    c1, c2 = st.columns([2, 1])
    with c2:
        preset = st.selectbox("Add a preset", ["-"] + list(SYMBOL_PRESETS))
        if st.button("Add preset symbols") and preset != "-":
            cfg["tickers"] = list(dict.fromkeys(cfg["tickers"] + SYMBOL_PRESETS[preset]))
            save_cfg()
            st.rerun()
        st.caption(
            "Yahoo symbols: NSE `RELIANCE.NS`, BSE `500325.BO`, gold `GC=F`, silver `SI=F`, "
            "crude `CL=F`, crypto `BTC-USD`, US `AAPL`."
        )
    with c1:
        options = list(dict.fromkeys(cfg["tickers"] + sum(SYMBOL_PRESETS.values(), [])))
        tickers = st.multiselect("Symbols", options, default=cfg["tickers"])
        extra = st.text_input("Add other symbols (comma separated)", "")
        tickers = list(dict.fromkeys(tickers + [t.strip().upper() for t in extra.split(",") if t.strip()]))

    d1, d2, d3, d4 = st.columns(4)
    train_start = d1.date_input("Train start", date.fromisoformat(cfg["train_start"]), min_value=date(1990, 1, 1))
    train_end = d2.date_input("Train end", date.fromisoformat(cfg["train_end"]), min_value=date(1990, 1, 1))
    trade_start = d3.date_input("Test start", date.fromisoformat(cfg["trade_start"]), min_value=date(1990, 1, 1))
    trade_end = d4.date_input("Test end", date.fromisoformat(cfg["trade_end"]), min_value=date(1990, 1, 1))

    with st.expander("Features"):
        ind_opts = list(dict.fromkeys(DEFAULT_INDICATORS + EXTRA_INDICATORS + cfg["indicators"]))
        indicators = st.multiselect("Technical indicators (stockstats names)", ind_opts, default=cfg["indicators"])
        f1, f2 = st.columns(2)
        use_vix = f1.checkbox("Add US VIX (^VIX) feature", cfg["use_vix"],
                              help="US market fear index. Only US trading days are kept when on.")
        use_turbulence = f2.checkbox("Add turbulence index", cfg["use_turbulence"])

    if st.button("⬇ Download & prepare data", type="primary", disabled=state == "running"):
        if not tickers:
            st.error("Pick at least one symbol.")
        elif not (train_start < train_end <= trade_start < trade_end):
            st.error("Dates must be: train start < train end ≤ test start < test end.")
        else:
            cfg.update(
                tickers=tickers, train_start=str(train_start), train_end=str(train_end),
                trade_start=str(trade_start), trade_end=str(trade_end), indicators=indicators,
                use_vix=use_vix, use_turbulence=use_turbulence,
            )
            save_cfg()
            launch(run, "download")
            st.rerun()

    info = read_json(RDIR / "data_info.json")
    if info:
        st.success(
            f"Data ready: **{len(info['symbols'])} symbols**, {info['train_days']} train days, "
            f"{info['trade_days']} test days (common start {info['common_start']})."
        )
        if info.get("missing"):
            st.warning(f"No Yahoo data for: {', '.join(info['missing'])} — skipped.")
        st.dataframe(pd.DataFrame(info["summary"]), hide_index=True, width="stretch")
        raw = read_csv_safe(RDIR / "raw_data.csv", usecols=["date", "tic", "close"])
        if raw is not None:
            norm = st.toggle("Normalise prices (start = 100)", True)
            px = raw.pivot(index="date", columns="tic", values="close").sort_index().ffill()
            if norm:
                px = px / px.bfill().iloc[0] * 100
            st.line_chart(px)


# ---------------------------------------------------------------- 2. train

with tab_train:
    st.subheader("Training setup")
    m1, m2, m3 = st.columns([2, 1, 1])
    models = m1.multiselect("Models", ALL_MODELS, default=[m for m in cfg["models"] if m in ALL_MODELS],
                            help="The example script trains a2c, ddpg, ppo, td3, sac.")
    timesteps = m2.number_input("Timesteps per model", 1000, 10_000_000, int(cfg["total_timesteps"]), step=5000,
                                help="Script default 20,000 (demo). Real runs: 100k–1M+.")
    dev_opts = ["auto", "cpu"] + (["cuda"] if gpu else [])
    device = m3.selectbox("Device", dev_opts, index=dev_opts.index(cfg["device"]) if cfg["device"] in dev_opts else 0)

    with st.expander("Environment parameters"):
        e1, e2, e3 = st.columns(3)
        initial_amount = e1.number_input("Initial cash", 1000.0, 1e10, float(cfg["initial_amount"]), step=100000.0)
        hmax = e2.number_input("hmax (max shares per trade)", 1, 100000, int(cfg["hmax"]))
        reward_scaling = e3.number_input("Reward scaling", 1e-8, 1.0, float(cfg["reward_scaling"]), format="%.6f")
        e4, e5, e6 = st.columns(3)
        buy_cost = e4.number_input("Buy cost (fraction)", 0.0, 0.1, float(cfg["buy_cost_pct"]), format="%.4f")
        sell_cost = e5.number_input("Sell cost (fraction)", 0.0, 0.1, float(cfg["sell_cost_pct"]), format="%.4f")
        seed_txt = e6.text_input("Seed (blank = random)", "" if cfg.get("seed") is None else str(cfg["seed"]))

    with st.expander("Model hyperparameters"):
        st.caption("Defaults = values used by the example train script / finrl/config.py.")
        model_params = dict(cfg["model_params"])
        cols = st.columns(max(len(models), 1))
        for col, m in zip(cols, models):
            with col:
                st.markdown(f"**{m.upper()}**")
                p = dict(model_params.get(m) or {})
                for k, v in list(p.items()):
                    key = f"{run}_{m}_{k}"
                    if isinstance(v, bool) or isinstance(v, str):
                        p[k] = st.text_input(k, str(v), key=key)
                    elif isinstance(v, int):
                        p[k] = int(st.number_input(k, value=v, step=1, key=key))
                    else:
                        p[k] = float(st.number_input(k, value=float(v), format="%.6f", key=key))
                model_params[m] = p

    has_data = (RDIR / "train_data.csv").exists()
    b1, b2 = st.columns([1, 4])
    if b1.button("🚀 Start training", type="primary", disabled=state == "running" or not has_data or not models):
        cfg.update(
            models=models, total_timesteps=int(timesteps), device=device,
            initial_amount=initial_amount, hmax=int(hmax), reward_scaling=reward_scaling,
            buy_cost_pct=buy_cost, sell_cost_pct=sell_cost, model_params=model_params,
            seed=int(seed_txt) if seed_txt.strip().lstrip("-").isdigit() else None,
        )
        save_cfg()
        launch(run, "train")
        st.rerun()
    if not has_data:
        b2.info("Download data first (tab 1).")

    st.divider()

    @st.fragment(run_every=3)
    def live_progress():
        s = job_status(RDIR)
        if s.get("state") == "running":
            st.markdown(f"🟡 **Running:** {s.get('stage')} — {s.get('message', '')}")
        mstat = s.get("models", {}) if s.get("stage") in ("train", "all") or s.get("models") else {}
        shown = cfg["models"] if cfg["models"] else []
        if mstat:
            done_steps = sum(v.get("timesteps", 0) for v in mstat.values())
            all_steps = sum(v.get("total", 0) for v in mstat.values()) or 1
            st.progress(min(done_steps / all_steps, 1.0), text=f"Overall: {done_steps:,} / {all_steps:,} steps")

        for m in shown:
            ms = mstat.get(m, {})
            eps = read_csv_safe(RDIR / "progress" / f"{m}_episodes.csv")
            steps = read_csv_safe(RDIR / "progress" / f"{m}_steps.csv")
            if not ms and eps is None:
                continue
            st.markdown(f"#### {m.upper()}  ·  {ms.get('state', 'not run')}")
            total = ms.get("total") or cfg["total_timesteps"]
            t = ms.get("timesteps", 0)
            label = f"{t:,} / {total:,} steps"
            if ms.get("state") == "training":
                label += f" · ETA {fmt_secs(ms.get('eta_sec'))}"
            elif ms.get("seconds"):
                label += f" · took {fmt_secs(ms['seconds'])}"
            st.progress(min(t / total, 1.0) if total else 0.0, text=label)

            c1, c2 = st.columns([1, 1])
            with c1:
                if eps is not None and not eps.empty:
                    last = eps.iloc[-1]
                    delta = last["return_pct"] - eps.iloc[-2]["return_pct"] if len(eps) > 1 else None
                    k1, k2, k3 = st.columns(3)
                    k1.metric("Last episode return", f"{last['return_pct']:.2f}%",
                              None if delta is None else f"{delta:+.2f}%")
                    k2.metric("Best episode", f"{eps['return_pct'].max():.2f}%")
                    k3.metric("Episodes", int(last["episode"]))

                    view = eps[["episode", "timesteps", "return_pct", "final_value", "improved", "saved_best"]].copy()
                    view["improved"] = view["improved"].map({1: "▲ better", 0: "▼ worse"})
                    view["saved_best"] = view["saved_best"].map({1: "💾 saved", 0: ""})

                    def row_color(r):
                        c = GREEN if r["improved"].startswith("▲") else RED
                        return [f"background-color: {c}"] * len(r)

                    st.dataframe(
                        view.iloc[::-1].style.apply(row_color, axis=1).format(
                            {"return_pct": "{:+.2f}%", "final_value": "{:,.0f}", "timesteps": "{:,}"}
                        ),
                        hide_index=True, width="stretch", height=min(38 * (len(view) + 1), 250),
                    )
                else:
                    st.caption("Waiting for the first episode to finish… (one episode = one pass over the training dates)")
            with c2:
                if steps is not None and len(steps) > 1:
                    fig = go.Figure()
                    fig.add_scatter(x=steps["timesteps"], y=steps["mean_reward"], mode="lines",
                                    name="mean reward", line=dict(color="#94a3b8"))
                    fig.add_scatter(x=steps["timesteps"], y=steps["mean_reward"].rolling(10, min_periods=1).mean(),
                                    mode="lines", name="smoothed", line=dict(color="#2563eb", width=2))
                    fig.update_layout(height=230, margin=dict(l=0, r=0, t=20, b=0), showlegend=False,
                                      xaxis_title="timesteps", yaxis_title="reward")
                    st.plotly_chart(fig, width="stretch", key=f"steps_{m}")

        with st.expander("Log", expanded=s.get("state") in ("running", "error")):
            st.code(tail(RDIR / "log.txt", 40) or "(empty)", language="text")

    live_progress()


# -------------------------------------------------------------- 3. results

with tab_results:
    st.subheader("Backtest on the test period")
    r1, r2, r3 = st.columns([2, 1, 1])
    bench_names = list(BENCHMARKS)
    cur_bench = next((k for k, v in BENCHMARKS.items() if v == cfg.get("benchmark")), None)
    bench_choice = r1.selectbox(
        "Benchmark", bench_names + (["Custom"] if cur_bench is None else []),
        index=bench_names.index(cur_bench) if cur_bench else len(bench_names),
    )
    benchmark = BENCHMARKS.get(bench_choice, cfg.get("benchmark", ""))
    custom = r1.text_input("Custom benchmark symbol", cfg.get("benchmark", "")) if bench_choice == "Custom" else None
    if custom is not None:
        benchmark = custom.strip()
    ckpt = r2.radio("Model to test", ["final", "best"], index=0 if cfg.get("backtest_checkpoint") != "best" else 1,
                    help="final = end of training (like the script). best = checkpoint with the best training episode.",
                    horizontal=True)
    turb = r3.number_input("Turbulence threshold", 0.0, 1e6, float(cfg.get("turbulence_threshold") or 70),
                           help="Used when VIX/turbulence features are on: the agent sells everything above it.")

    has_models = (RDIR / "models").exists() and any((RDIR / "models").glob("agent_*.zip"))
    if st.button("📊 Run backtest", type="primary", disabled=state == "running" or not has_models):
        cfg.update(benchmark=benchmark, backtest_checkpoint=ckpt, turbulence_threshold=turb)
        save_cfg()
        launch(run, "backtest")
        st.rerun()
    if not has_models:
        st.info("Train at least one model first (tab 2).")
    if state == "running" and status.get("stage") in ("backtest", "all"):
        st.info(f"Running: {status.get('message')}")

    metrics = read_json(RDIR / "backtest" / "metrics.json")
    values = read_csv_safe(RDIR / "backtest" / "portfolio_values.csv", index_col=0)
    if metrics and values is not None:
        mdf = pd.DataFrame(metrics).T
        prev = previous_metrics(run)

        drl = [m for m in mdf.index if m in ALL_MODELS]
        if drl:
            best = mdf.loc[drl, "Sharpe"].idxmax()
            st.success(
                f"🏆 Best model by Sharpe: **{best.upper()}** — return {mdf.at[best, 'Total return %']:.2f}%, "
                f"Sharpe {mdf.at[best, 'Sharpe']:.2f}, max drawdown {mdf.at[best, 'Max drawdown %']:.2f}%"
            )

        cols = st.columns(len(mdf))
        for col, (name, row) in zip(cols, mdf.iterrows()):
            delta = None
            if prev is not None and name in prev.index:
                delta = f"{row['Total return %'] - prev.at[name, 'Total return %']:+.2f}% vs last"
            col.metric(name.upper() if name in ALL_MODELS else name, f"{row['Total return %']:.2f}%", delta)

        st.markdown("**All metrics** — 🟩 better / 🟥 worse than the previous backtest of the same model · **bold** = best")
        st.dataframe(color_metric_table(mdf, prev), width="stretch")

        fig = go.Figure()
        for c in values.columns:
            dash = "dot" if c not in ALL_MODELS else None
            fig.add_scatter(x=values.index, y=values[c], name=c, mode="lines", line=dict(dash=dash))
        fig.update_layout(title="Portfolio value", height=420, hovermode="x unified",
                          margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, width="stretch")

        dd = values / values.cummax() - 1
        fig2 = go.Figure()
        for c in dd.columns:
            fig2.add_scatter(x=dd.index, y=dd[c] * 100, name=c, mode="lines")
        fig2.update_layout(title="Drawdown %", height=300, hovermode="x unified", margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig2, width="stretch")
    else:
        st.caption("No backtest yet for this run.")


# ----------------------------------------------------------- symbol view

with tab_symbol:
    st.subheader("Per-symbol trades")
    trade = read_csv_safe(RDIR / "trade_data.csv")
    action_files = sorted((RDIR / "backtest").glob("actions_*.csv")) if (RDIR / "backtest").exists() else []
    if trade is None or not action_files:
        st.caption("Run a backtest to see per-symbol buys and sells.")
    else:
        s1, s2 = st.columns(2)
        model_sel = s1.selectbox("Model", [p.stem.replace("actions_", "") for p in action_files])
        sym = s2.selectbox("Symbol", sorted(trade["tic"].unique()))
        actions = pd.read_csv(RDIR / "backtest" / f"actions_{model_sel}.csv", index_col=0)
        if "actions" in actions.columns and len(actions.columns) == 1:  # single-symbol format
            actions = actions.rename(columns={"actions": sym})
        price = trade[trade["tic"] == sym].set_index("date")["close"]
        if sym in actions.columns:
            act = actions[sym].astype(float)
            act.index = act.index.astype(str)
            buys = act[act > 0]
            sells = act[act < 0]
            fig = go.Figure()
            fig.add_scatter(x=price.index, y=price, name="close", mode="lines", line=dict(color="#475569"))
            fig.add_scatter(x=buys.index, y=price.reindex(buys.index), mode="markers", name="buy",
                            marker=dict(symbol="triangle-up", size=10, color=GREEN_TXT),
                            text=[f"buy {v:.0f}" for v in buys], hoverinfo="text+x")
            fig.add_scatter(x=sells.index, y=price.reindex(sells.index), mode="markers", name="sell",
                            marker=dict(symbol="triangle-down", size=10, color=RED_TXT),
                            text=[f"sell {-v:.0f}" for v in sells], hoverinfo="text+x")
            fig.update_layout(title=f"{sym} — {model_sel.upper()} trades", height=420, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, width="stretch")

            holdings = act.cumsum()
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Buy days", len(buys))
            k2.metric("Sell days", len(sells))
            k3.metric("Final holding (shares)", f"{holdings.iloc[-1]:.0f}" if len(holdings) else "0")
            k4.metric("Price change in test", f"{(price.iloc[-1] / price.iloc[0] - 1) * 100:+.2f}%")
            st.area_chart(holdings.rename("shares held"), height=200)
        else:
            st.warning(f"No actions recorded for {sym}.")


# --------------------------------------------------------------- history

with tab_history:
    st.subheader("All backtests")
    hist = read_csv_safe(HISTORY_FILE)
    if hist is None or hist.empty:
        st.caption("No backtests yet.")
    else:
        only_drl = st.toggle("Only DRL models", True)
        h = hist[hist["model"].isin(ALL_MODELS)] if only_drl else hist
        st.dataframe(
            h.sort_values("time", ascending=False).style.format(precision=2),
            hide_index=True, width="stretch",
        )
        if not h.empty:
            fig = go.Figure()
            for m, g in h.groupby("model"):
                fig.add_scatter(x=g["time"], y=g["Sharpe"], mode="lines+markers", name=m)
            fig.update_layout(title="Sharpe over backtests", height=320, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, width="stretch")
