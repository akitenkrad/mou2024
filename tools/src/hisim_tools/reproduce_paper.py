#!/usr/bin/env python3
"""reproduce_paper.py — Mou et al. (2024) HiSim 見出し的知見の一括再現レポート + 図．

Rust の `hisim reproduce` が書いた run ディレクトリを読み，論文 §5 / Table 2/3 の
中心的知見を 3 つの図で可視化しつつ PASS/off テーブルを表示する:

    1. table3_hybrid_vs_pureabm.png
       ABM 種別 (bc/hk/sj/lorenz) ごとに hybrid (LLM コア + ABM 周辺) と pure-ABM
       (core-ratio 0) の最終 Polarization・正規化 Mobilization を対比する棒グラフ．
       «BC/HK は合意 (低分極) / SJ/Lorenz は二極化» と «LLM コアが動員を牽引» を示す．
    2. bench_alignment.png
       SoMoSiMu-Bench 照合 (#MeToo / RoeOverturned / BlackLivesMatter)．運動別に
       観測 (シミュレータ) vs 合成参照の運動指標を並べ，整合帯を可視化する．
    3. mobilization_curves.png
       代表 run の動員曲線時系列を hybrid vs pure-ABM で重ね描き (BC / Lorenz)．

判定は Rust 側で確定しているので，本ツールは図の生成と要約の再表示に専念する
(再計算しない)．条件セルの数は `metrics.csv` の run スコープ指標
(`<条件>_mean_final_*`)，PASS/off と bench の整合は `events.jsonl` の
`x.mou2024.anchor` / `x.mou2024.bench_alignment` にある．

`--run` を付けると先に Rust バイナリ (`cargo run --release -- reproduce`) を実行して
最新結果を生成する．サンドボックス・CI では `--mock` も付けてライブ LLM を回避する
(pure-ABM 条件は core-ratio 0 で LLM を一切呼ばない)．

--results-dir を省略すると
`runvault path --experiment hisim --latest --subcommand reproduce`
が返す run ディレクトリを対象にする (`runvault` が PATH にある必要がある)．

Usage:
    uv run hisim-tools reproduce --run --mock          # mock で一括再現 + 図
    uv run hisim-tools reproduce --run --mock --quick  # 軽量版 (動作確認用)
    uv run hisim-tools reproduce                        # 既存の最新 run を可視化
    uv run hisim-tools reproduce --results-dir "$(runvault path --experiment hisim --latest --subcommand reproduce)"
    uv run hisim-tools reproduce --json

Outputs:
    <experiment>/figures/<run_slug>/{table3_hybrid_vs_pureabm,bench_alignment,mobilization_curves}.png
    stdout: アンカーごとの PASS / off と bench 整合．
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from runvault.read import (
    config_parameters,
    events_table,
    figures_dir,
    metrics_wide,
    run_scope_metrics,
    runvault_path,
)

# --------------------------------------------------------------------------- #
# runvault 側の名前 (Rust 側 record.rs と揃える)
# --------------------------------------------------------------------------- #
EXPERIMENT = "hisim"
ANCHOR_EVENT = "x.mou2024.anchor"
BENCH_EVENT = "x.mou2024.bench_alignment"

# --------------------------------------------------------------------------- #
# 表示設定 (CJK フォントが利用不能でも落ちないように try)
# --------------------------------------------------------------------------- #
try:
    plt.rcParams["font.family"] = "Hiragino Sans"
except Exception:  # pragma: no cover - フォント未インストール環境用フォールバック
    pass

COLOR_BG = "#FAFAF8"
COLOR_HYBRID = "#2196F3"
COLOR_PUREABM = "#FF9800"
COLOR_OBS = "#2196F3"
COLOR_REF = "#9C27B0"
ABM_ORDER = ["bc", "hk", "sj", "lorenz"]

#: 条件セルの run スコープ指標 (Rust 側 ReproCell::metrics と同じ並び)．
CELL_METRICS = [
    "mean_final_bias",
    "mean_final_diversity",
    "mean_final_polarization",
    "mean_final_mobilization",
    "mean_mobilization_gain",
    "mean_llm_calls",
]


# --------------------------------------------------------------------------- #
# Rust バイナリ実行
# --------------------------------------------------------------------------- #


def _run_binary(*, mock: bool, quick: bool, seed: int, output_dir: str) -> None:
    """`cargo run --release -- reproduce ...` を実行して最新結果を生成する．"""
    cmd = ["cargo", "run", "--release", "--", "reproduce", "--seed", str(seed),
           "--output-dir", output_dir]
    if mock:
        cmd.append("--mock")
    if quick:
        cmd.append("--quick")
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


# --------------------------------------------------------------------------- #
# run ディレクトリの読み出し
# --------------------------------------------------------------------------- #


def cell_table(scoped: dict[str, float], abms: list[str]) -> pd.DataFrame:
    """Table 3 の条件セルを 1 行 1 条件の表に組み直す．

    runvault にはこの表がファイルとして存在しない．`metrics.csv` の run スコープ指標
    は `<条件ラベル>_<指標名>` という名前で 1 本の run に同居しているので，ラベルで
    切り分ける．
    """
    rows: list[dict] = []
    for regime, prefix in (("pure-abm", "pureabm"), ("hybrid", "hybrid")):
        for abm in abms:
            label = f"{prefix}_{abm}"
            if f"{label}_{CELL_METRICS[0]}" not in scoped:
                continue
            row = {"label": label, "regime": regime, "abm": abm}
            row.update({name: scoped[f"{label}_{name}"] for name in CELL_METRICS})
            rows.append(row)
    return pd.DataFrame(rows)


def _events(run_dir: Path, kind: str) -> pd.DataFrame:
    """events.jsonl の 1 種別．無ければ空の表を返す (図をひとつ落とすだけ)．"""
    try:
        return events_table(run_dir, kind=kind)
    except (FileNotFoundError, SystemExit):
        return pd.DataFrame()


# --------------------------------------------------------------------------- #
# 描画
# --------------------------------------------------------------------------- #


def _table3(cells: pd.DataFrame, dataset: str, out_path: Path) -> None:
    """ABM 別 hybrid vs pure-ABM の最終 Polarization・Mobilization 棒グラフ．"""
    hybrid = {r.abm: r for r in cells[cells["regime"] == "hybrid"].itertuples()}
    pure = {r.abm: r for r in cells[cells["regime"] == "pure-abm"].itertuples()}
    abms = [a for a in ABM_ORDER if a in pure and a in hybrid]
    if not abms:
        print("  警告: 条件セルが無いため table3 をスキップ")
        return

    x = np.arange(len(abms))
    w = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=COLOR_BG)
    fig.suptitle(
        f"Mou et al. (2024) HiSim — Table 3: hybrid vs pure-ABM (dataset={dataset})",
        fontsize=13,
    )

    ax = axes[0]
    ax.set_facecolor(COLOR_BG)
    ax.bar(x - w / 2, [pure[a].mean_final_polarization for a in abms], w,
           color=COLOR_PUREABM, label="pure-ABM (core-ratio 0)")
    ax.bar(x + w / 2, [hybrid[a].mean_final_polarization for a in abms], w,
           color=COLOR_HYBRID, label="hybrid (LLM core + ABM)")
    ax.set_xticks(x)
    ax.set_xticklabels([a.upper() for a in abms])
    ax.set_xlabel("周辺 ABM 種別")
    ax.set_ylabel("最終 Polarization")
    ax.set_title("分極化: BC/HK は合意 (低) / SJ/Lorenz は二極化 (高)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    ax.set_facecolor(COLOR_BG)
    ax.bar(x - w / 2, [pure[a].mean_final_mobilization for a in abms], w,
           color=COLOR_PUREABM, label="pure-ABM (core-ratio 0)")
    ax.bar(x + w / 2, [hybrid[a].mean_final_mobilization for a in abms], w,
           color=COLOR_HYBRID, label="hybrid (LLM core + ABM)")
    ax.set_xticks(x)
    ax.set_xticklabels([a.upper() for a in abms])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("周辺 ABM 種別")
    ax.set_ylabel("最終 正規化 Mobilization")
    ax.set_title("動員: LLM コアが call-to-action で動員を牽引", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {out_path}")


def _bench_alignment(bench: pd.DataFrame, out_path: Path) -> None:
    """SoMoSiMu-Bench 照合: 運動別の観測 vs 参照 運動指標 (整合帯)．"""
    if bench.empty:
        print("  警告: bench の整合イベントが無いため bench_alignment をスキップ")
        return
    movements = list(dict.fromkeys(bench["movement"]))
    metrics = list(dict.fromkeys(bench[bench["movement"] == movements[0]]["metric"]))
    n = len(movements)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 5), facecolor=COLOR_BG, squeeze=False)
    fig.suptitle(
        "Mou et al. (2024) HiSim — Table 2: SoMoSiMu-Bench 照合 (観測 vs 合成参照)",
        fontsize=13,
    )
    y = np.arange(len(metrics))
    h = 0.38
    for j, movement in enumerate(movements):
        rows = bench[bench["movement"] == movement].set_index("metric").loc[metrics]
        ax = axes[0][j]
        ax.set_facecolor(COLOR_BG)
        ax.barh(y - h / 2, rows["observed"], h, color=COLOR_OBS, label="observed (sim)")
        ax.barh(y + h / 2, rows["reference"], h, color=COLOR_REF, label="reference (synthetic)")
        ax.set_yticks(y)
        ax.set_yticklabels(metrics, fontsize=8)
        ax.invert_yaxis()
        ok = int(rows["aligned"].sum())
        ax.set_title(f"{movement}  ({ok}/{len(rows)} 整合)", fontsize=11)
        ax.set_xlabel("指標値")
        if j == 0:
            ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3, axis="x")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    source = bench["reference_source"].iloc[0]
    print(f"  保存: {out_path}  (参照は {source}; 実 bench データではない)")


def _mobilization_curves(wide: pd.DataFrame, out_path: Path) -> None:
    """hybrid vs pure-ABM の動員曲線時系列 (代表 run)．

    11 条件が 1 本の run に同居するので，系列は `<条件ラベル>_mobilized` という
    名前で区別されている．
    """
    pairs = [
        ("pureabm_bc", "pure-ABM / BC", COLOR_PUREABM, "--"),
        ("hybrid_bc", "hybrid / BC", COLOR_HYBRID, "-"),
        ("pureabm_lorenz", "pure-ABM / Lorenz", "#4CAF50", "--"),
        ("hybrid_lorenz", "hybrid / Lorenz", "#F44336", "-"),
    ]
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    plotted = 0
    for label, legend, color, ls in pairs:
        column = f"{label}_mobilized"
        if column not in wide.columns:
            continue
        # 条件ごとに停止するステップが違う．pivot は足りない側を NaN で埋めるので，
        # 切り出した後に落とす — 描かない点と «値が 0» を取り違えないため．
        series = wide[["step", column]].dropna()
        if series.empty:
            continue
        ax.plot(series["step"], series[column], color=color, ls=ls, lw=2, label=legend)
        plotted += 1
    if plotted == 0:
        print("  警告: 条件別の動員系列が無いため mobilization_curves をスキップ")
        plt.close(fig)
        return
    ax.set_xlabel("時刻 t (ステップ)")
    ax.set_ylabel("動員エージェント数 mobilized")
    ax.set_title(
        "動員曲線 (代表 run): LLM コアが動員を牽引\n"
        "hybrid は pure-ABM より高い動員水準へ押し上げる",
        fontsize=12,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {out_path}")


# --------------------------------------------------------------------------- #
# レポート出力
# --------------------------------------------------------------------------- #


def _print_report(
    params: dict,
    scoped: dict[str, float],
    cells: pd.DataFrame,
    anchors: pd.DataFrame,
    bench: pd.DataFrame,
    results_dir: Path,
) -> None:
    print("=" * 78)
    print("Mou et al. (2024) HiSim — Table 2/3 + SoMoSiMu-Bench 一括再現レポート")
    mode = "mock" if params.get("mock") else "live"
    print(f"  source: {results_dir}  (mode={mode})")
    print("=" * 78)

    dataset = (params.get("datasets") or ["?"])[0]
    print(f"\n[Table 3: hybrid vs pure-ABM (dataset={dataset})]")
    print(f"  {'condition':<18}{'Bias':>8}{'Div.':>8}{'Pol.':>8}{'Mob.':>8}"
          f"{'Mob-gain':>10}{'LLM':>8}")
    for c in cells.itertuples():
        print(f"  {c.label:<18}{c.mean_final_bias:>8.3f}"
              f"{c.mean_final_diversity:>8.3f}{c.mean_final_polarization:>8.3f}"
              f"{c.mean_final_mobilization:>8.3f}{c.mean_mobilization_gain:>10.3f}"
              f"{c.mean_llm_calls:>8.1f}")

    print("\n[Table 2: SoMoSiMu-Bench 照合 (pure-ABM BC; 合成参照)]")
    for movement in dict.fromkeys(bench["movement"]) if not bench.empty else []:
        rows = bench[bench["movement"] == movement]
        n_aligned = int(rows["aligned"].sum())
        n_total = len(rows)
        ok = "OK " if n_aligned * 2 >= n_total else "off"
        source = rows["reference_source"].iloc[0]
        print(f"  {movement:<6} [{ok}] {n_aligned}/{n_total} 指標が整合 (source={source})")
        for r in rows.itertuples():
            mark = "aligned" if r.aligned else "off    "
            print(f"    {r.metric:<20} obs={r.observed:>7.3f} "
                  f"ref={r.reference:>7.3f} |d|={r.abs_error:>6.3f} "
                  f"tol={r.tolerance:>5.3f} [{mark}]")

    print("\n[論文知見アンカー (観測 vs 論文 Table 2/3)]")
    # `pass` は予約語なので itertuples では列名が潰れる．行は dict で取る．
    for _, a in anchors.iterrows():
        hi = a["target_hi"]
        hi_str = "inf" if hi is None or pd.isna(hi) else f"{hi:.3f}"
        status = "PASS" if a["pass"] else "OFF "
        print(f"  [{status}] {a['name']:<32} obs={a['observed']:.4f} "
              f"target=[{a['target_lo']:.3f},{hi_str}] paper={a['paper']}")
    print("-" * 78)
    print(f"{int(scoped.get('anchors_passed', 0))}/"
          f"{int(scoped.get('anchors_total', len(anchors)))} アンカーが in-band")
    print(f"{int(scoped.get('bench_aligned', 0))}/"
          f"{int(scoped.get('bench_total', 0))} bench 指標が整合帯")
    print("(中核知見: pure-ABM は LLM 0 呼び出し / BC・HK は合意・SJ・Lorenz は二極化 / "
          "LLM コアが動員を牽引 / bench 参照は合成 = ground-truth ではない)")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hisim-tools reproduce",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results-dir", "--results_dir", default=None,
                        help="`hisim reproduce` の run ディレクトリ "
                             "(省略時は runvault path --latest --subcommand reproduce)")
    parser.add_argument("--results-root", "--results_root", default="results",
                        help="runvault の results ルート (default: results)")
    parser.add_argument("--experiment", default=EXPERIMENT,
                        help=f"runvault の experiment 名 (default: {EXPERIMENT})")
    parser.add_argument("--output-dir", "--output_dir", default=None,
                        help="図の保存先 (既定: <experiment>/figures/<run_slug>)")
    parser.add_argument("--run", action="store_true",
                        help="先に Rust バイナリ (reproduce) を実行する．")
    parser.add_argument("--mock", action="store_true",
                        help="--run 時にライブ LLM を使わず mock で駆動する．")
    parser.add_argument("--quick", action="store_true",
                        help="--run 時に軽量モードで実行する (動作確認用)．")
    parser.add_argument("--seed", type=int, default=42, help="--run 時のシード基点．")
    parser.add_argument("--cargo-output-dir", "--cargo_output_dir", default="results",
                        help="--run 時に cargo の --output-dir へ渡すパス (既定: results)．")
    parser.add_argument("--json", action="store_true", help="JSON 形式で要約を出力する．")
    args = parser.parse_args(argv)

    if args.run:
        _run_binary(mock=args.mock, quick=args.quick, seed=args.seed,
                    output_dir=args.cargo_output_dir)

    results_dir = Path(
        args.results_dir
        or runvault_path(args.experiment, args.results_root, subcommand="reproduce")
    )
    if not (results_dir / "metrics.csv").exists():
        print(f"エラー: metrics.csv が見つかりません: {results_dir}\n"
              f"  先に `hisim-tools reproduce --run --mock` を実行してください．",
              file=sys.stderr)
        return 1

    params = config_parameters(results_dir) or {}
    scoped = run_scope_metrics(results_dir)
    abms = params.get("abm_values") or ABM_ORDER
    cells = cell_table(scoped, list(abms))
    anchors = _events(results_dir, ANCHOR_EVENT)
    bench = _events(results_dir, BENCH_EVENT)

    if args.json:
        payload = {
            "source": str(results_dir),
            "parameters": params,
            "run_scope_metrics": scoped,
            "cells": cells.to_dict(orient="records"),
            "anchors": anchors.to_dict(orient="records"),
            "bench_alignment": bench.to_dict(orient="records"),
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
        return 0

    _print_report(params, scoped, cells, anchors, bench, results_dir)

    out_dir = Path(args.output_dir) if args.output_dir else Path(figures_dir(results_dir))
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n[図] 出力先: {out_dir}")
    dataset = (params.get("datasets") or ["?"])[0]
    _table3(cells, dataset, out_dir / "table3_hybrid_vs_pureabm.png")
    _bench_alignment(bench, out_dir / "bench_alignment.png")
    _mobilization_curves(
        metrics_wide(results_dir / "metrics.csv"),
        out_dir / "mobilization_curves.png",
    )

    print("-" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
