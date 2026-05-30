#!/usr/bin/env python3
"""reproduce_paper.py — Mou et al. (2024) HiSim 見出し的知見の一括再現レポート + 図．

Rust の `hisim reproduce` が書き出す `reproduce_summary.json` (Table 3 の hybrid vs
pure-ABM 行列・SoMoSiMu-Bench 照合・論文知見アンカー) と条件別 `metrics_<label>.csv`
を読み，論文 §5 / Table 2/3 の中心的知見を 3 つの図で可視化しつつ PASS/off テーブルを
表示する:

    1. table3_hybrid_vs_pureabm.png
       ABM 種別 (bc/hk/sj/lorenz) ごとに hybrid (LLM コア + ABM 周辺) と pure-ABM
       (core-ratio 0) の最終 Polarization・正規化 Mobilization を対比する棒グラフ．
       «BC/HK は合意 (低分極) / SJ/Lorenz は二極化» と «LLM コアが動員を牽引» を示す．
    2. bench_alignment.png
       SoMoSiMu-Bench 照合 (#MeToo / RoeOverturned / BlackLivesMatter)．運動別に
       観測 (シミュレータ) vs 較正済み合成参照の運動指標を並べ，整合帯を可視化する．
    3. mobilization_curves.png
       代表 run の動員曲線時系列を hybrid vs pure-ABM で重ね描き (BC)．LLM コアが
       call-to-action を発信して動員を押し上げることを時系列で対比する．

`--run` を付けると先に Rust バイナリ (`cargo run --release -- reproduce`) を実行して
最新結果を生成する．サンドボックス・CI では `--mock` も付けてライブ LLM を回避する
(pure-ABM 条件は core-ratio 0 で LLM を一切呼ばない)．

Usage:
    uv run hisim-tools reproduce --run --mock          # mock で一括再現 + 図
    uv run hisim-tools reproduce --run --mock --quick  # 軽量版 (動作確認用)
    uv run hisim-tools reproduce                        # 既存 results/latest を可視化
    uv run hisim-tools reproduce --results-dir results/reproduce_20260530_000000
    uv run hisim-tools reproduce --json

Outputs:
    {results_dir}/figures/{table3_hybrid_vs_pureabm,bench_alignment,mobilization_curves}.png
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

from socsim_tools.io import resolve_results_dir

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


def _load_summary(results_dir: Path) -> dict:
    path = results_dir / "reproduce_summary.json"
    if not path.exists():
        raise FileNotFoundError(
            f"reproduce_summary.json が見つかりません: {path}\n"
            f"  先に `hisim-tools reproduce --run --mock` を実行してください．"
        )
    with path.open(encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# 描画
# --------------------------------------------------------------------------- #


def _table3(summary: dict, out_path: Path) -> None:
    """ABM 別 hybrid vs pure-ABM の最終 Polarization・Mobilization 棒グラフ．"""
    cells = summary["table3_hybrid_vs_pureabm"]
    hybrid = {c["abm"]: c for c in cells if c["regime"] == "hybrid"}
    pure = {c["abm"]: c for c in cells if c["regime"] == "pure-abm"}
    abms = [a for a in ABM_ORDER if a in pure]

    x = np.arange(len(abms))
    w = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=COLOR_BG)
    fig.suptitle(
        f"Mou et al. (2024) HiSim — Table 3: hybrid vs pure-ABM "
        f"(dataset={summary.get('table3_dataset', '?')})",
        fontsize=13,
    )

    ax = axes[0]
    ax.set_facecolor(COLOR_BG)
    ax.bar(x - w / 2, [pure[a]["mean_final_polarization"] for a in abms], w,
           color=COLOR_PUREABM, label="pure-ABM (core-ratio 0)")
    ax.bar(x + w / 2, [hybrid[a]["mean_final_polarization"] for a in abms], w,
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
    ax.bar(x - w / 2, [pure[a]["mean_final_mobilization"] for a in abms], w,
           color=COLOR_PUREABM, label="pure-ABM (core-ratio 0)")
    ax.bar(x + w / 2, [hybrid[a]["mean_final_mobilization"] for a in abms], w,
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


def _bench_alignment(summary: dict, out_path: Path) -> None:
    """SoMoSiMu-Bench 照合: 運動別の観測 vs 参照 運動指標 (整合帯)．"""
    comps = summary["bench_comparisons"]
    if not comps:
        print("  警告: bench_comparisons が空のため bench_alignment をスキップ")
        return
    metrics = [r["metric"] for r in comps[0]["rows"]]
    n = len(comps)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 5), facecolor=COLOR_BG, squeeze=False)
    fig.suptitle(
        "Mou et al. (2024) HiSim — Table 2: SoMoSiMu-Bench 照合 "
        "(観測 vs 較正済み合成参照)",
        fontsize=13,
    )
    y = np.arange(len(metrics))
    h = 0.38
    for j, c in enumerate(comps):
        ax = axes[0][j]
        ax.set_facecolor(COLOR_BG)
        obs = [r["observed"] for r in c["rows"]]
        ref = [r["reference"] for r in c["rows"]]
        ax.barh(y - h / 2, obs, h, color=COLOR_OBS, label="observed (sim)")
        ax.barh(y + h / 2, ref, h, color=COLOR_REF, label="reference (calibrated)")
        ax.set_yticks(y)
        ax.set_yticklabels(metrics, fontsize=8)
        ax.invert_yaxis()
        ok = c["n_aligned"]
        tot = c["n_total"]
        ax.set_title(f"{c['movement']}  ({ok}/{tot} 整合)", fontsize=11)
        ax.set_xlabel("指標値")
        if j == 0:
            ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3, axis="x")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {out_path}  (参照は較正済み合成; 実 bench データではない)")


def _mobilization_curves(results_dir: Path, out_path: Path) -> None:
    """hybrid vs pure-ABM の動員曲線時系列 (代表 run; BC)．"""
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
        path = results_dir / f"metrics_{label}.csv"
        if not path.exists():
            continue
        long_df = pd.read_csv(path)
        wide = long_df.pivot_table(index="t", columns="metric", values="value")
        if "mobilized" not in wide:
            continue
        ax.plot(wide.index, wide["mobilized"], color=color, ls=ls, lw=2, label=legend)
        plotted += 1
    if plotted == 0:
        print("  警告: metrics_<label>.csv が無いため mobilization_curves をスキップ")
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


def _print_report(summary: dict, results_dir: Path) -> None:
    print("=" * 78)
    print("Mou et al. (2024) HiSim — Table 2/3 + SoMoSiMu-Bench 一括再現レポート")
    print(f"  source: {results_dir}  (mode={summary.get('mode', '?')})")
    print("=" * 78)

    print(f"\n[Table 3: hybrid vs pure-ABM (dataset={summary.get('table3_dataset', '?')})]")
    print(f"  {'condition':<18}{'Bias':>8}{'Div.':>8}{'Pol.':>8}{'Mob.':>8}"
          f"{'Mob-gain':>10}{'LLM':>8}")
    for c in summary["table3_hybrid_vs_pureabm"]:
        print(f"  {c['label']:<18}{c['mean_final_bias']:>8.3f}"
              f"{c['mean_final_diversity']:>8.3f}{c['mean_final_polarization']:>8.3f}"
              f"{c['mean_final_mobilization']:>8.3f}{c['mean_mobilization_gain']:>10.3f}"
              f"{c['mean_llm_calls']:>8.1f}")

    print("\n[Table 2: SoMoSiMu-Bench 照合 (pure-ABM BC; 較正済み合成参照)]")
    for c in summary["bench_comparisons"]:
        ok = "OK " if c["n_aligned"] * 2 >= c["n_total"] else "off"
        print(f"  {c['movement']:<6} [{ok}] {c['n_aligned']}/{c['n_total']} 指標が整合 "
              f"(source={c['reference_source']})")
        for r in c["rows"]:
            mark = "aligned" if r["aligned"] else "off    "
            print(f"    {r['metric']:<20} obs={r['observed']:>7.3f} "
                  f"ref={r['reference']:>7.3f} |d|={r['abs_error']:>6.3f} "
                  f"tol={r['tolerance']:>5.3f} [{mark}]")

    print("\n[論文知見アンカー (観測 vs 論文 Table 2/3)]")
    n_pass = 0
    for a in summary["anchors"]:
        hi = a["target_hi"]
        hi_str = "inf" if hi is None or hi > 1e30 else f"{hi:.3f}"
        status = "PASS" if a["pass"] else "OFF "
        if a["pass"]:
            n_pass += 1
        print(f"  [{status}] {a['name']:<58} obs={a['observed']:.4f} "
              f"target=[{a['target_lo']:.3f},{hi_str}] paper={a['paper']}")
    print("-" * 78)
    print(f"{n_pass}/{len(summary['anchors'])} アンカーが in-band")
    print(f"{summary.get('bench_aligned', 0)}/{summary.get('bench_total', 0)} "
          f"bench 指標が整合帯")
    print("(中核知見: pure-ABM は LLM 0 呼び出し / BC・HK は合意・SJ・Lorenz は二極化 / "
          "LLM コアが動員を牽引 / bench 参照は較正済み合成 = ground-truth ではない)")


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
                        help="reproduce_summary.json のあるディレクトリ (既定: results/latest)")
    parser.add_argument("--output-dir", "--output_dir", default=None,
                        help="図の保存先 (既定: {results_dir}/figures)")
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

    results_dir = resolve_results_dir(args.results_dir)
    try:
        summary = _load_summary(results_dir)
    except FileNotFoundError as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    _print_report(summary, results_dir)

    out_dir = Path(args.output_dir) if args.output_dir else results_dir / "figures"
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n[図] 出力先: {out_dir}")
    _table3(summary, out_dir / "table3_hybrid_vs_pureabm.png")
    _bench_alignment(summary, out_dir / "bench_alignment.png")
    _mobilization_curves(results_dir, out_dir / "mobilization_curves.png")

    print("-" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
