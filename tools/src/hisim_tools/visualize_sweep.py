#!/usr/bin/env python3
"""
visualize_sweep.py — Mou et al. (2024) HiSim スイープ結果 可視化スクリプト

results/{ts}_sweep/sweep_summary.csv を読み，コア比率・ネットワーク構造・ABM
モデルが最終マクロ指標 (分極化・動員・多様性・コア影響) に与える依存を可視化する．

(1) コア比率依存: core-ratio → 最終 polarization / mobilized (ABM 種別ごと)
(2) ABM モデル別の最終分極化 (棒グラフ)
(3) ネットワーク構造別の最終コア影響 (BA がコア影響を増幅するか)

Usage:
    uv run hisim-tools visualize-sweep
    uv run hisim-tools visualize-sweep --results_dir results/20260525_103000_sweep

Outputs:
    output_dir/
    └── sweep_dependence.png
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = "Hiragino Sans"

COLOR_BG = "#FAFAF8"
PALETTE = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]


def load_summary(results_dir: str) -> pd.DataFrame:
    path = os.path.join(results_dir, "sweep_summary.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"sweep_summary.csv が見つかりません: {path}")
    return pd.read_csv(path)


def save_sweep_dependence(df: pd.DataFrame, out_path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=COLOR_BG)
    fig.suptitle("Mou et al. (2024) HiSim — 感度分析 (比率 / モデル / ネットワーク依存)", fontsize=14)

    # (1) コア比率 → 最終分極化 (ABM 種別ごとに平均)
    ax = axes[0]
    ax.set_facecolor(COLOR_BG)
    for i, abm in enumerate(sorted(df["abm"].unique())):
        sub = df[df["abm"] == abm]
        grp = sub.groupby("core_ratio")["final_polarization"].mean().reset_index()
        ax.plot(grp["core_ratio"], grp["final_polarization"], marker="o",
                color=PALETTE[i % len(PALETTE)], label=f"abm={abm}")
    ax.set_xlabel("コア比率 core-ratio")
    ax.set_ylabel("最終 分極化 polarization")
    ax.set_title("コア比率依存 (Fig. 3)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (2) ABM モデル別 最終分極化 (棒)
    ax = axes[1]
    ax.set_facecolor(COLOR_BG)
    grp = df.groupby("abm")["final_polarization"].mean().reset_index()
    ax.bar(grp["abm"], grp["final_polarization"],
           color=[PALETTE[i % len(PALETTE)] for i in range(len(grp))])
    ax.set_xlabel("ABM モデル")
    ax.set_ylabel("最終 分極化 polarization")
    ax.set_title("モデル別 分極化 (BC=合意 / SJ・Lorenz=二極化)")
    ax.grid(True, alpha=0.3, axis="y")

    # (3) ネットワーク構造別 コア影響
    ax = axes[2]
    ax.set_facecolor(COLOR_BG)
    grp = df.groupby("network")["final_core_influence"].mean().reset_index()
    ax.bar(grp["network"], grp["final_core_influence"],
           color=[PALETTE[i % len(PALETTE)] for i in range(len(grp))])
    ax.set_xlabel("ネットワーク構造")
    ax.set_ylabel("最終 コア影響 core_influence")
    ax.set_title("ネットワーク依存 (BA がコア影響を増幅)")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {out_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="hisim-tools visualize-sweep",
        description="Mou et al. (2024) HiSim スイープ結果 可視化スクリプト",
    )
    p.add_argument(
        "--results_dir",
        "--results-dir",
        default="results/latest",
        help="sweep 出力ディレクトリ (default: results/latest)",
    )
    p.add_argument(
        "--output_dir",
        "--output-dir",
        default=None,
        help="図の保存先 (default: {results_dir}/figures)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out_dir = args.output_dir if args.output_dir else os.path.join(args.results_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)

    print("=== Mou et al. (2024) HiSim スイープ結果 可視化 ===")
    print(f"結果:   {args.results_dir}")
    print(f"出力先: {out_dir}")
    print("-----------------------------------------")

    df = load_summary(args.results_dir)
    print(f"[1/1] 依存図を保存中 ... ({len(df)} 行)")
    save_sweep_dependence(df, os.path.join(out_dir, "sweep_dependence.png"))

    print("-----------------------------------------")
    print("完了．出力ファイル一覧:")
    for f in sorted(os.listdir(out_dir)):
        size_kb = os.path.getsize(os.path.join(out_dir, f)) / 1024
        print(f"  {f:35s} ({size_kb:6.1f} KB)")


if __name__ == "__main__":
    main()
