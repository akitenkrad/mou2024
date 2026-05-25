#!/usr/bin/env python3
"""
visualize.py — Mou et al. (2024) HiSim 単一実行結果 可視化スクリプト

results/latest (または --results_dir 指定先) の metrics.csv (long-format) を読み，
以下の図を生成する:
(1) 平均態度 (macro_bias) の時系列 (集団態度の偏り; Table 3 ΔBias)
(2) 動員 (mobilized) 規模の時系列 (動員曲線)
(3) 意見多様性 (macro_diversity) と分極化 (polarization) の時系列
(4) コア層の平均態度 (core_influence) の時系列 (コア→周辺ドライバ)

Usage:
    uv run hisim-tools visualize
    uv run hisim-tools visualize --results_dir results/20260525_103000
    uv run hisim-tools visualize --output_dir out

Outputs:
    output_dir/
    └── metrics_timeseries.png ← 態度・動員・多様性/分極化・コア影響
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

# --------------------------------------------------------------------------- #
# 日本語フォント設定
# --------------------------------------------------------------------------- #
plt.rcParams["font.family"] = "Hiragino Sans"

# --------------------------------------------------------------------------- #
# カラー設定
# --------------------------------------------------------------------------- #
COLOR_BG = "#FAFAF8"
COLOR_BIAS = "#2196F3"
COLOR_MOB = "#F44336"
COLOR_DIV = "#4CAF50"
COLOR_POL = "#9C27B0"
COLOR_CORE = "#FF9800"


def load_metrics(path: str) -> pd.DataFrame:
    """metrics.csv (long-format: t, metric, value) を wide-format にピボットする．"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"metrics.csv が見つかりません: {path}")
    long_df = pd.read_csv(path)
    wide = long_df.pivot_table(index="t", columns="metric", values="value").reset_index()
    wide.columns.name = None
    return wide.sort_values("t").reset_index(drop=True)


def save_metrics_timeseries(df: pd.DataFrame, out_path: str) -> None:
    """集団指標の時系列図 (4 パネル) を保存する．"""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), facecolor=COLOR_BG)
    fig.suptitle("Mou et al. (2024) HiSim — 集団態度動態の時系列", fontsize=14)
    t = df["t"]

    # (1) 平均態度 macro_bias
    ax = axes[0, 0]
    ax.set_facecolor(COLOR_BG)
    if "macro_bias" in df:
        ax.plot(t, df["macro_bias"], color=COLOR_BIAS, lw=2, marker="o", ms=3)
    ax.axhline(0.0, color="#999999", lw=0.8, ls=":")
    ax.set_xlabel("timestep t")
    ax.set_ylabel("平均態度 (macro_bias)")
    ax.set_title("集団態度の偏り (Table 3 ΔBias)")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, alpha=0.3)

    # (2) 動員 mobilized
    ax = axes[0, 1]
    ax.set_facecolor(COLOR_BG)
    if "mobilized" in df:
        ax.plot(t, df["mobilized"], color=COLOR_MOB, lw=2, marker="s", ms=3)
    ax.set_xlabel("timestep t")
    ax.set_ylabel("動員エージェント数")
    ax.set_title("動員曲線 (mobilization over time)")
    ax.grid(True, alpha=0.3)

    # (3) 多様性 + 分極化
    ax = axes[1, 0]
    ax.set_facecolor(COLOR_BG)
    if "macro_diversity" in df:
        ax.plot(t, df["macro_diversity"], color=COLOR_DIV, lw=2, marker="^", ms=3,
                label="多様性 (分散)")
    if "polarization" in df:
        ax.plot(t, df["polarization"], color=COLOR_POL, lw=1.8, ls="--", marker="x", ms=3,
                label="分極化")
    ax.set_xlabel("timestep t")
    ax.set_ylabel("多様性 / 分極化")
    ax.set_title("意見多様性と分極化 (Table 3 ΔDiv. / 5.5節)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (4) コア影響 core_influence
    ax = axes[1, 1]
    ax.set_facecolor(COLOR_BG)
    if "core_influence" in df:
        ax.plot(t, df["core_influence"], color=COLOR_CORE, lw=2, marker="d", ms=3)
    ax.axhline(0.0, color="#999999", lw=0.8, ls=":")
    ax.set_xlabel("timestep t")
    ax.set_ylabel("コア層平均態度")
    ax.set_title("コア→周辺ドライバ (core_influence; Fig. 3)")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {out_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="hisim-tools visualize",
        description="Mou et al. (2024) HiSim 単一実行結果 可視化スクリプト",
    )
    p.add_argument(
        "--results_dir",
        "--results-dir",
        default="results/latest",
        help="Rust シミュレーションの出力ディレクトリ (default: results/latest)",
    )
    p.add_argument(
        "--output_dir",
        "--output-dir",
        default=None,
        help="図の保存先ディレクトリ (default: {results_dir}/figures)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    metrics_path = os.path.join(args.results_dir, "metrics.csv")
    out_dir = args.output_dir if args.output_dir else os.path.join(args.results_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)

    print("=== Mou et al. (2024) HiSim 単一実行結果 可視化 ===")
    print(f"メトリクス: {metrics_path}")
    print(f"出力先:     {out_dir}")
    print("-----------------------------------------")

    df = load_metrics(metrics_path)
    print(f"[1/1] メトリクス時系列を保存中 ... ({len(df)} timestep)")
    save_metrics_timeseries(df, os.path.join(out_dir, "metrics_timeseries.png"))

    print("-----------------------------------------")
    print("完了．出力ファイル一覧:")
    for f in sorted(os.listdir(out_dir)):
        size_kb = os.path.getsize(os.path.join(out_dir, f)) / 1024
        print(f"  {f:35s} ({size_kb:6.1f} KB)")


if __name__ == "__main__":
    main()
