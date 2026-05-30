"""hisim-tools — Mou et al. (2024) HiSim ツール統合 CLI．

Usage:
    hisim-tools visualize [...]
    hisim-tools visualize-sweep [...]
    hisim-tools show-experiment-settings [...]
    hisim-tools reproduce [...]

各サブコマンドに続く引数は，対応するモジュールの argparse がそのまま受け取る．
サブコマンドレベルで `--help` を付けると，そのサブコマンド自身のヘルプが表示される．

dispatcher の組み立ては共有ヘルパ `socsim_tools.cli.build_dispatcher` に委譲する
(prog 名・サブコマンド・ヘルプ文・argv ルーティングは従来と同一)．可視化/設定表示・
再現の実体 (visualize / visualize_sweep / show_experiment_settings / reproduce_paper)
は repo 固有のまま．
"""

from __future__ import annotations

from socsim_tools.cli import build_dispatcher

main = build_dispatcher(
    prog="hisim-tools",
    description="Mou et al. (2024) HiSim 大規模社会運動シミュレーション 可視化・分析ツール",
    subcommands={
        "visualize": (
            "単一実行結果 (態度時系列・動員曲線・態度分布) の可視化",
            "hisim_tools.visualize:main",
        ),
        "visualize-sweep": (
            "スイープ結果 (比率・ネット・モデル依存図) の可視化",
            "hisim_tools.visualize_sweep:main",
        ),
        "show-experiment-settings": (
            "実行結果ディレクトリの設定 (config / sweep_config / run_metadata) の表示",
            "hisim_tools.show_experiment_settings:main",
        ),
        "reproduce": (
            "Table 2/3 (hybrid vs pure-ABM) + SoMoSiMu-Bench 照合の一括再現とレポート・図",
            "hisim_tools.reproduce_paper:main",
        ),
    },
)


if __name__ == "__main__":
    main()
