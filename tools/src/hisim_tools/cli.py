"""hisim-tools — Mou et al. (2024) HiSim ツール統合 CLI．

Usage:
    hisim-tools visualize [...]
    hisim-tools visualize-sweep [...]
    hisim-tools show-experiment-settings [...]
    hisim-tools reproduce [...]

各サブコマンドに続く引数は，対応するモジュールの argparse がそのまま受け取る．
サブコマンドレベルで `--help` を付けると，そのサブコマンド自身のヘルプが表示される．
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="hisim-tools",
        description="Mou et al. (2024) HiSim 大規模社会運動シミュレーション 可視化・分析ツール",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "visualize",
        help="単一実行結果 (態度時系列・動員曲線・態度分布) の可視化",
        add_help=False,
    )
    subparsers.add_parser(
        "visualize-sweep",
        help="スイープ結果 (比率・ネット・モデル依存図) の可視化",
        add_help=False,
    )
    subparsers.add_parser(
        "show-experiment-settings",
        help="実行結果ディレクトリの設定 (config / sweep_config / run_metadata) の表示",
        add_help=False,
    )
    subparsers.add_parser(
        "reproduce",
        help="SoMoSiMu-Bench 照合・Table 3 再現 (Phase 3; 未実装スタブ)",
        add_help=False,
    )

    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in {"-h", "--help"}:
        parser.parse_args(argv)
        return

    command = argv[0]
    rest = argv[1:]
    if command == "visualize":
        from hisim_tools.visualize import main as run_main

        run_main(rest)
    elif command == "visualize-sweep":
        from hisim_tools.visualize_sweep import main as run_main

        run_main(rest)
    elif command == "show-experiment-settings":
        from hisim_tools.show_experiment_settings import main as run_main

        run_main(rest)
    elif command == "reproduce":
        from hisim_tools.reproduce_paper import main as run_main

        run_main(rest)
    else:
        # 未知のコマンドは argparse のエラーメッセージに委ねる
        parser.parse_args(argv)


if __name__ == "__main__":
    main()
