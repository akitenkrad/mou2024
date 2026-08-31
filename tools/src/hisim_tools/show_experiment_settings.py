"""hisim-tools show-experiment-settings — 実行結果の設定表示．

run ディレクトリの `config.json` (runvault の封筒; 条件は `parameters` の下) を読み，
実行時に使われた全パラメータを整形表示する．`run.json` の `llm` ブロックと
`metrics.csv` の run スコープ指標 (LLM 呼び出し数・cache-hit) も併せて表示する．
移行前の `results/{timestamp}/config.json` (平の JSON)・`{ts}_sweep/sweep_config.json`・
`run_metadata.json` もそのまま読める．

--results-dir を省略すると `runvault path --experiment hisim --latest` が返す run を
対象にする (`runvault` が PATH にある必要がある)．

Usage:
    hisim-tools show-experiment-settings
    hisim-tools show-experiment-settings --results-dir "$(runvault path --experiment hisim --latest)"
    hisim-tools show-experiment-settings --json

設定テーブルは複合行 (`BA m / WS k,β / ER p`・`ABM α / ε`) を含み，sweep のグリッド
テーブルと `--json` の `kind` フィールドも hisim 固有なので，これらのレンダラは本
モジュールにローカルのまま残す．
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from runvault.read import config_parameters, load_run_meta, run_scope_metrics, runvault_path

# runvault の experiment 名 (Rust 側 record::EXPERIMENT と揃える)．
EXPERIMENT = "hisim"

#: sweep 親の設定にだけ現れるキー．これがあればグリッドの表として描く．
SWEEP_MARKER = "core_ratio_values"


def _find_config_file(results_dir: Path) -> Path:
    """`config.json` (runvault の封筒 / legacy の flat) か legacy の `sweep_config.json`．"""
    for name in ("config.json", "sweep_config.json"):
        path = results_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(
        f"設定ファイルが見つかりません: {results_dir}\n"
        f"  期待されるファイル: config.json (runvault の封筒 / legacy の flat) "
        f"または sweep_config.json (legacy の sweep)"
    )


def _legacy_run_metadata(results_dir: Path) -> dict | None:
    """移行前の `run_metadata.json` (あれば)．

    runvault の run にはこのファイルが無い — 同じ内容は `run.json` の `llm` ブロックと
    `metrics.csv` の run スコープ指標が持つ．
    """
    path = results_dir / "run_metadata.json"
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def render_run_config(cfg: dict, source: Path) -> str:
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("実行設定 (run)")
    lines.append("=" * 70)
    lines.append(f"設定ファイル: {source}")
    lines.append("-" * 70)
    lines.append(f"データセット     : {cfg.get('dataset', '-')}")
    lines.append(f"ABM 種別         : {cfg.get('abm', '-')}")
    lines.append(f"コア比率         : {cfg.get('core_ratio', '-')}")
    lines.append(f"エージェント数 N : {cfg.get('n_agents', '-')}")
    lines.append(f"タイムステップ T : {cfg.get('steps', '-')}")
    lines.append(f"ネットワーク     : {cfg.get('network', '-')}")
    lines.append(f"BA m / WS k,β / ER p : {cfg.get('ba_m', '-')} / "
                 f"{cfg.get('ws_k', '-')},{cfg.get('ws_beta', '-')} / {cfg.get('er_p', '-')}")
    lines.append(f"ABM α / ε        : {cfg.get('alpha', '-')} / {cfg.get('epsilon', '-')}")
    lines.append(f"動員しきい値     : {cfg.get('mobilization_threshold', '-')}")
    lines.append(f"LLM 予算         : {cfg.get('llm_budget', '-')}")
    lines.append(f"シード (コア)    : {cfg.get('seed', '-')}")
    lines.append(f"LLM 温度         : {cfg.get('llm_temperature', '-')}")
    lines.append(f"LLM seed         : {cfg.get('llm_seed', '-')}")
    lines.append("=" * 70)
    return "\n".join(lines)


def render_sweep_config(cfg: dict, source: Path) -> str:
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("実行設定 (sweep)")
    lines.append("=" * 70)
    lines.append(f"設定ファイル: {source}")
    lines.append("-" * 70)
    lines.append(f"データセット     : {cfg.get('dataset', '-')}")
    crs = cfg.get("core_ratio_values", [])
    lines.append(f"コア比率         : {', '.join(str(x) for x in crs)}")
    lines.append(f"ABM 種別         : {', '.join(cfg.get('abm_values', []))}")
    lines.append(f"ネットワーク     : {', '.join(cfg.get('network_values', []))}")
    lines.append(f"エージェント数   : {cfg.get('n_agents', '-')}")
    lines.append(f"タイムステップ T : {cfg.get('steps', '-')}")
    lines.append(f"試行数 runs      : {cfg.get('runs', '-')}")
    lines.append(f"シード基点       : {cfg.get('seed', '-')}")
    lines.append(f"LLM 温度         : {cfg.get('llm_temperature', '-')}")
    lines.append(f"LLM seed         : {cfg.get('llm_seed', '-')}")
    lines.append("=" * 70)
    return "\n".join(lines)


def render_llm_block(meta: dict, scoped: dict[str, float]) -> str:
    """`run.json` の `llm` ブロックと LLM 呼び出しの内訳．

    endpoint と «決定論の注記» は旧 `run_metadata.json` にあったが持ち込まない．
    前者は provider の分類に使われるだけで run の同一性には効かず，後者は設計の説明
    (docs/architecture.ja.md) であって run ごとの記録ではない．
    """
    llm = meta.get("llm") or {}
    lines: list[str] = []
    lines.append("")
    lines.append("LLM 実行メタデータ (run.json の llm ブロック / metrics.csv)")
    lines.append("-" * 70)
    lines.append(f"プロバイダ       : {llm.get('provider', '-')}")
    lines.append(f"モデル           : {llm.get('model_snapshot', '-')}")
    lines.append(f"温度             : {llm.get('temperature', '-')}")
    rng = meta.get("rng") or {}
    lines.append(f"master_seed      : {rng.get('master_seed', '-')}")
    if "llm_calls" in scoped:
        lines.append(f"呼び出し総数     : {int(scoped['llm_calls'])}")
    if "llm_cache_hits" in scoped:
        lines.append(f"cache-hit        : {int(scoped['llm_cache_hits'])}")
    if "llm_cache_hit_rate" in scoped:
        lines.append(f"cache-hit 率     : {scoped['llm_cache_hit_rate'] * 100:.1f}%")
    lines.append("=" * 70)
    return "\n".join(lines)


def render_legacy_run_metadata(meta: dict) -> str:
    """移行前の `run_metadata.json` の表示 (legacy な results/ 用)．"""
    lines: list[str] = []
    lines.append("")
    lines.append("LLM 実行メタデータ (run_metadata.json; 移行前の run)")
    lines.append("-" * 70)
    lines.append(f"プロバイダ       : {meta.get('provider', '-')}")
    lines.append(f"モデル           : {meta.get('llm_model', '-')}")
    lines.append(f"endpoint         : {meta.get('llm_endpoint', '-')}")
    lines.append(f"温度             : {meta.get('llm_temperature', '-')}")
    lines.append(f"seed             : {meta.get('llm_seed', '-')}")
    lines.append(f"コア比率         : {meta.get('core_ratio', '-')}")
    lines.append(f"呼び出し総数     : {meta.get('total_calls', '-')}")
    lines.append(f"cache-hit        : {meta.get('cache_hits', '-')}")
    rate = meta.get("cache_hit_rate")
    if rate is not None:
        lines.append(f"cache-hit 率     : {rate * 100:.1f}%")
    note = meta.get("determinism_note")
    if note:
        lines.append("-" * 70)
        lines.append(f"注記: {note}")
    lines.append("=" * 70)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hisim-tools show-experiment-settings",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        "--results_dir",
        default=None,
        help="run ディレクトリ (省略時は runvault path --latest)",
    )
    parser.add_argument(
        "--results-root",
        "--results_root",
        default="results",
        help="runvault の results ルート (default: results)",
    )
    parser.add_argument(
        "--experiment",
        default=EXPERIMENT,
        help=f"runvault の experiment 名 (default: {EXPERIMENT})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="表ではなく JSON 形式で出力する．",
    )
    args = parser.parse_args(argv)

    results_dir = Path(
        args.results_dir or runvault_path(args.experiment, args.results_root)
    )
    if not results_dir.exists():
        print(f"エラー: ディレクトリが存在しません: {results_dir}", file=sys.stderr)
        return 1

    try:
        cfg_path = _find_config_file(results_dir)
    except FileNotFoundError as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1
    if cfg_path.name == "sweep_config.json":
        with cfg_path.open() as f:
            cfg = json.load(f)
    else:
        cfg = config_parameters(results_dir) or {}
    kind = "sweep" if SWEEP_MARKER in cfg else "run"
    meta = load_run_meta(results_dir, required=False)
    # 指標を読むのは runvault の run だけ．legacy の metrics.csv は run スコープ行を
    # 持たず，時間軸の列名も `step` ではないので `run_scope_metrics` を通せない．
    scoped = run_scope_metrics(results_dir) if meta is not None else {}
    legacy = _legacy_run_metadata(results_dir)

    if args.json:
        payload = {
            "source": str(cfg_path),
            "kind": kind,
            "config": cfg,
            "run_meta": meta,
            "run_scope_metrics": scoped,
            "run_metadata": legacy,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    if kind == "run":
        print(render_run_config(cfg, cfg_path))
    else:
        print(render_sweep_config(cfg, cfg_path))
    if meta is not None:
        print(render_llm_block(meta, scoped))
    elif legacy is not None:
        print(render_legacy_run_metadata(legacy))
    return 0


if __name__ == "__main__":
    sys.exit(main())
