#!/usr/bin/env python3
"""スイープの «1 行 1 試行» の表．

run ディレクトリの探し方そのものは `runvault.read` にある．ここに残るのは HiSim 固有の
部分だけ — どの列を持つ表なのか (`network` / `abm` / `core_ratio` / `final_*`) である．
モデルの話であって run ディレクトリの読み方ではないので，共通部品には置かない．
"""
from __future__ import annotations

import json
import os

import pandas as pd
from runvault.read import config_parameters, sweep_children

__all__ = ["sweep_summary_table"]

#: 条件を表すパラメータ列 (子 run の config.json の parameters から採る)．
PARAMETER_KEYS = ["network", "abm", "core_ratio"]

#: terminal イベントからそのまま採る列 (イベント側の名前 → 表の列名)．
TERMINAL_COLUMNS = {
    "final_macro_bias": "final_macro_bias",
    "final_macro_diversity": "final_macro_diversity",
    "final_mobilized": "final_mobilized",
    "final_polarization": "final_polarization",
    "final_core_influence": "final_core_influence",
    "total_llm_calls": "total_llm_calls",
    "cache_hit_rate": "cache_hit_rate",
}

COLUMNS = [
    *PARAMETER_KEYS,
    "run",
    "seed",
    "converged",
    "final_step",
    *TERMINAL_COLUMNS.values(),
    "run_dir",
]


def _terminal_events(run_dir: str) -> list[dict]:
    """子 run の `events.jsonl` の `terminal` 行を dict のまま読む．

    `runvault.read.events_table` を使わないのは，あれが `pd.DataFrame` を作る過程で
    派生シードを壊すからである．シードは u64 で，1 つでも int64 の範囲を超える値が
    あると列ごと float64 に落ち，下位の桁が消える．シードは «この試行を組み直すための
    識別子» なので，丸めた値には意味が無い．
    """
    path = os.path.join(run_dir, "events.jsonl")
    if not os.path.exists(path):
        raise SystemExit(f"エラー: events.jsonl が見つかりません: {path}")
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("schema") == "terminal":
                rows.append(event)
    return rows


def sweep_summary_table(sweep_dir: str | os.PathLike) -> pd.DataFrame:
    """1 行 1 試行のサマリ表を用意する．

    runvault ではこの表はファイルとして存在しない．sweep 親の子 run
    (`lineage.parent_run_uid` が親の `run_uid`) を集め，各子の `config.json` の
    `parameters` と `events.jsonl` の `terminal` 行 (= 試行 1 本の最終値) から組み直す．
    legacy のスイープには `sweep_summary.csv` があるのでそれを読む．

    どちらの経路でも `run_dir` 列を付けるので，呼び出し側は条件からディレクトリ名を
    組み立てなくてよい．
    """
    sweep_dir = str(sweep_dir)
    legacy = os.path.join(sweep_dir, "sweep_summary.csv")
    if os.path.exists(legacy):
        df = pd.read_csv(legacy)
        df["run_dir"] = sweep_dir
        return df

    children = sweep_children(sweep_dir)
    if not children:
        raise SystemExit(
            f"エラー: この sweep 親に紐づく子 run が見つかりません: {sweep_dir}\n"
            "  子 run は lineage.parent_run_uid で親を指します．"
            "親と子が同じ results ルートにあるか確認してください．"
        )

    rows: list[dict] = []
    seeds: list[int] = []
    for child in children:
        params = config_parameters(child) or {}
        for event in _terminal_events(child):
            row = {key: params.get(key) for key in PARAMETER_KEYS}
            # «その条件の何本目か» は unit_id (`trial-<i>`) が持つ．派生シードは
            # 条件パラメータの `seed` (基点) と名前が衝突しないよう，イベント側では
            # `trial_seed` と名乗っている．
            row["run"] = int(str(event["unit_id"]).removeprefix("trial-"))
            row["converged"] = event["outcome"] == "converged"
            row["final_step"] = event["t"]
            row.update({col: event[name] for name, col in TERMINAL_COLUMNS.items()})
            row["run_dir"] = child
            rows.append(row)
            seeds.append(int(event["trial_seed"]))

    df = pd.DataFrame(rows)
    df["seed"] = pd.array(seeds, dtype="UInt64")
    return df[COLUMNS].sort_values([*PARAMETER_KEYS, "run"]).reset_index(drop=True)
