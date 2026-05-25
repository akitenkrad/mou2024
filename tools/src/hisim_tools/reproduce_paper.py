"""hisim-tools reproduce — SoMoSiMu-Bench 照合・Table 3 再現 (Phase 3; 未実装スタブ)．

Phase 3 で，SoMoSiMu-Bench (#Metoo / RoeOverturned / BlackLivesMatter) との照合，
ハイブリッド vs 純粋 ABM の Table 3 (ΔBias / ΔDiv. / DTW)，エコーチェンバー介入実験
(5.5節) を一括再現する予定．現状はスタブ．
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    print(
        "reproduce は Phase 3 で実装予定です (SoMoSiMu-Bench 照合・Table 3 再現・\n"
        "エコーチェンバー介入実験)．現状は run / sweep + visualize / visualize-sweep を\n"
        "使ってください．",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
