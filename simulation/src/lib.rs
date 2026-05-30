//! Mou et al. (2024) "Unveiling the Truth and Facilitating Change: Towards
//! Agent-based Large-scale Social Movement Simulation" (HiSim) の再現実装ライブラリ．
//!
//! socsim フレームワーク上に構築した，**社会ネットワーク上の 2 階層ハイブリッド**
//! (LLM コア + ABM 周辺) の公開 API を提供する．設定 (`config`)・世界状態
//! (`world`)・意見力学 ABM (`abm`)・LLM クライアント層 (`llm`)・プロンプト生成
//! (`prompts`)・応答パース (`parse`)・更新メカニズム (`mechanisms`)・実行ドライバ
//! (`simulation`)・集計メトリクス (`metrics`) をモジュールとして公開し，バイナリ
//! (`hisim`) と統合テストの双方から利用する．
//!
//! # 二層決定論
//!
//! socsim コア層 (網生成・階層割当・周辺 ABM 更新・scheduler・指標) は seed から
//! bit 単位で決定論的である．LLM レイヤ (コア層の行動選択) は socsim の bit
//! 再現性の **外側** にあり，`socsim-llm` のキャッシュ + `temperature=0` +
//! `seed` 固定で擬似決定論化する．`core-ratio = 0.0` では LLM 呼び出しが 0 で
//! 純粋 ABM となり，全体が bit 決定論的になる．詳細は `crate::llm` を参照．

pub mod abm;
pub mod bench;
pub mod config;
pub mod llm;
pub mod mechanisms;
pub mod metrics;
pub mod parse;
pub mod prompts;
pub mod reproduce_mock;
pub mod simulation;
pub mod world;
