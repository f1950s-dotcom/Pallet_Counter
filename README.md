# 映像解析によるPパレ受払い計数システム — PoC最小版

要件定義書 v0.2（映像解析によるPパレ受払い計数システム）に基づく、**エンドツーエンドのPoC最小実装**です。

> **通過 → 検出 → 追跡 → ライン通過/方向判定 → 状態分類 → 枚数算出 → 計数イベント確定 → 在庫反映 → 確認/補正 → CSV出力**
> という要件 3.6 のフロー全体を、1本の合成サンプル映像で「動く形」に通しています。

実カメラ映像・学習データが無い段階でも全体が動作するよう、検出は **モック検出器（合成映像の色域抽出）** を既定とし、**学習済みYOLOモデルの組み込み口**を別途用意しています（[ML の扱い](#ml-の扱い実機への道筋)参照）。

---

## クイックスタート

```bash
pip install -r requirements.txt          # numpy, opencv-python-headless

# 1) 合成サンプル映像を生成（6シーン：空/段積み/ダブルディープ/高段/単一/非Pパレ）
python tools/generate_sample_video.py

# 2) パイプラインで計数（期首在庫100で理論在庫まで算出）
python -m palletcounter run data/videos/sample.mp4 --opening 100

# 3) ダッシュボード＋確認/補正UIを起動（http://127.0.0.1:8000/）
python -m palletcounter dashboard --opening 100

# 4) 計数結果を中間データ(CSV)として出力（8章の手入力補助）
python -m palletcounter export -o pallet_events.csv

# （任意）検出・計数を重畳した検証用動画を書き出し
python tools/annotate_video.py data/videos/sample.mp4 -o data/videos/annotated.mp4
```

サンプル映像での計数結果（正解と一致）:

| シーン | 方向 | 状態 | 枚数 | 確定 |
|---|---|---|---|---|
| 空フォーク | 入庫 | empty_fork | 0 | 自動 |
| 段積み3段（モードA） | 入庫 | empty_stack | 3 | 自動 |
| ダブルディープ（モードB） | 入庫 | loaded | 2 | 自動 |
| 段積み8段（モードA） | 出庫 | empty_stack | 8 | **要確認**（高段で信頼度↓） |
| 積載 単一（モードB） | 出庫 | loaded | 1 | 自動 |
| 非Pパレ通過 | 入庫 | empty_fork | 0（対象外） | 自動 |

→ 理論在庫 = 100 + (3+2) − (8+1) = **96**、確認待ち **1**。

---

## アーキテクチャ

要件 6 章の「1ドック単位ユニットを横展開する」思想に沿い、各処理段を疎結合な
モジュールに分離しています。検出器・トラッカーは同一インターフェースのまま
差し替え可能です。

```
data/videos/*.mp4 ─▶ CountingPipeline (palletcounter/pipeline.py)
                        │
   FR-02 動体トリガー ──┤  通過時のみ後段を起動
   FR-03 物体検出 ──────┤  detection/  (mock | yolo)
   FR-04 トラッキング ──┤  tracking/   (IoUトラッカー)
   FR-05 ライン/方向 ───┤  counting/   (line_crossing, 二重計数抑止)
   FR-06/07/08 分類/枚数┤  classification/ (state: モードA段数 / モードB ユニット)
   FR-09 イベント記録 ──┤  events/     (SQLite)
                        ▼
        FR-10 在庫集計   inventory/   理論在庫 = 期首 ± Σ受払い
        FR-11/12/13 UI   api/         確認/補正・ダッシュボード・CSV出力
```

### モジュールと要件の対応

| モジュール | 役割 | 主な要件 |
|---|---|---|
| `palletcounter/detection/` | 物体検出（mock / yolo の差し替え）、Pパレ/非Pパレ判別 | FR-03, FR-08 |
| `palletcounter/tracking/` | ID付与・軌跡（IoUトラッカー） | FR-04 |
| `palletcounter/counting/` | ライン通過・方向判定・二重計数抑止 | FR-05, NFR-05 |
| `palletcounter/classification/` | 状態分類と枚数算出（モードA/B） | FR-06, FR-07, 3.3, 3.5 |
| `palletcounter/events/` | 計数イベント永続化・確認/補正・CSV | FR-09, FR-11, FR-12 |
| `palletcounter/inventory/` | 理論在庫集計 | FR-10, UC-03 |
| `palletcounter/api/` | ダッシュボード・確認UI・API/CSV出力 | FR-11, FR-12, FR-13 |
| `palletcounter/pipeline.py` | 全段の統合（3.6 のフロー） | FR-01, FR-02 |
| `palletcounter/config.py` | 1ドック単位の設定（横展開の単位） | NFR-10 |

---

## 計数ロジックの中核（要件 3.3）

`StateClassifier`（`palletcounter/classification/state.py`）が、ライン通過した
フォークリフトに紐づく検出から状態を分類し、状態別ルールで枚数を出します。

- **空フォーク**（積荷なし）→ **0**
- **空パレット段積み（モードA）** → 縦方向の層数を分割計数 → **段数 ＝ 枚数**
- **荷物積載（モードB）** → 横方向の基底ユニット数 → **ユニット数**（ダブルディープ＝2 が標準）
- **Pパレ以外（nonpallet）は計数しない（0）**（FR-08 / 3.5）

### 人機ハイブリッドによる「差異ゼロ」（NFR-01）

モデル単体の常時99%は条件次第でハードルが高いため、**信頼度しきい値**で
自動確定と人確認を振り分けます。

- 信頼度 ≥ `auto_confirm_threshold`（既定0.85）→ **自動確定**
- それ未満 → **確認キュー**に積み、ダッシュボードで人が1タップ確定/補正

高段積み（≥7段：継ぎ目判別が難しい・要件リスク#3）や非Pパレ近接など
**曖昧なパスは意図的に信頼度を下げ**、人確認へ回します。これにより
**最終登録精度＝受払い差異ゼロ**を担保しつつ、自動確定率（省力化KPI）を最大化します。

### 二重計数抑止（NFR-05）

`LineCrossingDetector` がトラックごとに最後の計数フレームを記録し、
クールダウン中の往復・滞留・後退による重複計数を抑止します。

---

## ML の扱い（実機への道筋）

本PoCの既定検出器は **`MockDetector`**：合成映像が既知色で描いたオブジェクトを
HSV色域＋輪郭抽出で検出し、「画素から検出 → 計数」の流れを依存ライブラリ最小で
実機相当に再現します。**パイプラインの正しさ（追跡・通過・分類・在庫・UI）を
実データ無しで検証**するためのものです。

実機では要件 7 章の通り、Pパレ・フォークリフト・積荷を**自社データで学習**した
検出モデルが必要です。その組み込み口が **`YoloDetector`**（`detection/yolo.py`）で、
学習済み `.pt` を渡し `class_map` でラベルを対応付ければ、

```bash
pip install ultralytics
python -m palletcounter run <実映像> --detector yolo
```

のように**コード変更なしで差し替え**られます（COCO等の汎用重みにはPパレ
クラスが無いため、本番には自社学習モデルが前提）。

### 横展開（NFR-10）

1ドック＝1 `DockConfig`。`site_id` / 計数ライン位置 / 入出庫の向き / しきい値を
設定で差し替えるだけで、別ドック・別倉庫へ展開できます。

---

## テスト

```bash
pip install pytest
python -m pytest -q
```

- `test_line_crossing.py` … 方向判定・二重計数抑止（FR-05 / NFR-05）
- `test_classification.py` … 状態分類・枚数算出・非Pパレ除外・高段の要確認化
- `test_store_inventory.py` … イベント保存・確認/補正・理論在庫・CSV
- `test_pipeline_e2e.py` … 合成映像での計数が正解と一致する統合テスト

---

## 本PoCの範囲と限界（正直な明記）

- 検出は既定でモック（合成映像前提）。**実環境の精度（NFR-01の97〜99%）は
  実データでの学習・評価が前提**で、本PoCでは扱っていません。
- 外光変動・逆光・夜間耐性（NFR-07）、ピーク物量実測（NFR-04）、
  共通受払いシステムとの直接連携（FR-14・8章）は要件上 Phase 0 / 将来 / 要確認の項目で、
  本実装では**中間データ(CSV/API)までの疎結合**にとどめています。
- トラッカーは軽量IoU実装。本番では同インターフェースで ByteTrack 等へ差し替え可能。

これらは要件定義書のスコープ・前提（9章・10章）と整合しています。
