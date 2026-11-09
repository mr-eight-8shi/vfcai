# vfcai: Boundary Function Space-based Image Representation Architecture

**vfcai** is a machine learning architecture that maps any image into a "Boundary Function Space" based on input context, utilizing this spatial representation for image reconstruction, visualization, and multi-class classification.

The core mechanism converts an image into a set of linear function parameters (boundary vectors) for each central local region ($3 \times 3$), taking into account its surrounding context ($9 \times 9$).

<img width="4017" height="1441" alt="image" src="https://github.com/user-attachments/assets/4b238738-2512-4b85-ab2e-79e6d87b5811" />

---

## 💡 Core Idea & Specifications

### 1. Mapping from Images to Local Function Space

The input image is converted into a set of local linear function parameters defined as:

$$f(x, y) = dx \cdot x + dy \cdot y + \text{offset}$$

Each patch is mapped by trained filters into the following **5-channel spatial function map**:

1. **Presence Score** ($\sigma(\text{logit})$)
2. **Direction X ($dx$)**
3. **Direction Y ($dy$)**
4. **Signed Offset (Offset / SDF)**
5. **Confidence ($\sqrt{dx^2 + dy^2}$)**

### 2. Reconstruction from Function Space

Images can be successfully reconstructed while retaining their structural integrity using the transformed local function groups (presence, direction, and offset).

---

## 🛠️ System Architecture & Pipeline

```text
[① BinaryModelTrainer]
  ├─ Independent training of boundary filters using surrounding context (9x9) and central local space (3x3)
  └─ Learning spatial boundary parameters (direction vector, signed offset, presence probability)
        │
        ▼ (Input: Mapping via 5-channel spatial function map)
[② BoundaryFieldCNN Classifier]
  ├─ Generation and pre-caching of local function space maps (float16)
  └─ Execution of a robust CNN classifier taking only the local function space as input
        │
        ▼ (Inference and evaluation)
[③ Visualization & Evaluation Pipeline]
  └─ Generation of class-wise accuracy reports and 2-in-1 function reconstruction grids

```

### Module Specifications

1. **Integrated Boundary Regression Model (`BinaryModelTrainer`)**
* **Input:** $9 \times 9 \times 3$ patches containing surrounding context.
* **Output:** Presence logits, normalized direction vectors ($v_x, v_y$), and signed boundary offsets for the central $3 \times 3$ region.


2. **5-Channel Boundary Field CNN Classifier (`BoundaryFilteredImageClassifier`)**
* A CNN classifier that takes the extracted 5-channel function maps as input.



---

## 🏆 Evaluation Results

Evaluation results of the classification task using the unified pipeline.

* **Training data:** 1,000 samples / class (20 classes total across MNIST & Fashion-MNIST)

| Dataset | Accuracy |
| --- | --- |
| **MNIST** (Classes 0–9) | **99.20%** |
| **Fashion-MNIST** (Classes 10–19) | **92.00%** |

<img width="786" height="244" alt="image" src="https://github.com/user-attachments/assets/562f6f9d-23d1-410a-8495-697f9047ac73" />

---

## 🔍 2-in-1 Function Reconstruction & Visualization

Generates a 2-in-1 grid comparing the original image against the function reconstruction map ($\tanh(dx \cdot x + dy \cdot y + \text{offset}) \times \text{presence}$) extracted from the vector field via the visualizer pipeline (`vfcai visual`).

<img width="2081" height="897" alt="boundary_visualization_2in1_2" src="https://github.com/user-attachments/assets/b5fcb70b-7732-494f-928f-2239e0e878f7" />

---

## 🔮 Future Directions & Challenges

1. **Information Compression Rate Verification**
Investigate the impact of varying stride widths on classification accuracy, exploring high-efficiency information compression mechanisms using sparse continuous function groups.
2. **Enhancement of Few-Shot Learning**
Verify high-precision function restoration and classification in low-data environments using boundary-aware filters constrained by geometric and structural priors.

---

## 🚀 Installation & Usage

### 1. Installation

Install the package in editable mode locally using Hatchling:

```bash
pip install -e .

```

### 2. CLI Usage

Launch the interactive terminal interface:

```bash
vfcai

```

Alternatively, execute specific actions directly:

```bash
# 1. Train and evaluate boundary presence/regression filters
vfcai train

# 2. Train the 5-channel boundary field CNN classifier
vfcai classify --samples 300

# 3. Evaluate the saved classifier
vfcai evaluate

# 4. Generate the 2-in-1 function reconstruction grid and visualizations
vfcai visual

# 9. Reset the workspace directory
vfcai init

```

---
---

# vfcai: 局所関数空間に基づく画像表現アーキテクチャ

**vfcai** は、画像を入力コンテキストに基づき「局所関数空間（Boundary Function Space）」へ写像し、その空間表現を用いて画像の再構成・可視化および多クラス分類を行う機械学習アーキテクチャです。

本システムは、周辺コンテキスト ($9 \times 9$) を考慮しながら、中央局所領域 ($3 \times 3$) ごとの線形関数パラメータ（境界ベクトル）群へと画像を変換します。

<img width="4017" height="1441" alt="image" src="https://github.com/user-attachments/assets/4b238738-2512-4b85-ab2e-79e6d87b5811" />
<img width="887" height="622" alt="image" src="https://github.com/user-attachments/assets/2c821c37-4ce7-49a8-8b6e-e9394a024ed9" />

---

## 💡 コアアイデアと仕様

### 1. 画像から局所関数空間への写像

入力画像は以下の局所線形関数パラメータの集合へ変換されます。

$$f(x, y) = dx \cdot x + dy \cdot y + \text{offset}$$

各パッチは学習済みフィルターによって、以下の **5チャネルの空間関数マップ** へ写像されます。

1. **存在確率 (Presence Score)** ($\sigma(\text{logit})$)
2. **方向 X ($dx$)**
3. **方向 Y ($dy$)**
4. **符号付きオフセット (Offset / SDF)**
5. **信頼度 ($\sqrt{dx^2 + dy^2}$)**

### 2. 関数空間からの再構成

変換された局所関数群（存在、方向、オフセット）から、元画像の構造を保持した状態での画像再構成が可能です。

---

## 🛠️ システムアーキテクチャとパイプライン

```text
[① BinaryModelTrainer]
  ├─ 周辺コンテキスト(9x9)と中央局所空間(3x3)を用いた境界フィルターの訓練
  └─ 空間的な境界パラメータ（方向ベクトル、符号付きオフセット、存在確率）の学習
        │
        ▼ (入力：5チャネル空間関数マップによる写像)
[② BoundaryFieldCNN Classifier]
  ├─ 局所関数空間マップの生成とキャッシュ (float16)
  └─ 局所関数空間を入力とするCNN分類器の実行
        │
        ▼ (推論と評価)
[③ 可視化・評価パイプライン]
  └─ クラス別精度レポートおよび 2-in-1 の関数再構築グリッドの生成

```

### モジュール仕様

1. **統合境界回帰モデル (`BinaryModelTrainer`)**
* **入力:** 周辺コンテキストを含む $9 \times 9 \times 3$ パッチ。
* **出力:** 中央 $3 \times 3$ 領域の存在ロジット、正規化方向ベクトル ($v_x, v_y$)、符号付き境界オフセット。


2. **5チャネル境界フィールドCNN分類器 (`BoundaryFilteredImageClassifier`)**
* 抽出された5チャネルの関数マップを入力として受け取るCNN分類器。



---

## 🏆 評価結果

統一パイプラインを用いた分類タスクの評価結果です。

* **Training data:** 1,000 samples / class (20 classes total across MNIST & Fashion-MNIST)

| Dataset | Accuracy |
| --- | --- |
| **MNIST** (Classes 0–9) | **99.20%** |
| **Fashion-MNIST** (Classes 10–19) | **92.00%** |

<img width="786" height="244" alt="image" src="https://github.com/user-attachments/assets/562f6f9d-23d1-410a-8495-697f9047ac73" />

---

## 🔍 2-in-1 関数再構築と可視化

ビジュアライザーパイプライン (`vfcai visual`) により抽出されたベクトル場から、元画像と関数再構築マップ ($\tanh(dx \cdot x + dy \cdot y + \text{offset}) \times \text{presence}$) を対比させた 2-in-1 グリッドを生成します。

<img width="2081" height="897" alt="boundary_visualization_2in1_2" src="https://github.com/user-attachments/assets/b5fcb70b-7732-494f-928f-2239e0e878f7" />
<img width="887" height="622" alt="image" src="https://github.com/user-attachments/assets/2c821c37-4ce7-49a8-8b6e-e9394a024ed9" />

---

## 🔮 今後の課題・展望

1. **情報圧縮率の検証**
ストライド幅を変更した場合の分類精度への影響を検証し、少数の連続的な関数群による高効率な情報圧縮の仕組みを模索します。
2. **少数データ（Few-Shot）学習の強化**
幾何学的・構造的制約を持つ境界認識フィルターを利用した、少データ環境での高精度な関数復元と分類の検証。

---

## 🚀 導入・使用方法

### 1. インストール

Hatchlingを使用して、ローカルで開発（editable）モードでパッケージをインストールします：

```bash
pip install -e .

```

### 2. CLIの使用方法

インタラクティブな端末インターフェースを起動：

```bash
vfcai

```

または、特定のアクションを直接実行：

```bash
# 1. 境界の存在/回帰フィルタの訓練と評価
vfcai train

# 2. 5チャネル境界フィールドCNN分類器の訓練
vfcai classify --samples 300

# 3. 保存された分類器の評価
vfcai evaluate

# 4. 2-in-1の関数再構築グリッドと可視化の生成
vfcai visual

# 9. ワークスペースディレクトリのリセット
vfcai init

```

---
