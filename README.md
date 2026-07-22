# AMC MVP — Automatic Modulation Classification

A deep learning based **Automatic Modulation Classification (AMC)** system built on the RadioML2016.10a dataset. The project progressively improves from simple classical machine learning baselines to advanced deep learning architectures including LSTM, ResNet + LSTM, and the proposed **Grouped + Depthwise Separable CNN**, following recent AMC research literature.

---

## Dataset

**RadioML2016.10a** — Download from Kaggle:

https://www.kaggle.com/datasets/nolasthitnotomorrow/radioml2016-deepsigcom/data

The dataset contains:

- **11 modulation classes** — 8PSK, AM-DSB, AM-SSB, BPSK, CPFSK, GFSK, PAM4, QAM16, QAM64, QPSK, WBFM
- **20 SNR levels** — from -20 dB to +18 dB (step of 2)
- **220,000 total samples** — 1,000 samples per (class, SNR) pair
- **Each sample** — 128 I/Q pairs, shape (2, 128)

---

## ML Models — Stage by Stage

| Stage | Model | Type | Features Used | Notes |
|-------|-------|------|---------------|-------|
| **Stage 1** | Nearest Centroid | Classical ML | 4 weak features — mean and std of I and Q channels | Simplest possible baseline |
| **Stage 2** | Nearest Centroid | Classical ML | 22 strong features — amplitude, phase, differential phase, top FFT bins, cross-terms | Same model, much richer features |
| **Stage 3** | Softmax Regression | Classical ML | Same 22 strong features as Stage 2 | Trainable linear classifier with L2 regularisation |
| **Stage 4** | LSTM (PyTorch) | Deep Learning | Raw I/Q signal | 2-layer stacked LSTM |
| **Stage 5** | ResNet + LSTM | Deep Learning | Raw I/Q signal | Conv1D stem + Residual Blocks + LSTM |
| **Stage 6** | Grouped + Depthwise Separable CNN | Deep Learning | Raw I/Q signal | Lightweight convolutional architecture using grouped convolutions, depthwise separable convolutions and residual learning |

---

## Key Insight

- Stages 1–3 rely on handcrafted signal features before classification.
- Stage 4 (LSTM) automatically learns temporal features directly from raw I/Q sequences.
- Stage 5 (ResNet + LSTM) improves feature extraction by combining convolutional residual learning with sequence modelling.
- Stage 6 (Grouped + Depthwise Separable CNN) introduces grouped convolutions and depthwise separable convolutions to reduce computational complexity while maintaining strong feature extraction through residual learning.

---

## Proposed Stage 6 Architecture

The proposed Stage 6 architecture consists of:

- Initial Conv1D feature extractor
- Residual learning blocks
- Grouped convolution layers
- Depthwise separable convolution layers
- Global Average Pooling
- Fully connected classifier

The architecture learns directly from raw I/Q signals while significantly reducing computational cost compared to conventional convolutional neural networks.

---

## Target

- **Stage 4 (LSTM):** ≥90% accuracy at high SNR (Reference [15])
- **Stage 5 (ResNet + LSTM):** ≥92% accuracy at high SNR (Reference [16])
- **Stage 6 (Grouped + Depthwise Separable CNN):** Competitive high-SNR performance using an efficient lightweight CNN architecture.

---

# Running the Project

## Option A — VS Code (Local)

### Requirements

- Python 3.9 or higher
- pip

### Step 1 — Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/summer_intern_project.git
cd summer_intern_project
```

### Step 2 — Install dependencies

```bash
pip install numpy torch
```

### Step 3 — Download the dataset

Download `RML2016.10a_dict.pkl` from the Kaggle link above and place it inside the `dataset/` folder.

### Step 4 — Run

**Full run — all implemented stages including LSTM, ResNet + LSTM and Grouped + Depthwise Separable CNN (~40–50 min).**

```bash
python -u mvp.py --dataset "./dataset/RML2016.10a_dict.pkl"
```

**Skip LSTM — run only Stages 1–3 (fast, ~1–2 min):**

```bash
python -u mvp.py --dataset "./dataset/RML2016.10a_dict.pkl" --skip-lstm
```

**Run only the proposed Stage 6 architecture:**

```bash
python -u mvp.py --dataset "./dataset/RML2016.10a_dict.pkl" --skip-lstm --skip-stage5
```

**Run on synthetic data — no dataset needed, for testing the script:**

```bash
python mvp.py
```

Results are saved to `reports/numpy_mvp/metrics.json` after every run.

---

## Option B — Google Colab (Recommended for LSTM)

Colab provides a free NVIDIA T4 GPU which significantly reduces training time.

### Step 1 — Open a new notebook

Go to https://colab.research.google.com and create a new notebook.

### Step 2 — Enable GPU

Go to **Runtime → Change runtime type → T4 GPU → Save**

### Step 3 — Upload the dataset via Google Drive

Upload `RML2016.10a_dict.pkl` to your Google Drive.

### Step 4 — Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 5 — Upload mvp.py

Upload `mvp.py` using the Colab file panel.

### Step 6 — Confirm your files

```python
!ls /content/
!ls "/content/drive/My Drive/" | grep RML
```

### Step 7 — Install dependencies

```python
!pip install torch -q
```

### Step 8 — Run

```python
!python -u /content/mvp.py --dataset "/content/drive/My Drive/RML2016.10a_dict.pkl"
```

or

```python
!python -u /content/mvp.py --dataset "/content/drive/My Drive/mvp/RML2016.10a_dict.pkl"
```

### Keep the session alive

```javascript
%%javascript
function ClickConnect(){
  console.log("Keeping alive...");
  document.querySelector("colab-toolbar-button#connect").click()
}
setInterval(ClickConnect, 60000)
```

---

## Team Workflow (Git)

### First time setup

```bash
git clone https://github.com/YOUR_USERNAME/amc-mvp.git
cd amc-mvp
```

### Daily workflow

```bash
git pull
git add .
git commit -m "describe your change clearly"
git push
```

**Note:** The dataset file (`*.pkl`) is excluded by `.gitignore` and must be downloaded separately by every team member.

---

## Observations

- Deep learning models significantly outperform classical machine learning baselines.
- LSTM demonstrates strong temporal feature learning directly from raw I/Q signals.
- ResNet + LSTM improves feature extraction through residual convolutional learning.
- The proposed **Grouped + Depthwise Separable CNN** provides competitive classification performance while maintaining a lightweight architecture with approximately **1 million trainable parameters**.
- The model achieves strong performance at higher SNR levels while reducing computational complexity through grouped and depthwise separable convolutions.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Data generation, feature extraction, classical ML stages |
| `torch` | PyTorch — Deep Learning Models (Stages 4–6) |

---

