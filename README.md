# AMC MVP — Automatic Modulation Classification

A deep learning based Automatic Modulation Classification (AMC) system built on the RadioML2016.10a dataset. The project progressively improves from simple classical machine learning baselines to advanced deep learning architectures including LSTM, ResNet + LSTM, and the newly implemented Inception + ResNet model, following recent AMC research literature.

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
| **Stage 6** | Inception + ResNet | Deep Learning | Raw I/Q signal | Multi-scale Inception blocks with residual learning |

**Key Insight**

- **Stages 1–3** rely on handcrafted signal features before classification.
- **Stage 4 (LSTM)** automatically learns temporal features directly from raw I/Q sequences.
- **Stage 5 (ResNet + LSTM)** improves feature extraction by combining convolutional residual learning with sequence modelling.
- **Stage 6 (Inception + ResNet)** introduces multi-scale convolutional feature extraction using parallel kernels (1×1, 3×3, 5×5 and 7×7), allowing the network to capture modulation characteristics occurring at different temporal scales before residual refinement.

**Target**

- Stage 4 (LSTM): ≥90% accuracy at high SNR (Reference [15])
- Stage 5 (ResNet + LSTM): ≥92% accuracy at high SNR (Reference [16])
- Stage 6 (Inception + ResNet): Competitive high-SNR performance based on the architecture proposed in Reference [22]

---

## Running the Project

### Option A — VS Code (Local)

**Requirements**
- Python 3.9 or higher
- pip

**Step 1 — Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/summer_intern_project.git
cd summer_intern_project
```

**Step 2 — Install dependencies**
```bash
pip install numpy torch
```

**Step 3 — Download the dataset**

Download `RML2016.10a_dict.pkl` from the Kaggle link above and place it inside the `dataset/` folder.

**Step 4 — Run**

Full run — all implemented stages including LSTM, ResNet + LSTM and Inception + ResNet(~40-50min).
```bash
python -u mvp.py --dataset "./dataset/RML2016.10a_dict.pkl"
```

Skip LSTM — run only Stages 1–3 for a quick check (fast, ~1–2 min):
```bash
python -u mvp.py --dataset "./dataset/RML2016.10a_dict.pkl" --skip-lstm
```

Run on synthetic data — no dataset needed, for testing the script:
```bash
python mvp.py
```

**Results** are saved to `reports/numpy_mvp/metrics.json` after every run.

---

### Option B — Google Colab (Recommended for LSTM)

Colab provides a free NVIDIA T4 GPU which reduces LSTM training time from ~45 minutes to ~5 minutes.

**Step 1 — Open a new notebook**

Go to https://colab.research.google.com and create a new notebook.

**Step 2 — Enable GPU**

Go to `Runtime → Change runtime type → T4 GPU → Save`

**Step 3 — Upload the dataset via Google Drive**

Upload `RML2016.10a_dict.pkl` to your Google Drive (drag and drop at drive.google.com).
Do not use the Colab file uploader — large files get corrupted.

**Step 4 — Mount Google Drive**

Run this in a Colab cell:
```python
from google.colab import drive
drive.mount('/content/drive')
```

Sign in and allow access when prompted.

**Step 5 — Upload mvp.py**

Use the Colab file panel (folder icon on the left sidebar) to upload `mvp.py` directly. Small Python files upload fine this way.

**Step 6 — Confirm your files are there**
```python
!ls /content/
!ls "/content/drive/My Drive/" | grep RML
```

**Step 7 — Install dependencies**
```python
!pip install torch -q
```

**Step 8 — Run**
```python
!python -u /content/mvp.py --dataset "/content/drive/My Drive/RML2016.10a_dict.pkl"
```

If you placed the dataset inside a subfolder in Drive (e.g. a folder called `mvp`):
```python
!python -u /content/mvp.py --dataset "/content/drive/My Drive/mvp/RML2016.10a_dict.pkl"
```

**Keep the session alive** — paste this in a separate cell to prevent Colab from disconnecting:
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

**First time setup**
```bash
git clone https://github.com/YOUR_USERNAME/amc-mvp.git
cd amc-mvp
```

**Daily workflow**
```bash
git pull                          # always pull before starting work
# ... make your changes ...
git add .
git commit -m "describe your change clearly"
git push
```

**Note** — the dataset file (`*.pkl`) is excluded by `.gitignore` and will never be pushed to GitHub. Every team member must download it separately from the Kaggle link above.

---

### Observations

- Deep learning models significantly outperform classical machine learning baselines.
- The LSTM achieved the highest overall classification accuracy on the RadioML2016.10a dataset.
- The Inception + ResNet model demonstrated strong high-SNR performance (91.27% at 14 dB), validating the effectiveness of multi-scale convolutional feature extraction.
- The lower overall accuracy is primarily due to reduced performance under low-SNR conditions, suggesting opportunities for further optimisation through hyperparameter tuning and architectural refinement.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Data generation, feature extraction, classical ML stages |
| `torch` | PyTorch — Deep Learning Models (Stages 4–6) |
