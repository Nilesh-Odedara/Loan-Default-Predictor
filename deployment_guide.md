# 🏦 Loan Default Predictor — Setup & Deployment Guide

## Project Structure

```
Project/
│
├── app.py                        ← Streamlit application
├── requirements.txt              ← Python dependencies
│
├── models/
│   ├── logistic_regression.pkl
│   ├── knn.pkl
│   ├── decision_tree.pkl
│   └── random_forest.pkl
│
└── preprocessing/
    ├── scaler.pkl                ← MinMaxScaler (fitted on 9 numerical cols)
    └── label_encoders.pkl        ← Dict of LabelEncoders (lowercase keys)
```

---

## ▶️ Running Locally

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Start the app

```bash
streamlit run app.py
```

The app will open automatically at **http://localhost:8501**

> [!TIP]
> Make sure you run the command **from the `Project/` folder** so the relative paths
> `models/` and `preprocessing/` resolve correctly.

---

## 🧠 Preprocessing — How It Works

The app replicates the **exact** training preprocessing pipeline:

| Step | What happens |
|------|-------------|
| **Label Encoding** | Categorical columns (`education`, `employmenttype`, `maritalstatus`, `hasmortgage`, `hasdependents`, `loanpurpose`, `hascosigner`) are transformed using saved `LabelEncoder` objects |
| **MinMax Scaling** | Only the **9 numerical columns** are scaled: `age`, `income`, `loanamount`, `creditscore`, `monthsemployed`, `numcreditlines`, `interestrate`, `loanterm`, `dtiratio` |
| **Feature Order** | 16 features passed to model in the same order used during training |

> [!IMPORTANT]
> The `scikit-learn` version in `requirements.txt` is **pinned to `1.6.1`** — the same
> version used to train and save the models. Using a different version may cause
> `InconsistentVersionWarning` or broken predictions.

---

## 🌐 Deploying on Streamlit Community Cloud

### Step 1 — Push to GitHub

1. Create a new **public** GitHub repository (e.g., `loan-default-predictor`)
2. Initialize git in your project folder and push:

```bash
git init
git add .
git commit -m "Initial commit: Loan Default Predictor"
git remote add origin https://github.com/<your-username>/loan-default-predictor.git
git branch -M main
git push -u origin main
```

> [!WARNING]
> The `knn.pkl` (≈27 MB) and `random_forest.pkl` (≈370 MB) are very large.
> GitHub has a **100 MB file size limit** and a **1 GB repository limit**.
> See the note below on handling large files.

### Step 2 — Handle large model files with Git LFS

```bash
git lfs install
git lfs track "models/random_forest.pkl"
git lfs track "models/knn.pkl"
git add .gitattributes
git add models/
git commit -m "Add large models via Git LFS"
git push
```

> [!NOTE]
> Git LFS requires a free account at https://git-lfs.com and is supported
> natively on GitHub. Streamlit Community Cloud can pull LFS files automatically.

### Step 3 — Deploy on Streamlit Community Cloud

1. Go to **https://share.streamlit.io** and sign in with GitHub
2. Click **"New app"**
3. Select your repository: `loan-default-predictor`
4. Set **Branch**: `main`
5. Set **Main file path**: `app.py`
6. Click **"Deploy!"**

Streamlit will install `requirements.txt` automatically and launch the app.
Your public URL will look like:
```
https://<your-username>-loan-default-predictor.streamlit.app
```

---

## 🐛 Common Issues & Fixes

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: No module named 'streamlit'` | Run `pip install -r requirements.txt` first |
| `FileNotFoundError: models/random_forest.pkl` | Ensure you run `streamlit run app.py` **from the Project/ folder** |
| `InconsistentVersionWarning` on pickle load | Pin `scikit-learn==1.6.1` in `requirements.txt` (already done) |
| GitHub rejects large file | Use **Git LFS** for any `.pkl` > 100 MB |
| Prediction seems wrong | Verify your training used the same column lowercase naming and feature order |

---

## 📦 requirements.txt Contents

```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn==1.6.1
joblib>=1.2.0
```
