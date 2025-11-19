# 📈 Stock Price Movement Classifier

Predict whether **tomorrow’s closing price** will go **up (1)** or **down (0)** using machine learning models trained on daily **OHLCV** (Open, High, Low, Close, Volume) data from the Kaggle *Huge Stock Market Dataset*.

This repository contains:

- **Model 1 — Logistic Regression (Lab 2 implementation)**  
  A fully manual NumPy-based gradient descent classifier adapted from KU Leuven’s Assignment 2.  
  This model serves as the required baseline with **no machine-learning libraries**.

- **Model 2 — Multi-Layer Perceptron (Neural Network)**  
  A nonlinear model used for performance comparison and improvement.

---

# ⚙️ 1. Environment Setup

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/yaoztorun/stockprice_classifier.git
cd stockprice_classifier
````

---

## 2️⃣ Create and Activate a Virtual Environment

### **Windows (PowerShell)**

```bash
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

If activation fails:

```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

---

### **macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

# 📥 2. Download the Dataset (Kaggle)

## 1️⃣ Create a Kaggle API Token

* Go to **Kaggle → Account → API → Create New Token**
* This downloads a file named **kaggle.json**

---

## 2️⃣ Move the Token to the Correct Location

### **Windows**

```shell
%USERPROFILE%\.kaggle\kaggle.json
```

### **macOS / Linux**

```bash
~/.kaggle/kaggle.json
```

---

## 3️⃣ Download the Dataset

```bash
pip install kaggle

kaggle datasets download -d borismarjanovic/price-volume-data-for-all-us-stocks-etfs -p data/raw

unzip data/raw/price-volume-data-for-all-us-stocks-etfs.zip -d data/raw
```

Your folder structure should now contain:

```
data/raw/Stocks/*.txt
```

Each `.txt` file corresponds to one stock ticker with historical daily OHLCV data.

---

