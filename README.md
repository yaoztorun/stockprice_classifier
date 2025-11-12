📈 Stock Price Movement Classifier

Predict whether tomorrow’s closing price will go up (1) or down (0) using a simple, interpretable Logistic Regression model trained on daily OHLCV (Open, High, Low, Close, Volume) stock data.

Dataset: Huge Stock Market Dataset (Kaggle)

This is Model 1 — Logistic Regression, a linear baseline for next-day movement prediction.
A Decision Tree Classifier will later be added to capture non-linear threshold patterns.

⚙️ Environment Setup
1️⃣ Clone the Repository
git clone https://github.com/yaoztorun/stockprice_classifier.git
cd stockprice_classifier

2️⃣ Create and Activate Virtual Environment

Windows (PowerShell):

py -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt


If activation fails:

Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process


macOS / Linux:

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

3️⃣ Download the Dataset (from Kaggle)

Create a Kaggle API token:
Go to Kaggle → Account → API → Create New Token
It will download a file named kaggle.json.

Move it to:

Windows: %USERPROFILE%\.kaggle\kaggle.json

macOS/Linux: ~/.kaggle/kaggle.json

Download and unzip the dataset:

pip install kaggle
kaggle datasets download -d borismarjanovic/price-volume-data-for-all-us-stocks-etfs -p data/raw
unzip data/raw/price-volume-data-for-all-us-stocks-etfs.zip -d data/raw


Your folder should look like this:

data/raw/Stocks/*.txt


Each .txt file represents one stock ticker with historical daily data.