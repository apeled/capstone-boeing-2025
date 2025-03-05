# RankLab Setup Instructions

## Prerequisites
Ensure you have **Python (>=3.8)** and **R (>=4.0.0)** installed on your system.

## 1️⃣ Create and Activate a Virtual Environment
### On macOS/Linux:
```
python3 -m venv ranklab_env
source ranklab_env/bin/activate
```

### On Windows (Command Prompt):
```
python -m venv ranklab_env
ranklab_env\Scripts\activate
```

## 2️⃣ Install Python Dependencies
```
pip install dash dash-bootstrap-components plotly pandas rpy2 dash-ag-grid
```

## 3️⃣ Install R Dependencies
Open **R** and run:
```
install.packages("tidyverse")
install.packages("caret")
install.packages("MASS")
install.packages("e1071")
```

## 4️⃣ Run the R Script to Train the Model
```
Rscript decision_rank_predict.R
```

## 5️⃣ Run the Dash App
```
python ranklab_dash_app.py
```
The app will be available at **http://127.0.0.1:8050/**.
