# Fraud Detection System

A machine learning-based fraud detection system with a web interface for real-time transaction analysis.

## Features

- **Machine Learning Model**: Detects fraudulent transactions using transaction patterns
- **Web Interface**: User-friendly frontend for inputting transaction details
- **Real-time Predictions**: Instant fraud probability scoring

## Algorithm Used

**Random Forest Classifier**
- Ensemble learning method using multiple decision trees
- Combines predictions from 100 decision trees (n_estimators=100)
- Handles non-linear relationships and feature interactions
- Robust against overfitting through bootstrap aggregating
- Provides feature importance rankings

## Model Features

The model analyzes these transaction attributes:
- Transaction type (TRANSFER, CASH_OUT, PAYMENT, CASH_IN, DEBIT)
- Transaction amount
- Origin account balances (before/after)
- Destination account balances (before/after)

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model:
```bash
python train_model.py
```

3. Start the web interface:
```bash
python app.py
```

4. Open http://localhost:5000 in your browser

## How the Model Works

**Data Processing:**
1. Loads transaction data from CSV file
2. Encodes transaction types (TRANSFER, CASH_OUT, etc.) into numerical values
3. Uses 6 key features: transaction type, amount, and account balances

**Training Process:**
1. Splits data into 80% training and 20% testing sets
2. Trains Random Forest with 100 decision trees
3. Each tree learns different patterns from bootstrap samples
4. Final prediction combines votes from all trees

**Prediction:**
1. Takes transaction details as input
2. Encodes transaction type using trained encoder
3. Passes features through all 100 trees
4. Returns fraud probability (0-1) and binary classification
5. Threshold: >0.5 probability = FRAUD, ≤0.5 = LEGITIMATE

## Quick Start (Next Time)

1. Navigate to project directory:
```bash
cd c:\Users\User\projects\ml\project-op
```

2. Start the web application:
```bash
python app.py
```

3. Open browser and go to: http://localhost:5000

**Note:** Model is already trained and saved as `fraud_model.pkl`

## Usage

Enter transaction details in the web form to get fraud predictions with probability scores.