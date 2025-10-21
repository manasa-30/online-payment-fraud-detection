from fraud_model import FraudDetector

def main():
    # Load trained model
    detector = FraudDetector()
    detector.load_model('fraud_model.pkl')
    
    print("Fraud Detection System")
    print("=" * 30)
    
    # Get user input
    transaction_type = input("Transaction Type (PAYMENT/TRANSFER/CASH_OUT/DEBIT/CASH_IN): ").upper()
    amount = float(input("Amount: "))
    old_balance_orig = float(input("Origin Old Balance: "))
    new_balance_orig = float(input("Origin New Balance: "))
    old_balance_dest = float(input("Destination Old Balance: "))
    new_balance_dest = float(input("Destination New Balance: "))
    
    # Make prediction
    result = detector.predict(transaction_type, amount, old_balance_orig, new_balance_orig, old_balance_dest, new_balance_dest)
    
    print("\nPrediction Result:")
    print(f"Status: {result['result']}")
    print(f"Fraud Probability: {result['fraud_probability']:.2%}")

if __name__ == "__main__":
    main()