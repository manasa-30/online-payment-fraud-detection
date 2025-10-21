from fraud_model import FraudDetector

def main():
    print("Training Fraud Detection Model...")
    
    # Initialize detector
    detector = FraudDetector()
    
    # Train model
    accuracy = detector.train('op.csv')
    
    # Save trained model
    detector.save_model('fraud_model.pkl')
    
    print(f"\nTraining completed with {accuracy:.2%} accuracy")
    print("Model saved as 'fraud_model.pkl'")

if __name__ == "__main__":
    main()