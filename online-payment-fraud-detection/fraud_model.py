import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib

class FraudDetector:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
        self.label_encoder = LabelEncoder()
        self.is_trained = False
    
    def train(self, data_path):
        # Load data with optimized dtypes and chunking
        dtype_dict = {
            'amount': 'float32',
            'oldbalanceOrg': 'float32', 
            'newbalanceOrig': 'float32',
            'oldbalanceDest': 'float32',
            'newbalanceDest': 'float32',
            'isFraud': 'int8'
        }
        
        # Read in chunks for large files
        chunk_size = 50000
        chunks = []
        
        for chunk in pd.read_csv(data_path, chunksize=chunk_size, dtype=dtype_dict):
            chunks.append(chunk)
            
        df = pd.concat(chunks, ignore_index=True)
        
        # Encode transaction type
        df['type_encoded'] = self.label_encoder.fit_transform(df['type']).astype('int8')
        
        # Features with memory optimization
        features = ['type_encoded', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
        X = df[features].astype('float32')
        y = df['isFraud'].astype('int8')
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model trained successfully!")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return accuracy
    
    def predict(self, transaction_type, amount, old_balance_orig, new_balance_orig, old_balance_dest, new_balance_dest):
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        # Encode transaction type
        type_encoded = self.label_encoder.transform([transaction_type])[0]
        
        # Create feature array
        features = np.array([[type_encoded, amount, old_balance_orig, new_balance_orig, old_balance_dest, new_balance_dest]])
        
        # Predict
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0][1]
        
        return {
            'is_fraud': bool(prediction),
            'fraud_probability': probability,
            'result': 'FRAUD' if prediction else 'LEGITIMATE'
        }
    
    def save_model(self, model_path='fraud_model.pkl'):
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder
        }
        joblib.dump(model_data, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path='fraud_model.pkl'):
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.is_trained = True
        print(f"Model loaded from {model_path}")

if __name__ == "__main__":
    # Example usage
    detector = FraudDetector()
    
    # Train model (assuming op.csv exists)
    try:
        detector.train('op.csv')
        detector.save_model()
        
        # Test prediction
        result = detector.predict('TRANSFER', 181.0, 181.0, 0.0, 0.0, 0.0)
        print(f"\nTest Prediction: {result}")
        
    except FileNotFoundError:
        print("op.csv not found. Please add the dataset to train the model.")