import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import joblib

def train_and_save_model():
    """Train the model and save it along with encoders and scaler"""
    print("Loading and preprocessing data...")
    
    # Load and preprocess the dataset
    data = pd.read_csv('pps1.csv')
    
    # Only use the key features that we collect in the UI
    key_features = ['Age', 'Category', 'Purchase Amount (USD)', 'Season', 'Review Rating', 'Previous Purchases']
    
    # Initialize encoders for categorical columns we're using
    encoders = {}
    categorical_columns = ['Category', 'Season']
    
    # Encode categorical variables
    for col in categorical_columns:
        if col in data.columns:
            encoders[col] = LabelEncoder()
            data[col] = encoders[col].fit_transform(data[col])
    
    # Encode target variable
    label_encoder_subscription = LabelEncoder()
    data['Subscription Status'] = label_encoder_subscription.fit_transform(data['Subscription Status'])
    
    # Select only the key features for training
    X = data[key_features]
    y = data['Subscription Status']
    
    print(f"Features: {X.columns.tolist()}")
    print(f"Dataset shape: {X.shape}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Balance classes with SMOTE
    print("Balancing classes with SMOTE...")
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    print(f"Original class distribution: {np.bincount(y_train)}")
    print(f"Balanced class distribution: {np.bincount(y_train_balanced)}")
    
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_balanced = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)
    
    # Train a simple but effective model
    print("Training Random Forest model...")
    
    # Use Random Forest with balanced class weights - it's more robust
    best_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced',
        min_samples_split=5,
        min_samples_leaf=2
    )
    
    best_model.fit(X_train_balanced, y_train_balanced)
    
    # Use the best model we found
    print(f"Selected model: {type(best_model).__name__}")
    
    # Evaluate model
    print("Evaluating model...")
    y_pred = best_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Model Performance:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1:.3f}")
    
    # Save model and preprocessing objects
    print("Saving model and preprocessing objects...")
    
    model_data = {
        'model': best_model,
        'scaler': scaler,
        'encoders': encoders,
        'subscription_encoder': label_encoder_subscription,
        'metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'model_type': type(best_model).__name__,
        },
        'feature_columns': X.columns.tolist()
    }
    
    # Save using joblib for better performance with sklearn objects
    joblib.dump(model_data, 'trained_model.pkl')
    print("Model saved as 'trained_model.pkl'")
    
    return model_data

if __name__ == "__main__":
    train_and_save_model()
