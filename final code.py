#Gradient Boosting Machines (GBM)
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

# Load and preprocess the dataset
data = pd.read_csv('pps1.csv')
label_encoder_category = LabelEncoder()
label_encoder_season = LabelEncoder()
label_encoder_subscription = LabelEncoder()

# Encoding categorical variables
data['Category'] = label_encoder_category.fit_transform(data['Category'])
data['Season'] = label_encoder_season.fit_transform(data['Season'])
data['Subscription Status'] = label_encoder_subscription.fit_transform(data['Subscription Status'])

# Define features and target
X = data.drop(columns=['Customer ID', 'Subscription Status'])
y = data['Subscription Status']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Balance classes with SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Scale features
scaler = StandardScaler()
X_train_balanced = scaler.fit_transform(X_train_balanced)
X_test = scaler.transform(X_test)

# Gradient Boosting Classifier with GridSearchCV for Hyperparameter Tuning
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0]
}
grid_search_gbm = GridSearchCV(GradientBoostingClassifier(random_state=42), param_grid, scoring='accuracy', cv=5)
grid_search_gbm.fit(X_train_balanced, y_train_balanced)

# Best GBM model from GridSearch
best_gbm_model = grid_search_gbm.best_estimator_
print("Best Parameters for Gradient Boosting:", grid_search_gbm.best_params_)

# Evaluate the GBM model
y_pred_gbm = best_gbm_model.predict(X_test)
print("\nGradient Boosting Model Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_gbm):.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred_gbm, zero_division=1))
print(f"Precision: {precision_score(y_test, y_pred_gbm, zero_division=1):.2f}")
print(f"Recall: {recall_score(y_test, y_pred_gbm, zero_division=1):.2f}")
print(f"F1 Score: {f1_score(y_test, y_pred_gbm, zero_division=1):.2f}")

# Prediction function using the trained GBM model
def predict_subscription(model):
    category = input("Enter product category (e.g., Clothing, Footwear): ").title()
    purchase_amount = float(input("Enter purchase amount (USD): "))
    season = input("Enter season (e.g., Winter, Spring): ").title()
    review_rating = float(input("Enter review rating (e.g., 3.5): "))

    # Check if inputs are valid
    if category not in label_encoder_category.classes_ or season not in label_encoder_season.classes_:
        print(f"Error: Category '{category}' or Season '{season}' not found in training data.")
        return

    # Encode inputs and scale
    category_encoded = label_encoder_category.transform([category])[0]
    season_encoded = label_encoder_season.transform([season])[0]
    input_data = [[category_encoded, purchase_amount, season_encoded, review_rating]]
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data_scaled)
    if prediction[0] == 1:
        print("Prediction: The customer will subscribe.")
    else:
        print("Prediction: The customer will not subscribe.")

# Call the prediction function with the best GBM model
print("\nUsing Gradient Boosting for prediction example:")
predict_subscription(best_gbm_model)
