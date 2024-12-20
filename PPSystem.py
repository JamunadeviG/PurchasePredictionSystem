import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE

data = pd.read_csv('pps1.csv')

label_encoder_category = LabelEncoder()
label_encoder_season = LabelEncoder()
label_encoder_subscription = LabelEncoder()

data['Category'] = label_encoder_category.fit_transform(data['Category'])
data['Season'] = label_encoder_season.fit_transform(data['Season'])
data['Subscription Status'] = label_encoder_subscription.fit_transform(data['Subscription Status'])

X = data.drop(columns=['Customer ID', 'Subscription Status'])
y = data['Subscription Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_balanced = scaler.fit_transform(X_train_balanced)
X_test = scaler.transform(X_test)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_balanced, y_train_balanced)

y_pred = knn_model.predict(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=1))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=1):.2f}")
print(f"Recall: {recall_score(y_test, y_pred, zero_division=1):.2f}")
print(f"F1 Score: {f1_score(y_test, y_pred, zero_division=1):.2f}")

def predict_subscription():
    category = input("Enter product category (e.g., Clothing, Footwear): ").title()
    purchase_amount = float(input("Enter purchase amount (USD): "))
    season = input("Enter season (e.g., Winter, Spring): ").title()
    review_rating = float(input("Enter review rating (e.g., 3.5): "))


    if category not in label_encoder_category.classes_ or season not in label_encoder_season.classes_:
        print(f"Error: Category '{category}' or Season '{season}' not found in training data.")
        return


    category_encoded = label_encoder_category.transform([category])[0]
    season_encoded = label_encoder_season.transform([season])[0]


    input_data = [[category_encoded, purchase_amount, season_encoded, review_rating]]
    input_data_scaled = scaler.transform(input_data)

    prediction = knn_model.predict(input_data_scaled)

    if prediction[0] == 1:
        print("Prediction: The customer will subscribe.")
    else:
        print("Prediction: The customer will not subscribe.")

predict_subscription()
