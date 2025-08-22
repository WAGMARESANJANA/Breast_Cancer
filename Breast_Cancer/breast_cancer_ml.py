# breast_cancer_ml.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load dataset
data = pd.read_csv(r"S:\BrainyBeam\Breast_Cancer\breast-cancer.csv")

# Step 2: Drop 'id' column
if 'id' in data.columns:
    data = data.drop('id', axis=1)

# Step 3: Encode labels (M = 1, B = 0)
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

# Step 4: Split features and target
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 6: Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 7: Train Random Forest model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Step 8: Predictions
y_pred = model.predict(X_test)

# Step 9: Evaluation
acc = accuracy_score(y_test, y_pred)
print("\nModel Accuracy: {:.2f}%".format(acc * 100))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))