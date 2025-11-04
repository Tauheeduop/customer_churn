# ==========================================
#  Customer Churn Prediction 
#  Using Decision Tree Classifier
# ==========================================

#  Import necessary libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
#from sklearn.linear_model import LogisticRegression
#from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# ==========================================
# 1️⃣ Load the dataset
# ==========================================
data = pd.read_csv("data/churn.csv")

# ==========================================
# 2️⃣ Data Preprocessing
# ==========================================

# Replace blank spaces in 'TotalCharges' with 0 and convert to float
data['TotalCharges'] = data['TotalCharges'].replace(" ", 0).astype(float)

# Convert 'Churn' column into numeric values: Yes → 1, No → 0
data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})

# Encode all categorical columns into numeric using LabelEncoder
le = LabelEncoder()
for column in data.columns:
    if data[column].dtype == 'object':  # Check if column is categorical
        data[column] = le.fit_transform(data[column])

# ===================================================================
# 3️⃣ Split dataset into Features (X) and Target (y)
# ===================================================================
X = data.drop('Churn', axis=1)   # All columns except target
y = data['Churn']                # Target column

# ====================================================================
# 4️⃣ Split into Training & Testing sets
# ====================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,         # 20% data for testing
    random_state=42,       # reproducibility
    stratify=y             # keep churn ratio same in train/test
)

# ==========================================
# 5️⃣ Feature Scaling (Standardization)
# ==========================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================================
# 6️⃣ Model Training
# ==========================================
#model = LogisticRegression()

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# ==========================================
# 7️⃣ Model Prediction
# ==========================================
y_pred = model.predict(X_test)

# ==========================================
# 8️⃣ Model Evaluation
# ==========================================
# Accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy:.2f}\n")

# Confusion Matrix
print("🧩 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))
# ==========================================