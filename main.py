import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

# Load the dataset
data = pd.read_csv("data/churn.csv")

# Data preprocessing
data['TotalCharges'] = data['TotalCharges'].replace(" ", 0).astype(float)
data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})
#print(data['Churn'].value_counts())

# Encode categorical variables
le = LabelEncoder()
for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = le.fit_transform(data[column])

# Display data types and first few rows
'''
print(data.dtypes.head(100))
print(data.head())
'''
# Separate features and target variable
X = data.drop('Churn', axis=1)
y = data['Churn']

# Split the dataset into training and testing sets with stratification
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42, stratify=y)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#model training and evaluation would go here
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
# confusion matrix

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))
