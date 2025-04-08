import joblib
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
import seaborn as sns


# Load dataset
df = pd.read_csv('cleaned_data.csv')

# Display basic information
print(df.info())
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Handle missing values
df.fillna('Unknown', inplace=True)

# Select important features
features = ['Age', 'Gender', 'self_employed', 'work_interfere', 'family_history', 'treatment']
target = 'treatment'

# Filter dataset
df = df[features]

# Encode categorical variables
encoder = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = encoder.fit_transform(df[col])

# Check the processed data
print(df.head())

# Split data
X = df.drop(target, axis=1)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Random Forest Classifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

# Evaluate
print("Random Forest Performance:")
print(classification_report(y_test, rf_preds))

# Neural Network Classifier
nn = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
nn.fit(X_train, y_train)
nn_preds = nn.predict(X_test)

# Evaluate
print("Neural Network Performance:")
print(classification_report(y_test, nn_preds))

# Confusion matrix for Random Forest
cm_rf = confusion_matrix(y_test, rf_preds)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix: Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Feature importance for Random Forest
feature_importance = rf.feature_importances_
plt.barh(df.drop(target, axis=1).columns, feature_importance)
plt.title('Feature Importance: Random Forest')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# Confusion matrix for Neural Network
cm_nn = confusion_matrix(y_test, nn_preds)
sns.heatmap(cm_nn, annot=True, fmt='d', cmap='Oranges')
plt.title('Confusion Matrix: Neural Network')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Save trained Random Forest model
joblib.dump(rf, 'random_forest_model.pkl')

# Save the fitted scaler (fit on all 5 features)
joblib.dump(scaler, 'scaler.pkl')

print("✅ Model and scaler saved successfully!")
