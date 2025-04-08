import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# Load dataset
df = pd.read_csv('uncleand_data.csv')

# Drop unnecessary columns
df.drop(columns=['Timestamp', 'Country', 'state', 'comments'], inplace=True)

# Clean Age column
df = df[(df['Age'] >= 0) & (df['Age'] <= 100)]

# Clean Gender column
df['Gender'].replace(
    ['Male ', 'male', 'M', 'm', 'Male', 'Cis Male', 'Man', 'cis male', 'Mail',
     'Male-ish', 'Male (CIS)', 'Cis Man', 'msle', 'Malr', 'Mal', 'maile', 'Make'], 'Male', inplace=True)
df['Gender'].replace(
    ['Female ', 'female', 'F', 'f', 'Woman', 'Female', 'femail', 'Cis Female',
     'cis-female/femme', 'Femake', 'Female (cis)', 'woman'], 'Female', inplace=True)
df['Gender'].replace(
    ['Female (trans)', 'queer/she/they', 'non-binary', 'fluid', 'queer', 'Androgyne',
     'Trans-female', 'male leaning androgynous', 'Agender', 'A little about you', 'Nah',
     'All', 'ostensibly male, unsure what that really means', 'Genderqueer', 'Enby', 'p',
     'Neuter', 'something kinda male?', 'Guy (-ish) ^_^', 'Trans woman'], 'Others', inplace=True)

# Fill missing values
df['work_interfere'] = df['work_interfere'].fillna('Do not know')
df['self_employed'] = df['self_employed'].fillna('No')
df['benefits'].fillna(df['benefits'].mode()[0], inplace=True)
df['wellness_program'].fillna(df['wellness_program'].mode()[0], inplace=True)
df['leave'].fillna(df['leave'].mode()[0], inplace=True)

# Select important columns
selected_columns = ['Age', 'Gender', 'self_employed', 'work_interfere', 'family_history', 'treatment']
target = 'treatment'
df = df[selected_columns]

# Encode categorical variables
label_maps = {
    'Gender': {'Male': 1, 'Female': 0, 'Others': 2},
    'self_employed': {'No': 0, 'Yes': 1},
    'work_interfere': {'Do not know': 0, 'Never': 1, 'Rarely': 2, 'Sometimes': 3, 'Often': 4},
    'family_history': {'No': 0, 'Yes': 1}
}

for col, mapping in label_maps.items():
    df[col] = df[col].map(mapping)

# Define features and target
y = df['treatment']
X = df.drop('treatment', axis=1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=125)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Restore feature names
feature_names = X.columns
X_train = pd.DataFrame(X_train_scaled, columns=feature_names)
X_test = pd.DataFrame(X_test_scaled, columns=feature_names)


# Accuracy reporting function
def acc_report(actual, predicted):
    acc_score = accuracy_score(actual, predicted)
    cm_matrix = confusion_matrix(actual, predicted)
    class_rep = classification_report(actual, predicted)
    print('The accuracy of the model is:', acc_score)
    print(cm_matrix)
    print(class_rep)


# Confusion matrix plot function
def plot_conf_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    plt.show()


# ----- Train Models -----

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
pred_dt_test = dt.predict(X_test)
print("\n--- Decision Tree Classifier - Testing Set ---")
acc_report(y_test, pred_dt_test)
plot_conf_matrix(y_test, pred_dt_test, "Decision Tree")

# Random Forest
rf = RandomForestClassifier(criterion='entropy', max_depth=12)
rf.fit(X_train, y_train)
pred_rf_test = rf.predict(X_test)
print("\n--- Random Forest Classifier - Testing Set ---")
acc_report(y_test, pred_rf_test)
plot_conf_matrix(y_test, pred_rf_test, "Random Forest")

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
pred_lr_test = lr.predict(X_test)
print("\n--- Logistic Regression - Testing Set ---")
acc_report(y_test, pred_lr_test)
plot_conf_matrix(y_test, pred_lr_test, "Logistic Regression")

# Save Logistic Regression model and scaler
# ... [Keep all the imports and previous code until model training] ...

# Save Logistic Regression model and scaler (keep this part)
try:
    joblib.dump(lr, 'logistic_regression_model.pkl')
    print("✅ Logistic Regression model saved.")
except Exception as e:
    print("❌ Error saving model:", e)

try:
    joblib.dump(scaler, 'scaler.pkl')
    print("✅ Scaler saved.")
except Exception as e:
    print("❌ Error saving scaler:", e)


# ---------------------------
# USER INPUT & PREDICTION
# ---------------------------

def get_user_input():
    """Collects user input through terminal (replace with your form logic)"""
    print("\n👉 Provide Mental Health Treatment Predictions Inputs:")

    inputs = {
        'Age': int(input("Age (0-100): ")),
        'Gender': input("Gender (Male/Female/Others): ").strip().capitalize(),
        'self_employed': input("Self Employed (Yes/No): ").strip().capitalize(),
        'work_interfere': input("Work Interference (Never/Rarely/Sometimes/Often/Do not know): ").strip().capitalize(),
        'family_history': input("Family History (Yes/No): ").strip().capitalize()
    }
    return inputs


# Load saved model and scaler
try:
    lr = joblib.load('logistic_regression_model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("\n✅ Model & scaler loaded successfully")
except Exception as e:
    print(f"❌ Error loading resources: {e}")
    exit()

# Get user input
user_data = get_user_input()

# Map categorical inputs to numerical values using label_maps
mapped_input = [
    user_data['Age'],
    label_maps['Gender'].get(user_data['Gender'], 2),  # Default to 'Others' if unknown
    label_maps['self_employed'].get(user_data['self_employed'], 0),  # Default to 'No'
    label_maps['work_interfere'].get(user_data['work_interfere'], 0),  # Default to 'Do not know'
    label_maps['family_history'].get(user_data['family_history'], 0)  # Default to 'No'
]

# Create DataFrame with correct feature names
input_df = pd.DataFrame([mapped_input], columns=feature_names)

# Scale features
scaled_input = scaler.transform(input_df)
scaled_input_df = pd.DataFrame(scaled_input, columns=feature_names)  # Preserve feature names



