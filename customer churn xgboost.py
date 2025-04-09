import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to save plots instead of showing them
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from xgboost import XGBClassifier

# Load the data
file_path = "customer_churn_dataset-testing-master.csv"
df = pd.read_csv(file_path)

# Basic info
print("DataFrame Info:")
print(df.info())

# First few rows
print("\nFirst 5 rows:")
print(df.head())

# Missing values
print("\nMissing values:")
print(df.isnull().sum())

# Descriptive statistics
print("\nDescriptive stats:")
print(df.describe())

# Churn count plot
if 'Churn' in df.columns:
    sns.countplot(x='Churn', data=df)
    plt.title('Churn Count')
    plt.savefig('churn_count_plot.png')
    plt.close()
    print("Churn count plot saved as churn_count_plot.png")

# Correlation heatmap
plt.figure(figsize=(12, 8))
df_encoded_corr = pd.get_dummies(df, drop_first=True)
sns.heatmap(df_encoded_corr.corr(), annot=False, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
plt.close()
print("Correlation heatmap saved as correlation_heatmap.png")

# Prepare data for model
if 'Churn' in df.columns:
    df_encoded = pd.get_dummies(df, drop_first=True)
    X = df_encoded.drop('Churn', axis=1)
    y = df_encoded['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train XGBoost
    model = model = XGBClassifier(eval_metric='logloss')

    model.fit(X_train, y_train)

    # Predictions and evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"XGBoost Model Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix - print and plot
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix (array form):")
    print(cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()
    print("Confusion matrix plot saved as confusion_matrix.png")

else:
    print("No 'Churn' column found for prediction.")
