# ============================
# HEART DISEASE DETECTION
# COMPLETE PROJECT IN ONE CELL
# ============================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ----------------------------
# Load Dataset
# ----------------------------
df = pd.read_csv("cleaned_merged_heart_dataset.csv")

print("="*50)
print("DATASET INFORMATION")
print("="*50)
print(df.head())
print("\nShape:", df.shape)
print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

# ----------------------------
# EDA
# ----------------------------
print("\nGenerating Visualizations...")

numeric_cols = ['age','trestbps','chol','thalachh','oldpeak']

for col in numeric_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()

plt.figure(figsize=(6,4))
sns.countplot(data=df, x='target')
plt.title('Heart Disease Distribution')
plt.show()

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# ----------------------------
# Prepare Data
# ----------------------------
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ----------------------------
# Feature Scaling
# ----------------------------
preprocessor = ColumnTransformer(
    [('scaler', StandardScaler(), X.columns)],
    remainder='passthrough'
)

# ----------------------------
# Models
# ----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=7),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42
    ),
    "KNN": KNeighborsClassifier(
        n_neighbors=4,
        metric='euclidean'
    )
}

results = {}

# ----------------------------
# Training & Evaluation
# ----------------------------
for name, model in models.items():

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    results[name] = accuracy

    print("\n" + "="*50)
    print(name)
    print("="*50)

    print("Accuracy:", round(accuracy*100,2), "%")

    print("\nClassification Report")
    print(classification_report(y_test, y_pred))

    plt.figure(figsize=(5,4))
    sns.heatmap(
        confusion_matrix(y_test, y_pred),
        annot=True,
        fmt='d',
        cmap='Blues'
    )
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# ----------------------------
# Best Model
# ----------------------------
print("\n" + "="*50)
print("MODEL COMPARISON")
print("="*50)

for model, score in results.items():
    print(f"{model}: {score:.4f}")

best_model = max(results, key=results.get)

print("\nBest Model:", best_model)
print("Best Accuracy:", round(results[best_model]*100,2), "%")

# ----------------------------
# Sample Prediction
# ----------------------------
sample = X.iloc[[0]]

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42
    ))
])

pipeline.fit(X_train, y_train)

prediction = pipeline.predict(sample)

print("\nSample Prediction:")
print("Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease")