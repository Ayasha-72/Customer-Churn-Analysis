import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -----------------------------
# Load Dataset
# -----------------------------
DATA_PATH = os.path.join("data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print(df.head())

# -----------------------------
# Basic Info
# -----------------------------
print("\nShape:", df.shape)
print("\nInfo:")
print(df.info())

# -----------------------------
# Data Cleaning
# -----------------------------
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

print("\nAfter cleaning shape:", df.shape)

# -----------------------------
# Churn Rate
# -----------------------------
churn_rate = df['Churn'].value_counts(normalize=True) * 100
print("\nChurn Rate (%):")
print(churn_rate)

# -----------------------------
# Visualization 1: Churn Count
# -----------------------------
sns.countplot(x='Churn', data=df)
plt.title("Customer Churn Count")
plt.show()

# -----------------------------
# Visualization 2: Churn by Contract
# -----------------------------
sns.countplot(x='Contract', hue='Churn', data=df)
plt.title("Churn by Contract Type")
plt.show()

# -----------------------------
# Visualization 3: Monthly Charges vs Churn
# -----------------------------
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
plt.title("Monthly Charges vs Churn")
plt.show()

# -----------------------------
# Visualization 4: Tenure Distribution
# -----------------------------
sns.histplot(data=df, x='tenure', hue='Churn', bins=30, kde=True)
plt.title("Tenure Distribution by Churn")
plt.show()

# -----------------------------
# Visualization 5: Payment Method vs Churn
# -----------------------------
sns.countplot(x='PaymentMethod', hue='Churn', data=df)
plt.xticks(rotation=45)
plt.title("Churn by Payment Method")
plt.show()

print("\nAnalysis completed successfully!")
