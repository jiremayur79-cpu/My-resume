# UPI FRAUD DETECTION PROJECT
# Step-by-step guide for beginners

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("UPI FRAUD DETECTION PROJECT")
print("=" * 60)

# STEP 1: CREATE SAMPLE UPI TRANSACTION DATA
print("\n[STEP 1] Creating Sample UPI Transaction Dataset...")

# Create synthetic UPI transaction data
np.random.seed(42)
n_samples = 10000

# Normal transactions
normal_amount = np.random.uniform(100, 5000, n_samples)
normal_time = np.random.uniform(0, 24, n_samples)
normal_distance = np.random.uniform(0, 100, n_samples)
normal_frequency = np.random.poisson(2, n_samples)

# Create features
data = {
    'transaction_amount': normal_amount,
    'time_of_day': normal_time,
    'distance_from_home': normal_distance,
    'transaction_frequency': normal_frequency,
    'days_since_last_transaction': np.random.uniform(0, 30, n_samples),
    'device_type': np.random.choice([0, 1, 2], n_samples),
    'is_new_device': np.random.choice([0, 1], n_samples),
    'is_international': np.random.choice([0, 1], n_samples)
}

# Create fraud labels (80% normal, 20% fraud)
fraud_label = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
data['is_fraud'] = fraud_label

# Make fraudulent transactions have different patterns
fraud_indices = np.where(fraud_label == 1)[0]
data_array = np.array([data['transaction_amount'], data['time_of_day'], data['distance_from_home']])
for idx in fraud_indices[:int(len(fraud_indices) * 0.5)]:
    data['transaction_amount'][idx] *= 2
    data['distance_from_home'][idx] *= 3

# Create DataFrame
df = pd.DataFrame(data)
print(f"Dataset created: {len(df)} transactions")
print(f"Fraud cases: {df['is_fraud'].sum()} ({df['is_fraud'].sum()/len(df)*100:.1f}%)")
print(f"\nFirst 5 rows:\n{df.head()}")

# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
print("\n[STEP 2] Exploring the Data...")

print(f"\nDataset Info:")
print(df.info())
print(f"\nBasic Statistics:")
print(df.describe())

print(f"\nMissing Values: {df.isnull().sum().sum()}")
print(f"Fraud Distribution:\n{df['is_fraud'].value_counts()}")

# STEP 3: DATA PREPROCESSING
print("\n[STEP 3] Preparing Data for Model...")

X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set: {len(X_train)} samples")
print(f"Testing set: {len(X_test)} samples")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features scaled using StandardScaler")

# STEP 4: TRAIN MACHINE LEARNING MODELS
print("\n[STEP 4] Training Machine Learning Models...")

# Model 1: Logistic Regression
print("\n--- Logistic Regression ---")
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)

lr_accuracy = accuracy_score(y_test, lr_pred)
lr_precision = precision_score(y_test, lr_pred)
lr_recall = recall_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred)

print(f"Accuracy: {lr_accuracy:.4f} ({lr_accuracy*100:.2f}%)")
print(f"Precision: {lr_precision:.4f}")
print(f"Recall: {lr_recall:.4f}")
print(f"F1-Score: {lr_f1:.4f}")

# Model 2: Random Forest
print("\n--- Random Forest Classifier ---")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)

print(f"Accuracy: {rf_accuracy:.4f} ({rf_accuracy*100:.2f}%)")
print(f"Precision: {rf_precision:.4f}")
print(f"Recall: {rf_recall:.4f}")
print(f"F1-Score: {rf_f1:.4f}")

# STEP 5: DETAILED CLASSIFICATION REPORT
print("\n[STEP 5] Detailed Classification Report (Logistic Regression)...")
print("\n" + classification_report(y_test, lr_pred, target_names=['Normal', 'Fraud']))

# STEP 6: VISUALIZATIONS
print("\n[STEP 6] Creating Visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('UPI Fraud Detection - Analysis & Results', fontsize=16, fontweight='bold')

# Plot 1: Fraud Distribution
ax1 = axes[0, 0]
fraud_counts = df['is_fraud'].value_counts()
ax1.bar(['Normal', 'Fraud'], fraud_counts.values, color=['green', 'red'], alpha=0.7)
ax1.set_title('Fraud Distribution in Dataset', fontweight='bold')
ax1.set_ylabel('Count')
for i, v in enumerate(fraud_counts.values):
    ax1.text(i, v + 100, str(v), ha='center', fontweight='bold')

# Plot 2: Transaction Amount by Fraud Status
ax2 = axes[0, 1]
df.boxplot(column='transaction_amount', by='is_fraud', ax=ax2)
ax2.set_title('Transaction Amount: Normal vs Fraud', fontweight='bold')
ax2.set_xlabel('Fraud Status (0=Normal, 1=Fraud)')
ax2.set_ylabel('Amount (₹)')
plt.sca(ax2)
plt.xticks([1, 2], ['Normal', 'Fraud'])

# Plot 3: Confusion Matrix (Logistic Regression)
ax3 = axes[1, 0]
cm = confusion_matrix(y_test, lr_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3, cbar=False)
ax3.set_title('Confusion Matrix - Logistic Regression', fontweight='bold')
ax3.set_ylabel('True Label')
ax3.set_xlabel('Predicted Label')
ax3.set_xticklabels(['Normal', 'Fraud'])
ax3.set_yticklabels(['Normal', 'Fraud'])

# Plot 4: Model Performance Comparison
ax4 = axes[1, 1]
models = ['Logistic\nRegression', 'Random\nForest']
accuracy = [lr_accuracy, rf_accuracy]
precision = [lr_precision, rf_precision]
recall = [lr_recall, rf_recall]

x = np.arange(len(models))
width = 0.25

ax4.bar(x - width, accuracy, width, label='Accuracy', alpha=0.8)
ax4.bar(x, precision, width, label='Precision', alpha=0.8)
ax4.bar(x + width, recall, width, label='Recall', alpha=0.8)

ax4.set_ylabel('Score')
ax4.set_title('Model Performance Comparison', fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(models)
ax4.legend()
ax4.set_ylim([0, 1])
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('upi_fraud_detection_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved as 'upi_fraud_detection_analysis.png'")
plt.close()

# STEP 7: FEATURE IMPORTANCE
print("\n[STEP 7] Feature Importance (Random Forest)...")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop Features:")
print(feature_importance)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'], color='steelblue')
plt.xlabel('Importance Score')
plt.title('Feature Importance in UPI Fraud Detection', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ Feature importance saved as 'feature_importance.png'")
plt.close()

# SUMMARY
print("\n" + "=" * 60)
print("PROJECT SUMMARY")
print("=" * 60)
print(f"\n✓ Dataset: 10,000 UPI transactions")
print(f"✓ Features: {X.shape[1]} (Amount, Time, Distance, etc.)")
print(f"✓ Models Trained: Logistic Regression, Random Forest")
print(f"✓ Best Model Accuracy: {max(lr_accuracy, rf_accuracy)*100:.2f}%")
print(f"✓ Fraud Detection Rate (Recall): {max(lr_recall, rf_recall)*100:.2f}%")
print(f"✓ Visualizations: Generated 2 charts")
print("\n" + "=" * 60)
print("PROJECT COMPLETE!")
print("=" * 60)