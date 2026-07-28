
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score

sns.set_theme(style="whitegrid")

csv_path= "C:/Users/narwa/Downloads/WA_Fn-UseC_-Telco-Customer-Churn.csv"
customers_df=pd.read_csv(csv_path)

print("Dataset loaded successfully! Preview:")
print(customers_df.head(), "\n")

customers_df['did_churn'] = customers_df['Churn'].map({'Yes': 1, 'No': 0})
customers_df.rename(columns={
    'customerID': 'customer_id',
    'tenure': 'tenure_months',
    'MonthlyCharges': 'monthly_bill'
}, inplace=True)

customers_df['TotalCharges'] = pd.to_numeric(customers_df['TotalCharges'], errors='coerce')
customers_df['TotalCharges'] = customers_df['TotalCharges'].fillna(customers_df['TotalCharges'].median())

#exploratory data analysis(eda)

print("Running exploratory data analysis")
plt.figure(figsize=(8, 4))
sns.histplot(data=customers_df, x='gender', hue='did_churn', palette='muted')
plt.title('Customer Profile:Gender Distribution & Churn')
plt.xlabel('gender')
plt.ylabel('Customer Count')
plt.legend(['Retained','Churned'])
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=customers_df, 
    x='tenure_months', 
    y='monthly_bill', 
    hue='did_churn', 
    palette='coolwarm', 
    alpha=0.8
)

plt.title('Usage Trends: Tenure vs Monthly Charges')
plt.xlabel('Months with Company')
plt.ylabel('Monthly Bill ($)')
plt.show()

plt.figure(figsize=(5, 4))
sns.countplot(data=customers_df, x='did_churn', palette='Set2')
plt.title('Our Churn Profile')
plt.xticks([0, 1], ['Active Customers', 'Lost (Churned)'])
plt.ylabel('Count')
plt.show()

print("Building customer behavioral segments")
clustering_features = ['tenure_months', 'monthly_bill', 'TotalCharges']
X_clustering = customers_df[clustering_features]

data_scaler = StandardScaler()
scaled_clustering_features = data_scaler.fit_transform(X_clustering)

kmeans_model = KMeans(n_clusters=3, random_state=42, n_init=10)
customers_df['behavior_segment'] = kmeans_model.fit_predict(scaled_clustering_features)
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=customers_df, 
    x='monthly_bill', 
    y='tenure_months', 
    hue='behavior_segment', 
    palette='viridis', 
    style='behavior_segment'
)
plt.title('Customer Behavioral Segments (K-Means)')
plt.xlabel('Monthly Bill ($)')
plt.ylabel('Tenure(Month)')
plt.show()

print(" Prepping data for churn prediction...")
ml_dataset = customers_df.copy()

ml_dataset['gender_num'] = ml_dataset['gender'].map({'Male': 0, 'Female': 1})
ml_dataset['paperless_num'] = ml_dataset['PaperlessBilling'].map({'Yes': 1, 'No': 0})
churn_predictors = [
    'SeniorCitizen', 
    'gender_num', 
    'tenure_months', 
    'monthly_bill', 
    'TotalCharges', 
    'paperless_num', 
    'behavior_segment']
X = ml_dataset[churn_predictors]
y = ml_dataset['did_churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training Logistic Regression")
logistic_model = LogisticRegression(random_state=42, max_iter=1000)
logistic_model.fit(X_train, y_train)
lr_predictions = logistic_model.predict(X_test)

print("Training Decision Tree")
tree_model = DecisionTreeClassifier(max_depth=4, random_state=42) 
tree_model.fit(X_train, y_train)
tree_predictions = tree_model.predict(X_test)

print("\n Performance Summary")
print(f"\n[Model A] Logistic Regression Accuracy:{accuracy_score(y_test, lr_predictions):.2%}")
print(classification_report(y_test,lr_predictions,target_names=['Retained', 'Churned']))
print(f"[Model B] Decision Tree Accuracy: {accuracy_score(y_test, tree_predictions):.2%}")
print(classification_report(y_test, tree_predictions, target_names=['Retained', 'Churned']))

plt.figure(figsize=(16, 8))
plot_tree(
    tree_model, 
    feature_names=churn_predictors, 
    class_names=['Will Stay', 'Will Churn'], 
    filled=True, 
    rounded=True, 
    fontsize=10
)
plt.title("Visual Decision Tree Logic Rules for Customer Churn")
plt.show()

print("\n Actionable Customer Retention Playbook")
print("• [Action Item 1] High Monthly Charges & Short Tenure: Accounts paying high monthly bills in their first 12 months show the highest churn risk.")
print(" Strategy: Offer promotional discounts or long-term contract lock-ins for high-bill accounts in months 1–6.")
print("• [Action Item 2] Paperless Billing Signals: Customers with paperless billing opt-in correlate with higher turnover.")
print(" Strategy: Target paperless billing customers with automated retention emails and digital loyalty rewards.")
