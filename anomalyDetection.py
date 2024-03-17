import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the blackjack dataset from the CSV file
data = pd.read_csv("blackjack_dataset.csv")

# Separate the features and labels
X = data.drop(['outcome', 'label'], axis=1)
y = data['label']

# Create an Isolation Forest model
isolation_forest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)

# Fit the model to the data
isolation_forest.fit(X)

# Predict anomalies
y_pred = isolation_forest.predict(X)

# Convert the predictions to binary labels
# -1 indicates an anomaly, 1 indicates a normal instance
y_pred_binary = [0 if x == -1 else 1 for x in y_pred]

# Evaluate the model
accuracy = accuracy_score(y, y_pred_binary)
precision = precision_score(y, y_pred_binary)
recall = recall_score(y, y_pred_binary)
f1 = f1_score(y, y_pred_binary)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)