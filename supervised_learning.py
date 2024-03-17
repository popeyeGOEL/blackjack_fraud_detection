import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Load the blackjack dataset from the CSV file
data = pd.read_csv("blackjack_dataset.csv")

# Separate the features and labels
X = data.drop(['outcome', 'label'], axis=1)
y = data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost Model
xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)

# Make predictions using the XGBoost model
xgb_predictions = xgb_model.predict(X_test)

# Evaluate the XGBoost model
xgb_accuracy = accuracy_score(y_test, xgb_predictions)
xgb_precision = precision_score(y_test, xgb_predictions)
xgb_recall = recall_score(y_test, xgb_predictions)
xgb_f1 = f1_score(y_test, xgb_predictions)

print("XGBoost Model Performance:")
print("Accuracy:", xgb_accuracy)
print("Precision:", xgb_precision)
print("Recall:", xgb_recall)
print("F1-score:", xgb_f1)

# Deep Neural Network (DNN) Model
dnn_model = Sequential()
dnn_model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
dnn_model.add(Dropout(0.2))
dnn_model.add(Dense(32, activation='relu'))
dnn_model.add(Dropout(0.2))
dnn_model.add(Dense(1, activation='sigmoid'))

dnn_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
dnn_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

# Make predictions using the DNN model
dnn_predictions = (dnn_model.predict(X_test) > 0.5).astype(int)

# Evaluate the DNN model
dnn_accuracy = accuracy_score(y_test, dnn_predictions)
dnn_precision = precision_score(y_test, dnn_predictions)
dnn_recall = recall_score(y_test, dnn_predictions)
dnn_f1 = f1_score(y_test, dnn_predictions)

print("DNN Model Performance:")
print("Accuracy:", dnn_accuracy)
print("Precision:", dnn_precision)
print("Recall:", dnn_recall)
print("F1-score:", dnn_f1)