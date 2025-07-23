import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. Load Data
data = pd.read_csv("database.csv")

# 2. Select relevant columns
data = data[['Date', 'Time', 'Latitude', 'Longitude', 'Depth', 'Magnitude']]

# 3. Convert Date and Time to Timestamp
def to_timestamp(row):
    try:
        dt = datetime.datetime.strptime(row['Date'] + ' ' + row['Time'], '%m/%d/%Y %H:%M:%S')
        return time.mktime(dt.timetuple())
    except Exception:
        return np.nan

data['Timestamp'] = data.apply(to_timestamp, axis=1)
data = data.dropna(subset=['Timestamp'])

# 4. Prepare final data
final_data = data.drop(['Date', 'Time'], axis=1)

# 5. Visualization (simple scatter plot)
plt.figure(figsize=(12, 6))
plt.scatter(final_data['Longitude'], final_data['Latitude'], s=2, c=final_data['Magnitude'], cmap='viridis')
plt.colorbar(label='Magnitude')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Earthquake Locations (colored by Magnitude)')
plt.show()

# 6. Split data into features and targets
X = final_data[['Timestamp', 'Latitude', 'Longitude']]
y = final_data[['Magnitude', 'Depth']]

# 7. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Feature scaling
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# 9. Build the neural network model
model = Sequential([
    Dense(16, activation='relu', input_shape=(3,)),
    Dense(16, activation='relu'),
    Dense(2, activation='linear')  # Linear for regression
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# 10. Train the model
history = model.fit(
    X_train_scaled, y_train_scaled,
    validation_data=(X_test_scaled, y_test_scaled),
    epochs=20,
    batch_size=10,
    verbose=1
)

# 11. Evaluate the model
loss, mae = model.evaluate(X_test_scaled, y_test_scaled)
print(f"Test Loss: {loss:.4f}, Test MAE: {mae:.4f}")

# 12. Predict and inverse transform to original scale
y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# 13. Show some predictions
results = pd.DataFrame({
    'Predicted Magnitude': y_pred[:, 0],
    'Actual Magnitude': y_test.values[:, 0],
    'Predicted Depth': y_pred[:, 1],
    'Actual Depth': y_test.values[:, 1]
})
print(results.head())