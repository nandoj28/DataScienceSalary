from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Select the top 10 features based on importance
top_10_features = [features[i] for i in indices[-12:]]

# Subset the data to include only the top 10 features
X_top_features = X[top_10_features]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_top_features, y, test_size=0.2, random_state=42)

# Initialize the models
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
# Placeholder for other models, e.g., XGBoost or LightGBM

# Train the Random Forest model
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred_rf = rf_model.predict(X_test)

# Calculate and print metrics for Random Forest
print("Random Forest Regressor Metrics:")
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred_rf))
print("Root Mean Squared Error (RMSE):", mean_squared_error(y_test, y_pred_rf, squared=False))
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred_rf))
print("R^2 Score:", r2_score(y_test, y_pred_rf))



# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.optimizers import Adam

# # Define the neural network model
# model = Sequential([
#     Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
#     Dense(32, activation='relu'),
#     Dense(1)  # Output layer for regression (single neuron, no activation)
# ])

# # Compile the model specifying the optimizer, loss function, and metrics
# model.compile(optimizer=Adam(learning_rate=0.01),
#               loss='mean_squared_error',
#               metrics=['mae', 'mse'])

# # Train the model
# history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

# # Evaluate the model on the test set
# test_loss, test_mae, test_mse = model.evaluate(X_test, y_test, verbose=1)

# print(f"Test Loss: {test_loss}")
# print(f"Test MAE: {test_mae}")
# print(f"Test MSE: {test_mse}")