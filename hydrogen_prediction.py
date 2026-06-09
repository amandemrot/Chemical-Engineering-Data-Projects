import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 1. Generate Synthetic SMR Data
np.random.seed(42)
n_samples = 500
data = {
    'Temperature_C': np.random.uniform(700, 900, n_samples),
    'Pressure_bar': np.random.uniform(10, 30, n_samples),
    'Steam_to_Carbon_Ratio': np.random.uniform(2.5, 4.0, n_samples),
    'Feedstock_Flow': np.random.uniform(50, 150, n_samples)
}
df = pd.DataFrame(data)

# Create target: Hydrogen Yield (with some noise)
df['H2_Yield'] = (0.5 * df['Temperature_C']) + (0.2 * df['Pressure_bar']) + (10 * df['Steam_to_Carbon_Ratio']) + np.random.normal(0, 2, n_samples)

# 2. Train Random Forest Model
X = df.drop('H2_Yield', axis=1)
y = df['H2_Yield']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# 3. Feature Importance Analysis
importance = pd.Series(model.feature_importances_, index=X.columns)
print("--- Feature Importance ---")
print(importance.sort_values(ascending=False))

# 4. Predictions & Evaluation
predictions = model.predict(X_test)
print(f"\nModel RMSE (Root Mean Squared Error): {np.sqrt(mean_squared_error(y_test, predictions)):.2f}")

# 5. Plotting
plt.figure(figsize=(8,6))
plt.scatter(y_test, predictions, alpha=0.5, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel("Actual Hydrogen Yield")
plt.ylabel("Predicted Hydrogen Yield")
plt.title("Actual vs Predicted Hydrogen Yield (Random Forest)")
plt.show()