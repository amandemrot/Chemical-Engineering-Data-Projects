import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Create 8760 hours (1 year of data)
dates = pd.date_range(start='2025-01-01', periods=8760, freq='h')

# Generate Data
df = pd.DataFrame({
    'Timestamp': dates,
    'Production_Units': np.random.randint(50, 500, 8760),
    'Electricity_kWh': np.random.normal(200, 20, 8760),
    'Steam_Tons': np.random.normal(50, 5, 8760),
    'Fuel_Cost_USD': np.random.normal(1000, 100, 8760)
})

# Add some "inefficiency" spikes
df.loc[df['Production_Units'] < 100, 'Electricity_kWh'] += 150 

# Save to CSV
df.to_csv('plant_energy_data.csv', index=False)
print("Success! File 'plant_energy_data.csv' has been created.")