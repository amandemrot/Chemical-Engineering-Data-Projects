import pandas as pd

# Load the data
df = pd.read_csv('plant_energy_data.csv')

# Calculate 'Energy_Intensity' (Energy per unit of production)
# We sum electricity and steam as total energy consumption
df['Energy_Intensity'] = (df['Electricity_kWh'] + df['Steam_Tons']) / df['Production_Units']

# Define "Inefficient" as any hour where intensity is higher than the average + 1 standard deviation
threshold = df['Energy_Intensity'].mean() + df['Energy_Intensity'].std()
inefficient_data = df[df['Energy_Intensity'] > threshold]

# Save the report
inefficient_data.to_csv('optimization_report.csv', index=False)

print("Analysis Complete!")
print(f"Total hours analyzed: {len(df)}")
print(f"Number of inefficient hours found: {len(inefficient_data)}")
print("Check 'optimization_report.csv' for the specific timestamps to investigate.")