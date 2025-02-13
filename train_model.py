import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

df = pd.read_csv('cleaned_weather_data.csv')

df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month

features = ['Month', 'DewPointAvgF', 'HumidityAvgPercent', 'WindAvgMPH', 'PrecipitationSumInches']
X = df[features]
y = df['TempAvgF']

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model (Linear Regression)
model = LinearRegression()
model.fit(X_scaled, y)

# Save model and scaler
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Model and scaler saved successfully!")

