import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Load cleaned data
df = pd.read_csv('cleaned_weather_data.csv')

# Convert date and extract month
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month

# Select features and target
features = ['Month', 'DewPointAvgF', 'HumidityAvgPercent', 'WindAvgMPH', 'PrecipitationSumInches']
X = df[features]
y = df['TempAvgF']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model (Linear Regression)
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Save model and scaler
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Model and scaler saved successfully!")
