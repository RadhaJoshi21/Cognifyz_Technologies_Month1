import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

file_path = 'C:\\Users\\joshi\\Downloads\\Dataset .csv'  
data = pd.read_csv(file_path)

data['Cuisines'] = data['Cuisines'].fillna('Unknown')

categorical_cols = ['Restaurant Name', 'City', 'Address', 'Locality', 'Locality Verbose', 
                    'Cuisines', 'Currency', 'Has Table booking', 'Has Online delivery', 
                    'Is delivering now', 'Switch to order menu', 'Rating color', 'Rating text']

data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

target = 'Aggregate rating'
X = data_encoded.drop(columns=[target])
y = data_encoded[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared: {r2}")

feature_importance = np.abs(model.coef_)
feature_names = X_train.columns

importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("Top 10 Most Influential Features:")
print(importance_df.head(10))
