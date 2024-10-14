import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Load the dataset
data = pd.read_excel('Flight_Tickets_data.xlsx')
# Select relevant columns
required_columns = ['Airline', 'Source', 'Destination', 'Route', 'Duration', 'Total-Stop', 'Price', 'Date_of_Journey']
missing_columns = [col for col in required_columns if col not in data.columns]
if missing_columns:
    raise ValueError(f"Missing columns in the dataset: {missing_columns}")
# Prepare data for the model
X = data[['Airline', 'Source', 'Destination', 'Route', 'Total-Stop']]
y = data['Price']

# Preprocess the data
categorical_features = ['Airline', 'Source', 'Destination', 'Route', 'Total-Stop']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])
# Create a pipeline that first transforms the data and then fits the model
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', LinearRegression())])
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train the model
pipeline.fit(X_train, y_train)
# Save the model
joblib.dump(pipeline, 'flight_price_model.pkl')
print('Model trained successfully')


