import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the dataset
data = pd.read_csv('indian_liver_patient.csv')

# Replace gender text with numeric values
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

# Handle missing values if any
data.fillna(data.mean(), inplace=True)

# Define features and target
X = data.drop(columns=['Dataset'])  # Replace 'Dataset' with the actual target column name
y = data['Dataset']  # Replace 'Dataset' with the actual target column name

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the RandomForest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model
with open('indian_liver_patient.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

print("Model trained and saved as indian_liver_patient.pkl")
