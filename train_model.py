import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load the processed data from the pickle file
with open('data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

# Convert data and labels to numpy arrays
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Split data into training and testing sets (80% training, 20% testing)
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

# Initialize a RandomForestClassifier model
model = RandomForestClassifier()

# Train the model
print("Training the model...")
model.fit(x_train, y_train)
print("Model training complete.")

# Make predictions on the test set
y_predict = model.predict(x_test)

# Calculate the accuracy of the model
score = accuracy_score(y_predict, y_test)

print(f'\nAccuracy on the test set: {score * 100:.2f}%')

# Save the trained model to a file
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("Successfully saved the trained model to model.p")