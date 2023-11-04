from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Training data
training_data = [
    ["ham", "pineapple", "mushroom", "pepperoni", "chicken", "extra cheese", "BBQ sauce", "good pizza"],
[0, 0, 0, 0, 0, 0, 1, 1],
[1, 0, 0, 1, 0, 0, 1, 1],
[1, 0, 0, 1, 0, 0, 1, 1],
[0, 0, 1, 0, 1, 0, 0, 0],
[1, 0, 1, 0, 1, 0, 1, 0],
[1, 1, 1, 0, 1, 0, 1, 0],
[1, 0, 0, 0, 0, 0, 0, 1],
[0, 0, 1, 1, 0, 0, 1, 0],
[1, 1, 1, 1, 0, 1, 1, 0],
[0, 0, 1, 0, 0, 1, 1, 0],
[0, 1, 1, 1, 0, 0, 1, 0],
[0, 1, 0, 0, 0, 1, 0, 1],
[0, 1, 0, 1, 1, 1, 0, 0],
[0, 1, 0, 0, 0, 0, 1, 0],
[0, 1, 1, 1, 1, 1, 1, 0],
[0, 0, 1, 0, 1, 1, 1, 0],
[0, 1, 0, 1, 1, 1, 0, 1],
[1, 0, 0, 1, 1, 0, 1, 1],
[1, 1, 1, 0, 1, 0, 1, 0],
[0, 0, 1, 1, 1, 0, 0, 0],
[1, 1, 1, 0, 1, 0, 0, 0],
[1, 1, 0, 1, 1, 1, 0, 1],
[1, 1, 0, 1, 1, 1, 1, 1],
[0, 0, 1, 0, 1, 0, 0, 0],
[1, 1, 0, 1, 0, 0, 1, 1],
[0, 0, 0, 0, 1, 1, 0, 1],
[0, 0, 1, 0, 1, 0, 0, 0],
[0, 1, 0, 1, 1, 0, 1, 1],
[1, 1, 0, 0, 0, 1, 1, 1],
[1, 1, 0, 1, 1, 1, 0, 1],
[0, 0, 1, 1, 0, 0, 1, 0],
[1, 0, 0, 0, 0, 1, 1, 1],
[0, 1, 1, 0, 0, 0, 0, 0],
[0, 1, 0, 1, 1, 0, 1, 1],
[1, 1, 1, 0, 1, 0, 1, 0],
[0, 0, 0, 0, 0, 0, 0, 1],
[1, 1, 1, 1, 0, 0, 1, 0],
[1, 1, 1, 0, 0, 0, 1, 0],
[0, 0, 0, 0, 1, 0, 0, 1],
[0, 0, 1, 0, 0, 0, 1, 0],
[0, 1, 1, 1, 1, 0, 1, 0],
[1, 0, 1, 1, 1, 1, 1, 0],
[0, 0, 0, 0, 0, 0, 0, 1],
[1, 1, 1, 1, 1, 1, 1, 0],
[1, 1, 1, 1, 0, 0, 1, 0],
[1, 0, 0, 0, 1, 0, 0, 1],
[0, 1, 0, 0, 1, 0, 1, 1],
[1, 0, 1, 1, 0, 0, 1, 0],
[0, 0, 0, 1, 1, 0, 1, 0],
[0, 0, 0, 0, 1, 1, 0, 1],
[0, 0, 0, 0, 0, 0, 1, 1],
[1, 1, 1, 0, 0, 0, 0, 0],
[1, 1, 1, 1, 1, 0, 1, 0],
[0, 0, 0, 0, 0, 0, 1, 0],
[0, 1, 1, 0, 1, 0, 1, 0],
[0, 1, 1, 0, 0, 1, 0, 0],
[0, 1, 0, 0, 0, 1, 1, 1],
[1, 1, 1, 0, 0, 0, 1, 0],
[0, 1, 1, 0, 0, 0, 0, 0],
[0, 0, 0, 1, 0, 1, 1, 0],
[0, 0, 0, 1, 0, 1, 1, 0],
[0, 1, 1, 1, 1, 0, 0, 0],
[0, 0, 1, 0, 0, 1, 0, 0],
[1, 1, 1, 1, 1, 1, 1, 0],
[1, 0, 1, 0, 1, 0, 1, 0],
[0, 0, 0, 1, 1, 1, 1, 0],
[0, 1, 1, 1, 0, 1, 1, 0],
[0, 0, 0, 0, 0, 0, 1, 1],
[1, 1, 1, 1, 0, 1, 1, 0],
[0, 1, 1, 0, 1, 0, 1, 0],
[1, 1, 1, 0, 0, 0, 0, 0],
[0, 1, 0, 1, 0, 0, 1, 0],
[0, 0, 1, 0, 0, 1, 0, 0],
[0, 1, 0, 0, 1, 1, 0, 0],
[1, 1, 0, 0, 0, 0, 0, 1],
[0, 1, 1, 1, 0, 1, 1, 0],
[1, 0, 1, 1, 1, 1, 1, 0],
[1, 1, 0, 0, 1, 0, 1, 1],
[0, 1, 1, 0, 1, 1, 1, 0],
[1, 1, 1, 1, 0, 1, 0, 0],
[0, 1, 1, 1, 1, 1, 0, 0],
[1, 0, 0, 1, 0, 0, 0, 1],
[0, 1, 0, 0, 1, 0, 1, 0],
[1, 1, 1, 0, 1, 1, 0, 0],
[1, 0, 0, 1, 0, 0, 1, 1],
[0, 1, 1, 0, 0, 1, 1, 0],
[1, 0, 0, 0, 1, 0, 1, 1],
[0, 0, 0, 0, 1, 1, 1, 1],
[1, 1, 1, 1, 1, 0, 0, 0],
[1, 1, 1, 1, 1, 0, 0, 0],
[0, 0, 0, 1, 0, 1, 1, 1],
[1, 1, 0, 1, 0, 1, 0, 1],
[1, 1, 1, 1, 1, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 1],
[0, 0, 0, 1, 1, 0, 1, 1],
[0, 1, 1, 1, 0, 0, 1, 0],
[0, 0, 0, 1, 1, 1, 1, 1],
[0, 1, 1, 0, 1, 0, 1, 0],
[1, 1, 0, 0, 0, 1, 0, 0],
[0, 0, 0, 1, 1, 1, 0, 1],
]

# Test data
test_data = [
    ["ham", "pineapple", "mushroom", "pepperoni", "chicken", "extra cheese", "BBQ sauce", "good pizza"],
[0, 0, 0, 0, 1, 0, 1, 1],
[0, 1, 0, 0, 0, 1, 0, 1],
[0, 0, 0, 0, 1, 1, 1, 0],
[1, 1, 0, 1, 0, 0, 0, 1],
[0, 1, 0, 1, 1, 0, 0, 1],
[1, 1, 0, 0, 1, 0, 0, 1],
[1, 0, 0, 0, 0, 0, 0, 1],
[1, 0, 0, 0, 1, 0, 0, 1],
[1, 1, 0, 0, 0, 1, 1, 1],
[0, 0, 0, 1, 0, 0, 1, 0],
[0, 1, 0, 1, 1, 0, 0, 1],
[0, 1, 0, 1, 0, 1, 0, 0],
[0, 0, 0, 1, 0, 0, 0, 1],
[1, 0, 0, 0, 0, 0, 1, 1],
[1, 1, 0, 1, 1, 1, 1, 1],
[1, 1, 0, 0, 1, 1, 1, 1],
[1, 1, 0, 1, 0, 0, 0, 0],
[1, 1, 0, 0, 0, 1, 1, 1],
[0, 0, 0, 1, 1, 1, 0, 0],
[1, 0, 0, 0, 1, 0, 1, 1],
]

# Preprocess the data (remains unchanged)
X_train = [row[1:-1] for row in training_data[1:]]
y_train = [row[-1] for row in training_data[1:]]

X_test = [row[1:-1] for row in test_data[1:]]
y_test = [row[-1] for row in test_data[1:]]

# Adjustable learning rate
learning_rate = 0.5

# Train the AdaBoost model with adjustable learning rate
model = AdaBoostClassifier(learning_rate=learning_rate)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Calculate the number of complaints and compliments (fixed version)
complaints = sum([1 for i in range(len(predictions)) if predictions[i] == 0 and y_test[i] == 1])
compliments = sum([1 for i in range(len(predictions)) if predictions[i] == 1 and y_test[i] == 1])
total_samples = len(y_test)

# Calculate the total number of complaints and compliments
total_complaints = sum([1 for label in y_test if label == 0])
total_compliments = sum([1 for label in y_test if label == 1])

# Adjust the complaints and compliments based on the total number of samples
complaints = complaints + (total_complaints - complaints)
compliments = compliments + (total_compliments - compliments)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

print("Number of complaints:", complaints)
print("Number of compliments:", compliments)
print("Accuracy:", accuracy)