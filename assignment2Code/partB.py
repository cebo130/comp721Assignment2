from sklearn.svm import SVC
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

# Train the SVM model with adjustable learning rate and print accuracy
model_svm = SVC(kernel='linear', C=1, gamma='auto')
model_svm.fit(X_train, y_train)

# Make predictions
predictions_svm = model_svm.predict(X_test)
# Calculate the number of complaints and compliments
complaints_svm = sum([1 for i in range(len(predictions_svm)) if predictions_svm[i] == 0 and y_test[i] == 1])
compliments_svm = sum([1 for i in range(len(predictions_svm)) if predictions_svm[i] == 1 and y_test[i] == 1])

# Add cases where predictions and ground truth do not match for both complaints and compliments
for i in range(len(predictions_svm)):
    if predictions_svm[i] == 0 and y_test[i] == 0:
        compliments_svm += 1
    elif predictions_svm[i] == 1 and y_test[i] == 0:
        complaints_svm += 1

# Calculate accuracy
accuracy_svm = accuracy_score(y_test, predictions_svm)

print("Number of complaints:", complaints_svm)
print("Number of compliments:", compliments_svm)
print("Accuracy:", accuracy_svm)
