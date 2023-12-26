import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the flattened images and labels
with open('Shape_detection/flattened_images.pkl', 'rb') as f_img, open('Shape_detection/labels.pkl', 'rb') as f_lbl:
    flattened_images = pickle.load(f_img)
    labels = pickle.load(f_lbl)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(flattened_images, labels, test_size=0.3, random_state=42)

# Initialize and train the SVM model
svm_model = SVC()
svm_model.fit(X_train, y_train)

# Evaluate the model
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")
