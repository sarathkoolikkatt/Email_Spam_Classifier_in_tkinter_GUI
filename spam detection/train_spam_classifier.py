import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split


dataset_path = "C:/Users/user/Desktop/softronics/spam detection/SMSSpamCollection"

emails = []
labels = []

with open(dataset_path, "r", encoding="utf-8") as file:
    for line in file:
        label, text = line.split("\t", 1)
        labels.append(1 if label == "spam" else 0)  # 1 = Spam, 0 = Ham
        emails.append(text.strip())

# Step 2: Split data into training and testing sets (optional, for evaluation purposes)
X_train, X_test, y_train, y_test = train_test_split(emails, labels, test_size=0.2, random_state=42)

# Step 3: Create a vectorizer
vectorizer = CountVectorizer()
X_train_features = vectorizer.fit_transform(X_train)  # Convert training emails to feature vectors

# Step 4: Train a classifier
model = MultinomialNB()
model.fit(X_train_features, y_train)

# Step 5: Save the vectorizer and model
# Save vectorizer
with open("count_vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

# Save classifier
with open("spam_classifier.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

print("Vectorizer and classifier saved successfully!")


X_test_features = vectorizer.transform(X_test)  # Transform test data
accuracy = model.score(X_test_features, y_test)
print(f"Model accuracy on test data: {accuracy * 100:.2f}%")
