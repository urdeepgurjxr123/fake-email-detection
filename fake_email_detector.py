# Step 1: Libraries import
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

# Step 2: Sample dataset (embedded)
data_dict = {
    'text': [
        "Congratulations! You've won a free iPhone. Click here to claim.",
        "Hey, are we meeting tomorrow?",
        "URGENT! Your account has been compromised. Reset your password now.",
        "Can we reschedule our meeting to next week?",
        "Win a $1000 gift card by clicking this link!",
        "Happy Birthday! Wishing you a wonderful day.",
        "Limited time offer! Buy now and get 50% off.",
        "Please find attached the project report for review.",
        "Get cheap loans instantly, no credit check required.",
        "Let's catch up for coffee this weekend."
    ],
    'label': [
        "spam", "ham", "spam", "ham", "spam",
        "ham", "spam", "ham", "spam", "ham"
    ]
}

data = pd.DataFrame(data_dict)

# Step 3: Preprocessing
data['label_num'] = data['label'].map({'ham':0, 'spam':1})
X = data['text']
y = data['label_num']

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 6: Naive Bayes Model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Step 7: Evaluation
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 8: Save model and vectorizer
pickle.dump(model, open('fake_email_model.pkl', 'wb'))
pickle.dump(vectorizer, open('tfidf_vectorizer.pkl', 'wb'))

# Step 9: Prediction function
def predict_email(text):
    vect_text = vectorizer.transform([text])
    pred = model.predict(vect_text)
    return "Spam/Fake Email" if pred[0]==1 else "Genuine Email"

# Step 10: Test examples
test_emails = [
    "Congratulations! You won a free trip to Paris!",
    "Hey, can you send me the assignment file?",
    "Your bank account is at risk! Verify immediately!"
]

for email in test_emails:
    print(f"Email: {email}\nPrediction: {predict_email(email)}\n")