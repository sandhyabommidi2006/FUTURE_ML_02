import pandas as pd
import re
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load Dataset
df = pd.read_csv("customer_support_tickets.csv")

print("Dataset Loaded Successfully")
print(df.head())

# Remove empty rows
df = df.dropna(subset=['Ticket Description', 'Ticket Type'])

# Text Cleaning Function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

# Clean Ticket Description
df['Clean_Text'] = df['Ticket Description'].apply(clean_text)

# Features and Target
X = df['Clean_Text']
y = df['Ticket Type']

# Convert Text into Numbers
vectorizer = TfidfVectorizer(stop_words='english')

X_vectorized = vectorizer.fit_transform(X)

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(round(accuracy * 100, 2), "%")

# Detailed Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Test New Ticket
new_ticket = "I cannot login to my account"

new_ticket_clean = clean_text(new_ticket)

new_ticket_vector = vectorizer.transform([new_ticket_clean])

predicted_type = model.predict(new_ticket_vector)[0]

print("\nNew Ticket:")
print(new_ticket)

print("\nPredicted Ticket Type:")
print(predicted_type)

# Priority Logic

if predicted_type == "Technical issue":
    priority = "High"
elif predicted_type == "Billing inquiry":
    priority = "Medium"
else:
    priority = "Low"

print("Predicted Priority:", priority)
# Graph
plt.figure(figsize=(8,5))

df['Ticket Type'].value_counts().plot(kind='bar')

plt.title("Support Ticket Classification")
plt.xlabel("Ticket Type")
plt.ylabel("Count")

plt.tight_layout()
plt.show()
