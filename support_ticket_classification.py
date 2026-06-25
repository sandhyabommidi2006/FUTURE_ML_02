
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

# ----------------------------
# Load Dataset
# ----------------------------

df = pd.read_csv("IT Service Ticket Classification Dataset .csv")

print("=" * 50)
print("Dataset Loaded Successfully!")
print("=" * 50)

print("\nFirst 5 Rows")
print(df.head())

print("\nDataset Shape")
print(df.shape)

print("\nColumns")
print(df.columns.tolist())

print("\nMissing Values")
print(df.isnull().sum())

# ----------------------------
# Category Distribution
# ----------------------------

print("\nTicket Categories")
print(df["Topic_group"].value_counts())

plt.figure(figsize=(10,5))

df["Topic_group"].value_counts().plot(kind="bar")

plt.title("IT Service Ticket Categories")
plt.xlabel("Category")
plt.ylabel("Number of Tickets")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()

# ----------------------------
# Prepare Data
# ----------------------------

X = df["Document"]
y = df["Topic_group"]

# Split dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# ----------------------------
# TF-IDF
# ----------------------------

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X_train_vector = vectorizer.fit_transform(X_train)

X_test_vector = vectorizer.transform(X_test)

# ----------------------------
# Train Model
# ----------------------------

print("\nTraining Model...")

model = LinearSVC(
    random_state=42,
    max_iter=5000
)

model.fit(X_train_vector, y_train)

print("Model Training Completed!")

# ----------------------------
# Prediction
# ----------------------------

y_pred = model.predict(X_test_vector)

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("MODEL ACCURACY")
print("==============================")
print(f"{accuracy*100:.2f}%")

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")

print(classification_report(y_test, y_pred))

# ----------------------------
# Confusion Matrix
# ----------------------------

ConfusionMatrixDisplay.from_estimator(
    model,
    X_test_vector,
    y_test,
    xticks_rotation=90
)

plt.title("Confusion Matrix")

plt.tight_layout()

plt.show()

# ----------------------------
# New Ticket Prediction
# ----------------------------

print("\n==============================")
print("NEW TICKET PREDICTION")
print("==============================")

new_ticket = [
    "Unable to login to my company email after password reset"
]

new_vector = vectorizer.transform(new_ticket)

prediction = model.predict(new_vector)

print("Ticket:")
print(new_ticket[0])

print("\nPredicted Category:")
print(prediction[0])

# ----------------------------
# Priority Prediction
# ----------------------------

def get_priority(ticket):

    ticket = ticket.lower()

    if any(word in ticket for word in ["urgent", "server", "crash", "critical"]):
        return "High"

    elif any(word in ticket for word in ["login", "password", "email"]):
        return "Medium"

    else:
        return "Low"

print("\nPredicted Priority:")
print(get_priority(new_ticket[0]))

print("\nProject Completed Successfully!")