import joblib


# Load feature extractor
features = joblib.load(
    "models/tfidf_features.pkl"
)

# Load trained model
model = joblib.load(
    "models/spam_model.pkl"
)


print("==============================")
print("     EMAIL SPAM DETECTOR")
print("==============================")
print("Type an email to analyze.")
print("Type 'exit' to quit.\n")


while True:

    message = input("Enter email: ")

    if message.lower() == "exit":
        print("Goodbye!")
        break

    if not message.strip():
        print("Please enter an email.\n")
        continue


    # Convert email into features
    message_features = features.transform(
        [message]
    )


    # Prediction
    prediction = model.predict(
        message_features
    )[0]


    # Probability
    probabilities = model.predict_proba(
        message_features
    )[0]


    if prediction == 1:

        confidence = probabilities[1] * 100

        print("\n🚨 SPAM")
        print(
            f"Confidence: {confidence:.2f}%"
        )

    else:

        confidence = probabilities[0] * 100

        print("\n✅ NOT SPAM")
        print(
            f"Confidence: {confidence:.2f}%"
        )


    print("--------------------------------\n")