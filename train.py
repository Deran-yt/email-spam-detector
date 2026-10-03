import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


print("======================================")
print("       EMAIL SPAM DETECTOR TRAINING")
print("======================================")

# ======================================
# 1. LOAD ORIGINAL DATASET
# ======================================

print("\nLoading SMS Spam Collection...")

df = pd.read_csv(
    "data/spam.csv",
    encoding="latin-1"
)

# Dataset contains unnecessary columns
df = df.iloc[:, :2]
df.columns = ["label", "message"]

# Convert labels
df["label"] = df["label"].map({
    "spam": 1,
    "ham": 0
})

df = df.dropna()

print("Original dataset:", len(df))


# ======================================
# 2. ADD REALISTIC LEGITIMATE EMAILS
# ======================================

legitimate_messages = [

    "Dear customer, your account needs verification.",
    "Dear customer, please verify your account to continue using our service.",
    "Your account verification is required to complete your registration.",
    "Please verify your email address to activate your account.",
    "Your account has been successfully verified.",
    "Your account information has been updated successfully.",
    "Your monthly account statement is now available.",
    "Dear customer, your account statement is ready to view.",
    "Your bank account statement is available in your online banking account.",
    "Please review your account details and contact support if anything is incorrect.",
    "Your password was successfully changed.",
    "Your password has been reset successfully.",
    "Your login details have been updated.",
    "We received your request to update your account information.",
    "Your profile has been successfully updated.",
    "Thank you for registering with our service.",
    "Welcome to our service. Your account has been created successfully.",
    "Your subscription has been renewed successfully.",
    "Your payment was successfully processed.",
    "Your order has been confirmed.",
    "Your order is being processed.",
    "Your order has been shipped.",
    "Your invoice is now available.",
    "Please find your invoice attached to this email.",
    "Your appointment has been confirmed.",
    "This is a reminder about your scheduled appointment.",
    "Your application has been successfully submitted.",
    "We have received your application and will review it shortly.",
    "Your request has been received successfully.",
    "Thank you for contacting customer support.",
    "Our support team has received your message.",
    "Please reply to this email if you need further assistance.",
    "Your security settings have been updated.",
    "A new login to your account was detected.",
    "We noticed a login from a new device.",
    "If you recognize this activity, no action is required.",
    "Please contact our support team if you do not recognize this activity.",
    "Your verification code is 482913.",
    "Use the verification code sent to your registered phone number.",
    "Your one-time password is required to complete the login.",
    "Your transaction has been completed successfully.",
    "Your payment receipt is attached.",
    "Thank you for your payment.",
    "Your refund has been processed.",
    "Your request has been approved.",
    "Your application status has been updated.",
    "Please log in to your account to view the latest information.",
    "Your account is active and ready to use.",
    "Dear customer, we are writing to inform you about an update to your account.",
    "Please review the information provided in your account dashboard."
]


# ======================================
# 3. ADD REALISTIC SPAM/PHISHING EMAILS
# ======================================

spam_messages = [

    "URGENT! Your account will be permanently closed. Verify immediately by clicking this link.",
    "Your account has been compromised! Click now to verify your identity.",
    "Congratulations! You have won ₹50,000. Click now to claim your prize.",
    "You have been selected to receive a FREE iPhone. Claim your prize now!",
    "URGENT: Verify your account immediately or your account will be suspended.",
    "Your bank account has been selected for a cash reward. Click here now.",
    "Congratulations! You are our lucky winner. Claim your FREE reward today.",
    "Click this link immediately to receive your prize money.",
    "You have won a lottery prize of ₹1,00,000. Send your details to claim it.",
    "Your account will be blocked today unless you verify your information immediately.",
    "Act now! Your account requires immediate verification. Click the link below.",
    "FREE CASH! You have been selected for a special reward. Claim now!",
    "Congratulations! You have won a brand new smartphone. Click to claim.",
    "URGENT SECURITY ALERT! Verify your password immediately using the link below.",
    "Your account has been locked. Click here to unlock it immediately.",
    "You are eligible for a guaranteed cash reward. Claim it now.",
    "Limited time offer! Click now to receive your free gift.",
    "You have been chosen as a winner. Send your bank details to receive the money.",
    "URGENT! Your payment has failed. Click this link to update your bank details.",
    "Verify your bank account immediately to avoid permanent suspension.",
    "Congratulations! You won ₹25,000 cash. Click here to collect your reward.",
    "Claim your exclusive FREE prize before the offer expires today.",
    "Your account will be deleted unless you verify your information within 24 hours."
]


# Convert additional examples into DataFrames

legitimate_df = pd.DataFrame({
    "label": [0] * len(legitimate_messages),
    "message": legitimate_messages
})

spam_df = pd.DataFrame({
    "label": [1] * len(spam_messages),
    "message": spam_messages
})


# ======================================
# 4. COMBINE DATASETS
# ======================================

df = pd.concat(
    [
        df[["label", "message"]],
        legitimate_df,
        spam_df
    ],
    ignore_index=True
)

df = df.drop_duplicates(subset=["message"])

print("Final dataset:", len(df))
print("Legitimate messages:", (df["label"] == 0).sum())
print("Spam messages:", (df["label"] == 1).sum())


# ======================================
# 5. SPLIT DATA
# ======================================

X = df["message"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ======================================
# 6. CREATE TF-IDF FEATURES
# ======================================

print("\nCreating text features...")

features = FeatureUnion([
    
    (
        "word_features",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
            max_df=0.98
        )
    ),

    (
        "character_features",
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=2,
            sublinear_tf=True
        )
    )
])


X_train_features = features.fit_transform(X_train)
X_test_features = features.transform(X_test)

print("Feature matrix:", X_train_features.shape)


# ======================================
# 7. TRAIN MODEL
# ======================================

print("\nTraining Logistic Regression model...")

model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    C=2.0
)

model.fit(
    X_train_features,
    y_train
)

print("Training completed!")


# ======================================
# 8. EVALUATE MODEL
# ======================================

predictions = model.predict(X_test_features)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n======================================")
print("             MODEL RESULTS")
print("======================================")

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        target_names=["Not Spam", "Spam"]
    )
)


# ======================================
# 9. SAVE MODEL
# ======================================

os.makedirs("models", exist_ok=True)

joblib.dump(
    features,
    "models/tfidf_features.pkl"
)

joblib.dump(
    model,
    "models/spam_model.pkl"
)


print("\n======================================")
print("              SUCCESS")
print("======================================")

print("Model saved:")
print("models/tfidf_features.pkl")
print("models/spam_model.pkl")

print("\nTraining finished successfully!")