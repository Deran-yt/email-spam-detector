import streamlit as st
import joblib


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="centered"
)


# ==========================================
# LOAD MODEL
# ==========================================

features = joblib.load("models/tfidf_features.pkl")
model = joblib.load("models/spam_model.pkl")


# ==========================================
# TITLE
# ==========================================

st.title("📧 Email Spam Detector")

st.write(
    "Enter an email message below and the machine learning model "
    "will classify it as Spam or Not Spam."
)

st.divider()


# ==========================================
# EMAIL INPUT
# ==========================================

email = st.text_area(
    "✉️ Enter your email",
    height=220,
    placeholder="Paste your email message here..."
)


# ==========================================
# ANALYZE BUTTON
# ==========================================

if st.button("🔍 Analyze Email", use_container_width=True):

    if not email.strip():

        st.warning("Please enter an email message.")

    else:

        # Convert email into TF-IDF features
        email_features = features.transform([email])

        # Prediction
        prediction = model.predict(email_features)[0]

        # Probabilities
        probabilities = model.predict_proba(email_features)[0]

        not_spam_probability = probabilities[0] * 100
        spam_probability = probabilities[1] * 100


        st.divider()


        # ==========================================
        # RESULT
        # ==========================================

        if prediction == 1:

            st.error("🚨 SPAM EMAIL")

            st.metric(
                "Spam Probability",
                f"{spam_probability:.2f}%"
            )

        else:

            st.success("✅ NOT SPAM")

            st.metric(
                "Not Spam Probability",
                f"{not_spam_probability:.2f}%"
            )


        # ==========================================
        # PROBABILITY BREAKDOWN
        # ==========================================

        st.subheader("📊 Prediction Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Not Spam",
                f"{not_spam_probability:.2f}%"
            )

        with col2:
            st.metric(
                "Spam",
                f"{spam_probability:.2f}%"
            )


        # Probability bars
        st.write("Not Spam")
        st.progress(int(not_spam_probability))

        st.write("Spam")
        st.progress(int(spam_probability))


        # ==========================================
        # BASIC EMAIL STATISTICS
        # ==========================================

        st.subheader("📝 Email Statistics")

        word_count = len(email.split())
        character_count = len(email)

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Words",
                word_count
            )

        with col2:
            st.metric(
                "Characters",
                character_count
            )