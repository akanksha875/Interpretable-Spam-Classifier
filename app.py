import streamlit as st
import pickle
import lime
import lime.lime_text
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model and vectorizer
model = pickle.load(open("spam_classifier_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Define prediction function for LIME
def predict_proba(texts):
    X = vectorizer.transform(texts)   # Convert text to TF-IDF
    return model.predict_proba(X)

# Initialize LIME explainer
explainer = lime.lime_text.LimeTextExplainer(class_names=["Ham", "Spam"])

# Streamlit app
st.title("📩 Interpretable Spam Classifier: See Why a Message is Spam")
st.write("Enter an email/message below to check if it's spam or not:")

# Input message
message = st.text_area("Enter your message:")

if st.button("Classify"):
    if message.strip() == "":
        st.warning("⚠️ Please enter a message first.")
    else:
        # Prediction
        X = vectorizer.transform([message])
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0]

        if prediction == 1:
            st.error(f"🚨 Spam Detected! (Confidence: {probability[1]*100:.2f}%)")
        else:
            st.success(f"✅ Not Spam (Confidence: {probability[0]*100:.2f}%)")

        # LIME explanation
        exp = explainer.explain_instance(
            message, 
            predict_proba, 
            num_features=10
        )

        # Show explanation in Streamlit
        st.subheader("🔎 Explanation of Prediction")
        st.components.v1.html(exp.as_html(), height=800, scrolling=True)
