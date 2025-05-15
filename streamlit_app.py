import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Set page config (title and favicon)
st.set_page_config(page_title="My Portfolio", page_icon="static/favicon_io/favicon.ico")

# Cache model loading to avoid reloading on every rerun
@st.cache_resource
def load_models():
    try:
        with open('models/spam_model.pkl', 'rb') as f:
            spam_model = pickle.load(f)
        with open('models/count_vectorizer.pkl', 'rb') as f:
            count_vectorizer = pickle.load(f)
        return spam_model, count_vectorizer
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

# Load models
spam_model, count_vectorizer = load_models()

# Sidebar for navigation
page = st.sidebar.selectbox("Choose a page", ["Home", "Spam Classifier", "Animals Classification", "Bones Fracture"])

# Home page
if page == "Home":
    st.title("Welcome to My Portfolio")
    st.write("Explore my projects using the sidebar.")
    st.markdown("""
    This portfolio showcases my work in machine learning and web development.
    - **Spam Classifier**: Detect spam emails using a Naive Bayes model.
    - **Animals Classification**: Coming soon!
    - **Bones Fracture**: Coming soon!
    """)
    # Example image (adjust path if you have a specific image)
    try:
        st.image("static/images/placeholder.jpg", caption="Portfolio Image")
    except FileNotFoundError:
        st.write("No image found. Place an image in 'static/images/'.")

# Spam Classifier page
elif page == "Spam Classifier":
    st.title("Spam Classifier")
    if spam_model is None or count_vectorizer is None:
        st.error("Models not loaded. Please check the model files.")
    else:
        with st.form(key="spam_form"):
            email_text = st.text_area("Enter email text:", height=150)
            submit_button = st.form_submit_button("Classify")
            
            if submit_button and email_text:
                try:
                    # Transform input text
                    text_vector = count_vectorizer.transform([email_text])
                    # Predict
                    pred = spam_model.predict(text_vector)[0]
                    prediction = "Spam" if pred == 1 else "Not Spam"
                    st.success(f"Prediction: **{prediction}**")
                except Exception as e:
                    st.error(f"Error processing prediction: {e}")

# Animals Classification page (placeholder)
elif page == "Animals Classification":
    st.title("Animals Classification")
    st.write("This feature is under development.")
    st.markdown("""
    Future plans:
    - Upload images to classify animals.
    - Use a pre-trained deep learning model.
    """)

# Bones Fracture page (placeholder)
elif page == "Bones Fracture":
    st.title("Bones Fracture")
    st.write("This feature is under development.")
    st.markdown("""
    Future plans:
    - Upload X-ray images to detect fractures.
    - Implement a convolutional neural network.
    """)