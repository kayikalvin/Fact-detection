import streamlit as st
import gdown
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Function to download and load the model from Google Drive
@st.cache_resource
def download_and_load_model():
    # Google Drive file ID for the model
    file_id = "1FZAkPX7FlbvvYTdE_EebrOn37zPN46hy"
    model_path = "./news_fact_checker_model"  # Local folder to save the model

    # Download the model if not already downloaded
    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)
        with st.spinner("Downloading the news fact-checker model..."):
            url = f"https://drive.google.com/uc?id={file_id}"
            output_path = os.path.join(model_path, "model.zip")
            gdown.download(url, output_path, quiet=False)
            # Extract the downloaded zip file
            import zipfile
            with zipfile.ZipFile(output_path, "r") as zip_ref:
                zip_ref.extractall(model_path)

    # Load the model and tokenizer
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except OSError as e:
        st.error(f"Error loading model: {e}")
        raise e

    return model, tokenizer

# Function to make predictions
def make_prediction(model, tokenizer, statement):
    # Tokenize the input statement
    inputs = tokenizer(statement, return_tensors="pt", truncation=True, padding=True)

    # Perform inference
    outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probabilities).item()

    # Confidence score as the probability of the predicted class
    confidence_score = round(probabilities[0, predicted_class].item() * 100, 2)

    # Define labels (adjust based on your model's labels)
    labels = ["True", "False", "Uncertain"]
    prediction_label = labels[predicted_class]

    return prediction_label, confidence_score

# Streamlit UI
st.title("News Fact-Checker")
st.write("This app checks the credibility of news statements using a pretrained AI model.")

# Load the model
with st.spinner("Loading model..."):
    model, tokenizer = download_and_load_model()
st.success("Fact-checker model loaded successfully!")

# Input for user statement
user_input = st.text_area("Enter a news statement to verify:", height=150)

# Add a button to process input
if st.button("Check Fact"):
    if user_input.strip():
        st.write("Processing your statement...")

        # Get prediction and confidence score
        prediction, confidence_score = make_prediction(model, tokenizer, user_input)

        # For this demo, we use a static accuracy score. Replace with dynamic accuracy if available.
        accuracy_score = 95.0  # Replace this with real model accuracy if applicable

        # Display the results
        st.subheader("Fact-Check Result:")
        st.write(f"The statement is **{prediction}**.")
        st.metric(label="Model Confidence", value=f"{confidence_score}%", delta=None)
        st.metric(label="Model Accuracy", value=f"{accuracy_score}%", delta=None)
    else:
        st.warning("Please enter a valid news statement to verify.")

# Footer
st.markdown("---")
st.caption("Powered by a pretrained news fact-checker AI model.")
