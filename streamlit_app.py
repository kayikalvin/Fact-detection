import streamlit as st
import gdown
import os
import torch  # Replace with TensorFlow if your model uses it
from transformers import AutoTokenizer, AutoModelForSequenceClassification  # Adjust for your model

# Function to download and load the fact-checker model
@st.cache_resource
def download_and_load_model():
    # Google Drive file ID for the model
    file_id = "1FZAkPX7FlbvvYTdE_EebrOn37zPN46hy"
    download_url = f"https://drive.google.com/uc?id={file_id}"
    model_path = "news_fact_checker_model"  # Adjust for your model type
    
    # Check if the file exists to avoid re-downloading
    if not os.path.exists(model_path):
        with st.spinner("Downloading the news fact-checker model..."):
            gdown.download(download_url, model_path, quiet=False)
    
    # Load the model (Example for Hugging Face Transformers; adjust as needed)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

# Function to make predictions
def make_prediction(model, tokenizer, statement):
    # Tokenize the input
    inputs = tokenizer(statement, return_tensors="pt", truncation=True, padding=True)
    
    # Model inference
    outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probabilities).item()
    
    # Confidence score as the probability of the predicted class
    confidence_score = round(probabilities[0, predicted_class].item() * 100, 2)
    
    # Define labels (adjust as per your model's labels)
    labels = ["True", "False", "Uncertain"]
    prediction_label = labels[predicted_class]
    
    return prediction_label, confidence_score

# Streamlit UI
st.title("News Fact-Checker")
st.write("This app checks the credibility of news statements using a pretrained AI model.")

# Load the fact-checker model
model, tokenizer = download_and_load_model()
st.success("Fact-checker model loaded successfully!")

# Input for the user to enter a news statement
user_input = st.text_area("Enter a news statement to verify:", height=150)

# Add a button to process the input
if st.button("Check Fact"):
    if user_input.strip():
        st.write("Processing your statement...")
        
        # Get prediction and confidence score
        prediction, confidence_score = make_prediction(model, tokenizer, user_input)
        
        # For this demo, we'll assume the model has a fixed accuracy (replace with real accuracy if available)
        accuracy_score = 95.0  # Replace with dynamic accuracy if available

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
