import streamlit as st
import os
import gdown
import zipfile
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Google Drive file ID for the model
file_id = "1FZAkPX7FlbvvYTdE_EebrOn37zPN46hy"
model_path = "./news_fact_checker_model"  # Local folder to save the model

def download_and_load_model():
    # Check if the model folder exists; if not, download and extract the model
    if not os.path.exists(model_path):
        st.write(f"Model folder '{model_path}' not found. Downloading model...")
        os.makedirs(model_path, exist_ok=True)
        
        # Download the model from Google Drive
        url = f"https://drive.google.com/uc?id={file_id}"
        output_path = os.path.join(model_path, "model.zip")
        
        try:
            # Download the model
            gdown.download(url, output_path, quiet=False)
            st.write(f"Model downloaded to: {output_path}")

            # Extract the downloaded zip file
            with zipfile.ZipFile(output_path, "r") as zip_ref:
                zip_ref.extractall(model_path)
                st.write(f"Model extracted to: {model_path}")

        except Exception as e:
            st.error(f"Error downloading or extracting the model: {e}")
            raise e

    # Verify the model folder contains necessary files
    if os.path.exists(model_path):
        model_files = os.listdir(model_path)
        st.write(f"Model files found: {model_files}")

        # Ensure model files like config.json and pytorch_model.bin exist
        if "config.json" not in model_files or "pytorch_model.bin" not in model_files:
            st.error("Model files are missing required files (config.json or pytorch_model.bin).")
            raise FileNotFoundError("Missing model files.")
    else:
        st.error(f"Model folder not found: {model_path}")
        raise FileNotFoundError(f"Model folder not found: {model_path}")

    # Try loading the model and tokenizer
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        st.success("Model loaded successfully.")
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        raise e


# Streamlit app UI and logic
st.title("News Fact-Checker")
st.write("This app checks the credibility of news statements using a pretrained AI model.")

# Call the model download and loading function
try:
    model, tokenizer = download_and_load_model()
except Exception as e:
    st.error(f"Error: {e}")

# Example of input and prediction
user_input = st.text_area("Enter a news statement to check:")
if user_input:
    inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    try:
        with st.spinner("Checking the statement..."):
            outputs = model(**inputs)
            predictions = outputs.logits.argmax(axis=-1).item()  # Get the predicted class
            st.write(f"Prediction: {predictions}")
            # Display dummy confidence for the prediction (can replace with real prediction confidence)
            confidence_score = 0.85  # Example confidence
            st.write(f"Confidence score: {confidence_score * 100:.2f}%")
    except Exception as e:
        st.error(f"Error making prediction: {e}")
