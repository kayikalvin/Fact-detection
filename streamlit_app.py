import streamlit as st
import gdown
import os

# Function to download and load the fact-checker model
@st.cache_resource  # Cache the model to download and load it only once
def download_and_load_model():
    # Google Drive file ID for the model
    file_id = "1FZAkPX7FlbvvYTdE_EebrOn37zPN46hy"
    download_url = f"https://drive.google.com/uc?id={file_id}"
    output_path = "news_fact_checker_model.pt"  # Adjust the file name if needed
    
    # Check if the file exists to avoid re-downloading
    if not os.path.exists(output_path):
        with st.spinner("Downloading the news fact-checker model..."):
            gdown.download(download_url, output_path, quiet=False)

    # Load the model (adjust based on your framework, e.g., PyTorch or TensorFlow)
    import torch
    model = torch.load(output_path, map_location=torch.device('cpu'))
    return model

# Streamlit UI
st.title("News Fact-Checker")
st.write("This app checks the credibility of news statements using a pretrained AI model.")

# Load the fact-checker model
model = download_and_load_model()
st.success("Fact-checker model loaded successfully!")

# Input for the user to enter a news statement
user_input = st.text_area("Enter a news statement to verify:", height=150)

# Add a button to process the input
if st.button("Check Fact"):
    if user_input.strip():
        st.write("Processing your statement...")

        # Example model inference (adjust based on your model logic)
        # Replace the following lines with actual prediction logic
        # Example for a PyTorch model:
        # prediction = model.predict(user_input)
        # For now, we'll simulate predictions for demonstration purposes:
        import random
        prediction = random.choice(["True", "False", "Uncertain"])

        # Display the result
        st.subheader("Fact-Check Result:")
        st.write(f"The statement is **{prediction}**.")
    else:
        st.warning("Please enter a valid news statement to verify.")

# Footer
st.markdown("---")
st.caption("Powered by a pretrained news fact-checker AI model.")
