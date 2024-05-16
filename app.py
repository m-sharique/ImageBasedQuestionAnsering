import requests
import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
import logging

# Setup logging
logging.basicConfig(filename='blipify.log', level=logging.ERROR)

processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-capfilt-large")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-capfilt-large")

def blipify_image(image, question):
    """BLIPifies an image to answer your art exploration question."""
    try:
        inputs = processor(image, question, return_tensors="pt")
        out = model.generate(**inputs)
        answer = processor.decode(out[0], skip_special_tokens=True)
        return answer.title()  # Capitalize the first letter of the answer
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return "Sorry, an error occurred while processing your question. Please try again."

st.title("BLIPify: Unlock the Secrets of Art")

st.subheader("BLIPify your artistic curiosity with a question below")

col1, col2 = st.columns(2)

with col1:
    image_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "webp"])
    question = st.text_input("Your BLIPification Question")

with col2:
    if image_file and question:
        try:
            image = Image.open(image_file)
            st.image(image, use_column_width=True)
            answer = blipify_image(image, question)
            st.write(f"**BLIPified Answer:** {answer}")  # Emphasize answer with bold text
        except Exception as e:
            st.error("Sorry, an error occurred while processing your question.")
            logging.error(f"Error in Streamlit app: {e}")
