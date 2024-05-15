import requests
import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-capfilt-large")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-capfilt-large")

def blipify_image(image, question):
    """BLIPifies an image to answer your art exploration question."""
    inputs = processor(image, question, return_tensors="pt")
    out = model.generate(**inputs)
    answer = processor.decode(out[0], skip_special_tokens=True)
    return answer.title()  # Capitalize the first letter of the answer

st.title("BLIPify: Unlock the Secrets of Art")

st.subheader("BLIPify your artistic curiosity with a question below")

with st.sidebar:
    image_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "webp"])
    question = st.text_input("Your BLIPification Question")

if image_file and question:
    image = Image.open(image_file)
    st.image(image)
    answer = blipify_image(image, question)
    st.write(f"**BLIPified Answer:** {answer}")  # Emphasize answer with bold text
