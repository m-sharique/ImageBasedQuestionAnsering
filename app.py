import requests
import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-capfilt-large")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-capfilt-large")

def answer_question(image, question):
    """Answers a question about an image using the Blip model."""
    inputs = processor(image, question, return_tensors="pt")
    out = model.generate(**inputs)
    answer = processor.decode(out[0], skip_special_tokens=True)
    return answer.title()  # Capitalize the first letter of the answer

st.title("Explore Art with BLIP")  # Change title to reflect art theme
st.subheader("Ask a question about the artwork below")

with st.sidebar:
    image_file = st.file_uploader("Upload an Artwork", type=["jpg", "jpeg", "png", "webp"])
    question = st.text_area("Your Art Exploration Question")

if image_file and question:
    image = Image.open(image_file)
    st.image(image, caption='Uploaded Artwork', use_column_width=True)
    answer = answer_question(image, question)
    st.write(f"**Answer:** {answer}")  # Emphasize answer with bold text
