import os

import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image

from gemini import (load_model, load_vision_model, load_embedding_model, load_qa_model)

working_dir = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="Gemini AI",
    layout="wide",
    page_icon="🧠",
)

with st.sidebar:
    selected = option_menu("Gemini AI", [
        "ChatBot",
        "Image Captioning",
        "Embed Text",
        "Ask me anything"
    ], menu_icon="robot", icons=["chat-dots-fill", "image-fill", "textarea-t", "patch-question-fill"], default_index=0)

# Function to translate role between gemini-pro and streamlit terminology
def translate_role_for_streamlit(user_role):
    if user_role == "model":
        return "assistant"
    else:
        return user_role

# Chatbot
if selected == "ChatBot":
    model = load_model()

    # Streamlit page title
    st.title("🤖 ChatBot")

    # Initialize chat session in streamlit if not already present
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = model.start_chat(history=[])

    # Display the chat history
    for message in st.session_state.chat_session.history:
        with st.chat_message(translate_role_for_streamlit(message.role)):
            st.markdown(message.parts[0].text)

    # Input field for user's message
    user_prompt = st.chat_input("Ask Gemini-Pro...")

    if user_prompt:
        st.chat_message("user").markdown(user_prompt)

        response_placeholder = st.chat_message("assistant").empty()
        response_placeholder.markdown("Thinking...")

        gemini_response = st.session_state.chat_session.send_message(user_prompt)
        response_placeholder.markdown(gemini_response.text)


# Image Captioning
if selected == "Image Captioning":

    st.title("📷 Image Captioning")

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        user_prompt = st.text_input("Ask something about the image")
        prompt = user_prompt if user_prompt else "Generate a caption for this image?"

        image = Image.open(uploaded_image)
        resized_image = image.resize((800, 500))
        st.image(resized_image, caption=f"Image size: {resized_image.size}")

        if st.button("Generate Caption"):
            placeholder = st.empty()
            placeholder.text("Captioning...")

            caption = load_vision_model(prompt, image)

            placeholder.info(caption)


if selected == "Embed Text":

    st.title("🔠 Embed Text")

    user_input = st.text_area(label="", placeholder="Enter your text here...")

    if st.button("Get Embedding"):
        if user_input.strip():
            placeholder = st.empty()
            placeholder.text("Embedding...")

            response = load_embedding_model(user_input)
            st.markdown(response)
            placeholder.empty()
        else:
            st.warning("Please enter some text...")

if selected == "Ask me anything":

    st.title("❓ Ask me a question")

    user_input = st.text_area(label="", placeholder="Enter your question here...")

    if st.button("Get Answer"):
        if user_input.strip():
            placeholder = st.empty()
            placeholder.text("Answering...")

            response = load_qa_model(user_input)
            st.markdown(response)
            placeholder.empty()
        else:
            st.warning("Please enter your question...")
