import streamlit as st
from src.constants import Directories
from src.main import download_image as download

def download_image():
    content_url = st.text_input("url")
    content_url_image_name = st.text_input("image name")
    image_type = st.selectbox("Image type", [Directories.CONTENT, Directories.STYLES])
    if st.button("Download content image"):
        download(image_type, content_url, content_url_image_name)

