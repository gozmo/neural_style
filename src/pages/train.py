import streamlit as st
from src.main import list_style_images
from src.main import list_content_images
from src.main import read_image
from src.main import read_config
from src.main import start_training
from src.main import list_experiments
from src.main import download_image 
from src.config import config_template as ct
from src.constants import Directories
from os import path
from datetime import datetime
from src.utils import cut_image
from src.utils import resize_image


def train():

    experiment_name = st.text_input("Experiment name")
    experiment_name = f"{datetime.now()}-{experiment_name}"


    if st.checkbox("Download images"):
        content_url = st.text_input("Content url")
        content_url_image_name = st.text_input("image name")
        if st.button("Download content image"):
            download_image(Directories.CONTENT, content_url, content_url_image_name)

        style_url = st.text_input("style url")
        style_url_image_name = st.text_input("image name", key="style")
        if st.button("Download style image"):
            download_image(Directories.STYLES, style_url, style_url_image_name)

    content_image_name = st.selectbox("Content image", list_content_images())
    content_image_path = path.abspath(f"{Directories.CONTENT}/{content_image_name}")
    content_image = read_image(content_image_path)
    content_image = cut_image(content_image)
    content_image = resize_image(content_image)
    st.image(content_image)


    
    style_image_name = st.selectbox("style image", list_style_images())
    style_image_path = path.abspath(f"{Directories.STYLES}/{style_image_name}")
    style_image = read_image(style_image_path)
    style_image = cut_image(style_image)
    style_image = resize_image(style_image)
    st.image(style_image)

    output_image_name = path.abspath(f"{Directories.EXPERIMENTS}/{experiment_name}/{experiment_name}.jpg")

    st.markdown("Config template")

    config_name = st.selectbox("Config", ["default"] + list_experiments())
    if config_name != "default":
        config_template = read_config(config_name)
    else:
        config_template = ct

    st.markdown("# Parameters")

    config = {}
    config["experiment_name"] = experiment_name
    config["content_image_name"] = content_image_path
    config["style_image_name"] = style_image_path
    config["output_image_name"] = output_image_name

    if config_template["output_image_name"] != None:
        output_image = read_image(config_template["output_image_path"])
        st.image(output_image)

    config["iterations"] = st.number_input("Iterations", min_value=10, max_value=10000, value=config_template["iterations"], step=50)
    config["image_width"] = st.number_input("Image width", min_value=100, max_value=5000, value=config_template["image_width"], step=50)
    config["learning_rate"] = st.number_input("Learning rate", min_value=0.1, max_value=100.0, value=config_template["learning_rate"], step=0.1)
    config["style_weight"] = st.number_input("Style weight", min_value=0, value=config_template["style_weight"])
    config["content_weight"] = st.number_input("Content weight", min_value=0, value=config_template["content_weight"])

    if st.button("Run"):
        start_training(config)
