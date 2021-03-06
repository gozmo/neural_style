import streamlit as st
from src.config import search_config_template
from src.config import config_template
from src.io import list_style_images
from src.io import list_content_images
from src.io import read_image
from src.io import read_config
from src.main import start_parameter_search
from src.io import list_experiments
from src.main import download_image 
from src.config import config_template
from src.constants import Directories
from os import path
from src.utils import cut_image
from src.utils import resize_image
from datetime import datetime
from collections import OrderedDict

def search():

    config = OrderedDict()
    experiment_name = st.text_input("Experiment name")
    experiment_name = f"{datetime.now()}-{experiment_name}"
    config['experiment_name'] = experiment_name
    config["initial_searches"] = st.number_input("Initial search", 10)

    content_image_name = st.selectbox("Content image", list_content_images())
    content_image_path = path.abspath(f"{Directories.CONTENT}/{content_image_name}")
    content_image = read_image(content_image_path)
    content_image = cut_image(content_image)
    content_image = resize_image(content_image, 500, 500)
    st.image(content_image)
    config['content_image_name'] = content_image_path

    style_image_name = st.selectbox("style image", list_style_images())
    style_image_path = path.abspath(f"{Directories.STYLES}/{style_image_name}")
    style_image = read_image(style_image_path)
    style_image = cut_image(style_image)
    style_image = resize_image(style_image, 500, 500)
    st.image(style_image)
    config['style_image_name'] = style_image_path


    st.markdown("# Static Parameters")

    config['image_width'] = st.number_input("Image width", min_value=100, max_value=1500, value=config_template["image_width"], step=50)


    st.markdown("# Search space")

    search_config = OrderedDict()
    search_config['iterations'] = st.text_input("iterations", search_config_template["iterations"])
    search_config["learning_rate"] = st.text_input("learning_rate", search_config_template["learning_rate"])
    search_config["history_size"] = st.text_input("history_size", search_config_template["history_size"])
    search_config["max_iter"] = st.text_input("max_iter", search_config_template["max_iter"])
    search_config["style_weight"] =st.text_input("style_weight", search_config_template["style_weight"])
    search_config["content_weight"] =st.text_input("content_weight", search_config_template["content_weight"])

    if st.button("Run"):
        start_parameter_search(config, search_config)
