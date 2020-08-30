import streamlit as st
from src.main import list_experiments
from src.main import read_config
from src.main import read_image
from src.main import list_checkpoint_images
import pandas as pd

def results():
    experiments = list_experiments()

    experiment_name = st.selectbox("Experiments", experiments)
    config = read_config(experiment_name)

    final_image_path = config["output_image_name"]
    final_image = read_image(final_image_path)
    st.image(final_image)

    st.markdown("Config")
    config_frame = pd.DataFrame.from_dict(config, orient="index")
    st.table(config_frame)

    st.markdown("Style image")
    style_image_path = config["style_image_name"]
    style_image = read_image(style_image_path)
    st.image(style_image)


    st.markdown("# Checkpoints")
    checkpoint_images = list_checkpoint_images(experiment_name)
    for checkpoint_image_path in checkpoint_images:
        checkpoint_image = read_image(checkpoint_image_path)
        st.image(checkpoint_image)
