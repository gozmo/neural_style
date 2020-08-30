import streamlit as st
from src import main
import pudb
import pandas as pd

def search_results():

    search_runs = main.list_search_runs()
    search_run = st.selectbox("Search runs", search_runs)

    parameter_runs = main.list_parameter_runs(search_run)
    
    for parameter_run in parameter_runs:
        config = main.read_parameter_config(search_run, parameter_run)

        output_image = main.read_image(config["output_image_name"])
        st.image(output_image)
        if st.checkbox("Show Config", False, key=config["output_image_name"]):
            config_frame = pd.DataFrame.from_dict(config, orient="index")
            st.table(config_frame)


