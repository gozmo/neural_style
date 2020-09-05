import streamlit as st
from src import main
import pudb
import pandas as pd
from src.constants import Directories


def __show_rated_images(parameter_runs, search_run, rating):
    for parameter_run in parameter_runs:
        config = main.read_parameter_config(search_run, parameter_run)

        experiment_root = f"{Directories.SEARCHES}/{search_run}/experiments/{parameter_run}" 
        output_image = main.read_image(f"{experiment_root}/output.jpg")
        user = main.read_user(experiment_root)

        if user["rating"] == rating:
            st.image(output_image)
            options = ["unrated", "good", "bad"]
            preselect = options.index(user["rating"])
            new_rating = st.radio("Good or Bad", options, index=preselect, key=f"radio_{experiment_root}")
            if new_rating != rating:
                main.set_rating(experiment_root, new_rating)
            if st.checkbox("Show Config", False, key=experiment_root):
                config_frame = pd.DataFrame.from_dict(config, orient="index")
                st.table(config_frame)



def search_results():
    search_runs = main.list_search_runs()
    search_run = st.selectbox("Search runs", search_runs)

    parameter_runs = main.list_parameter_runs(search_run)

    if st.checkbox("Show unrated"):
        __show_rated_images(parameter_runs, search_run, "unrated")
    if st.checkbox("Show good"):
        __show_rated_images(parameter_runs, search_run, "good")
    if st.checkbox("Show bad"):
        __show_rated_images(parameter_runs, search_run, "bad")

    optimize_search = st.slider("Optimize runs", 0, 20, value = 5)
    random = st.slider("Random runs", 0, 20, value = 0)
    if st.button("ReRun"):
        main.annotated_search(search_run, optimize_search, random)



