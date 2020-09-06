import streamlit as st
import pudb
import pandas as pd
from src.constants import Directories
from src.search_io import read_parameter_config
from src.io import read_image
from src.search_io import read_user
from src.search_io import set_rating
from src.search_io import list_search_runs
from src.search_io import list_parameter_runs
from src.main import annotated_search
import pandas as pd
from collections import defaultdict


def __show_rated_images(parameter_runs, search_run, rating):
    for parameter_run in parameter_runs:
        config = read_parameter_config(search_run, parameter_run)

        experiment_root = f"{Directories.SEARCHES}/{search_run}/experiments/{parameter_run}" 
        output_image = read_image(f"{experiment_root}/output.jpg")
        user = read_user(experiment_root)


        color_distribution = output_image.getcolors()
        if color_distribution != None and len(color_distribution) == 1:
            set_rating(experiment_root, "bad")
            continue


        if user["rating"] == rating:
            st.image(output_image)
            options = ["unrated", "good", "bad"]
            preselect = options.index(user["rating"])
            new_rating = st.radio("Good or Bad", options, index=preselect, key=f"radio_{experiment_root}")
            if new_rating != rating:
                set_rating(experiment_root, new_rating)
            if st.checkbox("Show Config", False, key=experiment_root):
                config_frame = pd.DataFrame.from_dict(config, orient="index")
                st.table(config_frame)

def __annotate_all_unrated_as_bad(parameter_runs, search_run):
    if st.button(f"Annotate the rest as bad"):
        for parameter_run in parameter_runs:
            config = read_parameter_config(search_run, parameter_run)
            experiment_root = f"{Directories.SEARCHES}/{search_run}/experiments/{parameter_run}" 
            user = read_user(experiment_root)

            if user["rating"] == "unrated":
                set_rating(experiment_root, "bad")

def __plot_parameters(parameter_runs, search_run):
    book_keeping = defaultdict(list)
    for parameter_run in parameter_runs:
        config = read_parameter_config(search_run, parameter_run)
        experiment_root = f"{Directories.SEARCHES}/{search_run}/experiments/{parameter_run}" 
        user = read_user(experiment_root)

        if user["rating"] == "good":
            for key, value in config.items():
                book_keeping[key].append(value)

    df = pd.DataFrame(book_keeping)
    df = df.transpose()
    st.dataframe(df)




def search_results():
    search_runs = list_search_runs()
    search_run = st.selectbox("Search runs", search_runs)

    parameter_runs = list_parameter_runs(search_run)

    if st.checkbox("Show unrated"):
        __show_rated_images(parameter_runs, search_run, "unrated")
        __annotate_all_unrated_as_bad(parameter_runs, search_run)
    if st.checkbox("Show good"):
        __show_rated_images(parameter_runs, search_run, "good")
    if st.checkbox("Show bad"):
        __show_rated_images(parameter_runs, search_run, "bad")

    optimize_search = st.slider("Optimize runs", 0, 150, value = 5)
    random = st.slider("Random runs", 0, 150, value = 0)
    if st.button("ReRun"):
        annotated_search(search_run, optimize_search, random)

    if st.checkbox("Plot parameters"):
        __plot_parameters(parameter_runs, search_run)



