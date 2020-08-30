import streamlit as st
from src.pages.train import train
from src.pages.results import results 
from src.pages.hpo import hpo
from src.pages.search import search
from src.pages.download_image import download_image
from src.pages.search_results import search_results

page = st.sidebar.radio("Page", ("Train", "Results", "Parameter Search", "Parameter Search Results", "Download image"))

if page == "Train":
    train()
elif page == "Results":
    results()
elif page == "Parameter Search":
    search()
elif page == "Parameter Search Results":
    search_results()
elif page == "Download image":
    download_image()

