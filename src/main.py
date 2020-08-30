from tqdm import tqdm
from PIL import Image
import os
from src.constants import Directories
import json
import urllib.request
from src.command_builder import CommandBuilder
import subprocess
import json
from src.config import config_template
from sklearn.model_selection import ParameterGrid
from src import style
from src.utils import parse


def start_training(config):
    os.makedirs(f"{Directories.EXPERIMENTS}/{config['experiment_name']}/checkpoints", exist_ok=True)
    with open(f"{Directories.EXPERIMENTS}/{config['experiment_name']}/config.json", "w") as f:
        f.write(json.dumps(config))

    style.run(config)



def start_parameter_search(config, search_space):
    parsed_search_space = {}
    for parameter_name, parameter_space_string in search_space.items():
        parsed_search_space[parameter_name] = [parse(x) for x in parameter_space_string.split(",")]


    pg = ParameterGrid(parsed_search_space)
    for idx, config_setup in tqdm(list(enumerate(pg))):
        output_dir = f"{Directories.SEARCHES}/{config['experiment_name']}/experiments/{idx}/" 
        checkpoint_dir = f"{output_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        output_file = f"{output_dir}/output.jpg"
        os.makedirs(output_dir, exist_ok=True)
        config_setup.update(config)
        config_setup["output_image_name"] = output_file
        config_setup["checkpoint_path"] = f"{checkpoint_dir}/cp_%d.jpg"
        with open(f"{output_dir}/config.json", "w") as f:
            f.write(json.dumps(config_setup))

        style.run(config_setup)


def read_image(image_path):
    image = Image.open(image_path)
    return image

def download_image(image_type, image_url, image_name):

    with urllib.request.urlopen(image_url) as response, \
         open(f"{image_type}/{image_name}", 'wb') as out_file:
        data = response.read()
        out_file.write(data)

def list_content_images():
    return sorted(os.listdir(Directories.CONTENT))

def list_style_images():
    return sorted(os.listdir(Directories.STYLES))

def list_experiments():
    return sorted(os.listdir(Directories.EXPERIMENTS), reverse=True)

def list_checkpoint_images(experiment_name):
    path = f"{Directories.EXPERIMENTS}/{experiment_name}/checkpoints/"
    files = [os.path.abspath(f"{path}/{filename}") for filename in os.listdir(path)]
    return sorted(files, reverse=True)

def list_search_runs():
    return sorted(os.listdir(Directories.SEARCHES), reverse=True)

def list_parameter_runs(search_run):
    path = f"{Directories.SEARCHES}/{search_run}/{Directories.EXPERIMENTS}"
    dirs = [directory for directory in os.listdir(path)]
    return sorted(dirs)

def read_parameter_config(search_run, parameter_run):
    path = f"{Directories.SEARCHES}/{search_run}/{Directories.EXPERIMENTS}/{parameter_run}/config.json"
    with open(path, "r") as f:
        config_dict = json.load(f)

    return config_dict

def list_search_checkpoints(search_run, parameter_run):
    path = f"{Directories.SEARCHES}/{search_run}/{Directories.EXPERIMENTS}/{parameter_run}/checkpoints/"
    files = [os.path.abspath(f"{path}/{filename}") for filename in os.listdir(path)]
    return sorted(files, reverse=True)

    

def read_config(experiment_name):
    path = f"{Directories.EXPERIMENTS}/{experiment_name}/config.json"
    with open(path, "r") as f:
        config_dict = json.load(f)

    return config_dict
    
