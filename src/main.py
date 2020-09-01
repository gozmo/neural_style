from tqdm import tqdm
from PIL import Image
from skopt import gp_minimize
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
from skopt.space import Categorical


#TODO: merge parameter search and training, setup is similar
def start_training(config):
    experiment_root = f"{Directories.EXPERIMENTS}/{config['experiment_name']}"
    config["experiment_root"] = experiment_root
    os.makedirs(f"{experiment_root}/checkpoints", exist_ok=True)
    with open(f"{experiment_root}/config.json", "w") as f:
        f.write(json.dumps(config))
    with open(f"{experiment_root}/user.json", "w") as f:
        f.write(json.dumps({}))

    style.run(config)



def __start_search_experiment(idx, config_setup, search_config):
    experiment_root = f"{Directories.SEARCHES}/{search_config['experiment_name']}/experiments/{idx}" 
    checkpoint_dir = f"{experiment_root}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    with open(f"{experiment_root}/config.json", "w") as f:
        f.write(json.dumps(config_setup))
    with open(f"{experiment_root}/user.json", "w") as f:
        f.write(json.dumps({"rating": "unrated"}))

    config_setup.update(search_config)

    config_setup["experiment_root"] = experiment_root
    

    output_file = f"{experiment_root}/output.jpg"
    config_setup["output_image_name"] = output_file
    config_setup["checkpoint_path"] = f"{checkpoint_dir}/"

    style.run(config_setup)

def start_parameter_search(search_config, search_space):
    search_root = f"{Directories.SEARCHES}/{search_config['experiment_name']}"
    os.makedirs(search_root)
    with open(f"{search_root}/search_config.json", "w") as f:
        f.write(json.dumps(search_config))
    with open(f"{search_root}/search_space.json", "w") as f:
        f.write(json.dumps(search_space))
    parsed_search_space = {}
    for parameter_name, parameter_space_string in search_space.items():
        parsed_search_space[parameter_name] = [parse(x) for x in parameter_space_string.split(",")]


    pg = ParameterGrid(parsed_search_space)
    for idx, config_setup in tqdm(list(enumerate(pg))):
        __start_search_experiment(idx, config_setup, search_config)


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
    
def read_user(experiment_root):
    path = f"{experiment_root}/user.json"
    with open(path, "r") as f:
        user_dict = json.load(f)

    return user_dict


def set_rating(experiment_root, rating):
    user = read_user(experiment_root)
    user["rating"] = rating
    path = f"{experiment_root}/user.json"

    with open(path, "w") as f:
        f.write(json.dumps(user))

def read_search_config(search_name):
    search_root = f"{Directories.SEARCHES}/{search_name}"
    with open(f"{search_root}/search_config.json", "r") as f:
        config = json.load(f)
    with open(f"{search_root}/search_space.json", "r") as f:
        search_space = json.load(f)
    return config, search_space

def annotated_search(search_name):
    search_config, search_space = read_search_config(search_name)
    parameter_runs = list_parameter_runs(search_name)
    prior_x = []
    prior_y = []
    for parameter_run in parameter_runs:
        run_config  = read_parameter_config(search_name, parameter_run)
        prior_x.append(list(run_config.values()))
        experiment_root = f"{Directories.SEARCHES}/{search_name}/experiments/{parameter_run}" 
        run_user = read_user(experiment_root)
        prior_y.append( 1.0 if run_user["rating"] == "good" else 0.0)

    idx = len(parameter_runs)
    def f(config_values):
        config = dict(zip(search_space.keys(), config_values))
        __start_search_experiment(idx, config, search_config)

        idx += 1
        return 1.0
    
    search_bounds = []
    for bound_str in search_space.values():
        a = [parse(elem) for elem in bound_str.split(",")]
        bound = Categorical(a)
        search_bounds.append(bound)

    res = gp_minimize(f,                  # the function to minimize
		      search_bounds,      # the bounds on each dimension of x
		      acq_func="EI",      # the acquisition function
		      n_calls=10,         # the number of evaluations of f
		      n_random_starts=10,  # the number of random initialization points
		      noise=0.1**2,       # the noise level (optional)
		      random_state=1234,
		      x0=prior_x,
		      y0=prior_y)   # the random seed


