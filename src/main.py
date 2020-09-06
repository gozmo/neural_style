from sklearn.model_selection import ParameterGrid
from skopt import gp_minimize
from skopt.space import Categorical
from skopt.utils import use_named_args 
from src import style
from src.config import config_template
from src.constants import Directories
from src.io import read_image
from src.utils import convert_to_serializable
from src.utils import parse
from src.utils import parse_search_string 
from src import search_io 
from tqdm import tqdm
import json
import numpy as np
import os
import subprocess
import urllib.request


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
    os.makedirs(search_root + "/experiments")
    with open(f"{search_root}/search_config.json", "w") as f:
        f.write(json.dumps(search_config))
    with open(f"{search_root}/search_space.json", "w") as f:
        f.write(json.dumps(search_space))

    key_order = list(search_space.keys())
    search_bounds = []
    for parameter_name in key_order:
        parameter_space_str = search_space[parameter_name]
        bound = parse_search_string(parameter_space_str, parameter_name)
        search_bounds.append(bound)

   
    n_calls = search_config["initial_searches"]
    progress_bar = tqdm(total=n_calls)
    @use_named_args(dimensions=search_bounds)
    def f(content_weight, iterations, learning_rate, style_weight):
        config = {"iterations": int(iterations),
                  "learning_rate": float(learning_rate),
                  "style_weight": int(style_weight),
                  "content_weight": int(content_weight)}

        idx = search_io.get_experiment_index(search_config)
        __start_search_experiment(idx, config, search_config)
        progress_bar.update()

        return 1.0
    
    res = gp_minimize(f,                  # the function to minimize
		      search_bounds,      # the bounds on each dimension of x
		      n_calls=n_calls,         # the number of evaluations of f
                      n_random_starts=n_calls,
                      n_initial_points=n_calls)


def download_image(image_type, image_url, image_name):

    with urllib.request.urlopen(image_url) as response, \
         open(f"{image_type}/{image_name}", 'wb') as out_file:
        data = response.read()
        out_file.write(data)

def annotated_search(search_name, optimize_runs, random):
    search_config, search_space = search_io.read_search_config(search_name)
    key_order = list(search_space.keys())
    parameter_runs = search_io.list_parameter_runs(search_name)
    prior_x = []
    prior_y = []
    for parameter_run in parameter_runs:
        run_config  = search_io.read_parameter_config(search_name, parameter_run)
        prior_x.append([run_config[key] for key in key_order])
        experiment_root = f"{Directories.SEARCHES}/{search_name}/experiments/{parameter_run}" 
        run_user = search_io.read_user(experiment_root)
        prior_y.append( 0.0 if run_user["rating"] == "good" else 1.0)


    search_bounds = []
    for parameter_name in key_order:
        parameter_space_str = search_space[parameter_name]
        bound = parse_search_string(parameter_space_str, parameter_name)
        search_bounds.append(bound)

    calls = optimize_runs + random
    progress_bar = tqdm(total=calls)
    @use_named_args(dimensions=search_bounds)
    def f(content_weight, iterations, learning_rate, style_weight):
        config = {"iterations": int(iterations),
                  "learning_rate": float(learning_rate),
                  "style_weight": int(style_weight),
                  "content_weight": int(content_weight)}

        idx = search_io.get_experiment_index(search_config)
        __start_search_experiment(idx, config, search_config)
        progress_bar.update()

        return 0.5
    

    res = gp_minimize(f,                  # the function to minimize
		      search_bounds,      # the bounds on each dimension of x
		      n_calls=calls,         # the number of evaluations of f
                      n_random_starts=random,
                      n_initial_points=calls,
		      x0=prior_x,
		      y0=prior_y)   # the random seed

    f(res.x)


