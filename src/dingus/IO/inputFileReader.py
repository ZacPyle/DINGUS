# src/dingus/IO/inputFileReader.py

import yaml

def read_input_file_YAML(file_path) -> dict:
    with open(file_path, 'r') as file:
        # Load the YAML content into a Python dictionary using the built-in YAML loader
        input_data = yaml.safe_load(file)

    return input_data