from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = 'DeepGlobeRoadExtraction'

list_of_files = [
    "data",
    "models",
    "results",
    "results/plots",
    "docs/stages.md",
    "logs",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/configuration.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    "main.py",
    "config.yaml",
    "dvc.yaml",
    "params.yaml",
    "secrets.yaml",
    "requirements.txt",
    "setup.py",
    "notebooks/experiments.ipynb",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    if not filepath.suffix:  # If the path does not have a file extension, it's a directory
        if not filepath.exists():
            filepath.absolute().mkdir(parents=True, exist_ok=True)
            logging.info(f"Creating directory: {filepath}")
        else:
            logging.info(f"Directory {filepath} already exists")

    elif filepath.suffix:  # If the path has a file extension, it's a file
        if not filepath.exists():
            # Make sure the parent directory exists
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.touch()
            logging.info(f"Creating file: {filepath}")
        else:
            logging.info(f"File {filepath} already exists")
    else:
        logging.error(f"Unknown file type: {filepath}")