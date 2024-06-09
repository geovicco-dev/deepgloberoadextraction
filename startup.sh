# Setup Project Folder
python project_template.py
# Install `uv` and Create Virtual Environment
pip install uv && uv venv --python=3.10 .venv
# Activate Virtual Environment
source .venv/bin/activate
# Install Dependencies
uv pip install -r requirements.txt
# Install Project as a Package
uv pip install -e .
