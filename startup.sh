# Setup Project Folder
echo "Setting up project folder..."
python project_template.py  # Run Python script to set up project folder structure

# Install `uv` and Create Virtual Environment
echo "Installing uv package and creating virtual environment..."
pip install uv && uv venv --python=3.10 .venv  # Install uv package and create virtual environment

# Activate Virtual Environment
echo "Activating virtual environment..."
source .venv/bin/activate  # Activate virtual environment

# Install Dependencies
echo "Installing project dependencies..."
uv pip install -r requirements.txt  # Install dependencies listed in requirements.txt

# Install Project as a Package
echo "Installing project as a package..."
uv pip install -e .  # Install project as a package in editable mode
