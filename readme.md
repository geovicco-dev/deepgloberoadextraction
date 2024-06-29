# Deep Globe Road Extraction using Convolutional Neural Networks

1. Clone Repository using GitHub CLI ot Git
    ```{bash}
    git clone https://github.com/geovicco-dev/deepgloberoadextraction.git && cd deepgloberoadextraction
    ```
    or 
    ```{bash}
    gh repo clone geovicco-dev/deepgloberoadextraction && cd deepgloberoadextraction
    ```
2. Install `uv` package manager using pip, setup virtual environment
    ```{bash}
    pip install uv && uv venv --python=3.10 .venv && source ./.venv/bin/activate
    ```
3. Install dependencies and install project as a local package
    ```{bash}
    uv pip install -r requirements.txt && uv pip install -e .
    ```
4. Pull data from DVC remote
    ```{bash}
    dvc pull -r dvc-remote
    ```