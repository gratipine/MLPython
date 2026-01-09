# MLPython

## Overview
This repo will be used to hold general explorations of machine learning using python.

## Running the code

### Conda environment
The conda environment to run this in can be generated using the environment file at the 
root of the repo. If using the environment file please bear in mind that there are
certain differences in what packages and package versions might be available depending 
on what sort of OS you have.

If you are unfamiliar with conda, [this|https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html]
is a great reading start.

### Running the notebooks
To run a jupyter notebook you have to:
- open a prefered console window - I like the anaconda one - and navigate to a folder somewhere above your repo
- activate your preferred environment (the one in this repo or another one that contains the necessary packages)
- run either **jupyter notebook** or **jupyter lab**


```bash
pip install virtualenv
python -m venv .venv
.venv\Scripts\activate
pip install -U setuptools
pip install -r requirements.txt 
```

