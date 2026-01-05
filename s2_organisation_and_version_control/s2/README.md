# s2_exercises

exercises day 2. organization and version control

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

## Instructions

The scripts can be run from the command line. For example, to train a model, run, below is also the default arguments:

```bash
python src/s2_exercises/train.py --lr 0.001 --batch-size 32 --epochs 10
```

or with uv using the above default arguments: 
```bash
uv run src/s2_exercises/train.py
```
This will train and save a model to `s2_organisation_and_version_control/s2/models/model.pth`.

To evaluate the model, run:

```bash
uv run src/s2_exercises/evaluate.py --model-checkpoint s2_organisation_and_version_control/s2/models/model.pth
```
This will print the accuracy of the model on the test set. and to visualize the model embeddings, run:

```bash
uv run src/s2_exercises/visualize.py --model-checkpoint s2_organisation_and_version_control/s2/models/model.pth
```

Linting code:

```bash
uv run ruff check .  # Lint all files in the current directory (and any subdirectories)
uv run ruff check path/to/code/  # Lint all files in `/path/to/code` (and any subdirectories).
```

To check if typing is done correctly on a file. use:

```bash
uv run mypy typing_exercise.py
```
