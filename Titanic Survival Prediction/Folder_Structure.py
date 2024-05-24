import os

# Base directory
base_dir = r"Your_path_here"

# Directory structure
directories = [".assets", "configs", "pipelines", "steps", "utils","data"]

# YAML files in configs
config_files = [
    "feature_engineering.yaml",
    "inference.yaml",
    "training_rf.yaml",
    "training_sgd.yaml",
]

# Python files in pipelines
pipeline_files = {
    "__init__.py": """from .feature_engineering import feature_engineering
from .inference import inference
from .training import training
""",
    "feature_engineering.py": "",
    "inference.py": "",
    "training.py": "",
}

# Python files in steps
steps_files = {
    "__init__.py": """from .data_loader import (
    data_loader,
)
from .data_preprocessor import (
    data_preprocessor,
)
from .data_splitter import (
    data_splitter,
)
from .inference_predict import (
    inference_predict,
)
from .inference_preprocessor import (
    inference_preprocessor,
)
from .model_evaluator import (
    model_evaluator,
)
from .model_promoter import (
    model_promoter,
)
from .model_trainer import (
    model_trainer,
)
""",
    "data_loader.py": "",
    "data_preprocessor.py": "",
    "data_splitter.py": "",
    "inference_predict.py": "",
    "inference_preprocessor.py": "",
    "model_evaluator.py": "",
    "model_promoter.py": "",
    "model_trainer.py": "",
}

# Python files in utils
utils_files = {
    "__init__.py": """from .preprocess import NADropper, ColumnsDropper, DataFrameCaster
""",
    "preprocess.py": "",
}

# Create directories
for directory in directories:
    os.makedirs(os.path.join(base_dir, directory), exist_ok=True)

# Create config YAML files
for file in config_files:
    open(os.path.join(base_dir, "configs", file), "a").close()

# Create pipeline files with content
for file, content in pipeline_files.items():
    with open(os.path.join(base_dir, "pipelines", file), "w") as f:
        f.write(content)

# Create steps files with content
for file, content in steps_files.items():
    with open(os.path.join(base_dir, "steps", file), "w") as f:
        f.write(content)

# Create utils files with content
for file, content in utils_files.items():
    with open(os.path.join(base_dir, "utils", file), "w") as f:
        f.write(content)

# Create the run.py file
open(os.path.join(base_dir, "run.py"), "a").close()

print("Folder structure and files created successfully.")
