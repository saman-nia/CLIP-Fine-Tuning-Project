# CLIP Fine-Tuning Project

I created this project to fine-tune a CLIP-like model on a custom image-text dataset. The goal is to run it monthly to keep up with new data. The project uses **Hatch** for environment management and **Docker Compose** for containerization.

## My solution includes:

- Config-driven training with config.yaml
- Docker + docker-compose setup
- Logging and optional MLflow monitoring
- Time-stamped model checkpoints for versioning
- S3 placeholders for remote data and checkpoint storage
- Optional notification stubs for email or Slack


## Project Structure

```
AI_ML_Engineer

├── data/                   # Dataset folder
│   ├── images/             # Your training images
│   └── descriptions.json   # Text descriptions for each image
├── models/                 # Saved model checkpoints
├── src/                    # Source code
│   ├── data_loader.py      # Data loading and Dataset logic
│   ├── trainer.py          # Main training script
│   ├── utils.py            # Logger, helper functions
│   └── __init__.py         # Marks `src` as a package
├── config.yaml             # Configuration for training parameters
├── pyproject.toml          # Defines dependencies for Hatch
├── Dockerfile              # Docker build instructions
├── docker-compose.yml      # Docker Compose file to build and run the container
└── README.md               # Documentation for the project
```


## Dependencies

The project requires the following tools and libraries:
- Python 3.10 or higher
- Hatch (for environment and dependency management)
- Docker and Docker Compose (optional, for running in containers)

Python dependencies are listed in the `pyproject.toml` file and are automatically handled by Hatch or Docker.


## Setting Up the Environment

### 1. Install Hatch
If you don’t already have Hatch installed, you can install it using `pip`:

```bash
pip install --upgrade pip
pip install hatch
```

### 2. Create the environment using the pyproject.toml in this folder

```bash
hatch env create
```
This will download and install all required dependencies (like PyTorch, open_clip_torch, etc.) into a local virtual environment that Hatch manages.


## Using Docker & Docker Compose

If you prefer using Docker:
1.  **Install Docker** (and Docker Compose v2).
2.  **Build and run** the container:

How to build the Docker image locally:
```bash
docker build . -t clip_finetune
```
How to run the container:
```bash
docker-compose up --build
```

*   This picks up our `Dockerfile`, builds the image, and starts the container.
*   By default, it mounts the `./data` and `./models` folders so your data is accessible and the final model checkpoint is saved to your local filesystem.


## Monthly Schedule

Each month, place the new images in `data/images/` (or set up S3 if needed), then run the above command (locally or via CI/CD). The pipeline saves a new model with a unique timestamp in `models/`.


## Logging & Monitoring

- Logs go to the console by default.
- If you enable `mlflow_enable` in `config.yaml`, you can track loss metrics in an MLflow server like `http://localhost:5000`.
- You can view metrics in the MLflow UI for better monitoring.


## Model Versioning and S3

- Every model is saved with a timestamp like `clip_visual_20250123_035711.chkpt`.
- If you set `upload_to_s3: true` (in real usage), checkpoints can be stored in S3 for sharing or deployment.


## Notifications (Optional)

Right now, there are placeholders in `config.yaml` for:

- Email or Slack integration
- If you enable them, you could send a message on success/failure
- You’d just add a short script or API call in the code


## Potential Use of LLMs
I did not apply LLMs in this project, but they could be helpful if we expand it later. For example, we might use them to:
- **Suggest best practices** for MLOps workflows, logging, and error handling.
