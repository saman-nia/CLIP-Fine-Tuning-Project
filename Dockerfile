FROM python:3.10-slim

# install hatch for building and running the project in container
RUN pip install --no-cache-dir hatch

# working directory
WORKDIR /app

# Copy the project files
COPY pyproject.toml .
COPY src ./src
COPY config.yaml .
COPY README.md .

# we only copy data and models for testing otherwise we mount them in the docker-compose.
# COPY data ./data
# COPY models ./models

# Build the env
RUN hatch env create

# Set the container env as default 
CMD ["hatch", "env", "run", "python", "src/trainer.py"]
