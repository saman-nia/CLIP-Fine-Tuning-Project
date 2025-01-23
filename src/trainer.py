import argparse
import yaml
import os

import mlflow
import torch
import open_clip
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn as nn
from torch.amp import autocast, GradScaler

from src.utils import setup_logger, ensure_dir_exists, generate_version_tag
from src.data_loader import create_dataloaders

def load_config(config_path: str = "config.yaml") -> dict:

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    train_cfg = config.get("train", {})

    # if we enable AWS 
    use_aws = os.environ.get("AWS_MODE", "false").lower() == "true"
    if use_aws:
        # train_cfg["use_s3_for_data"] = True
        # train_cfg["upload_to_s3"] = True
        # train_cfg["s3_data_bucket"] = os.environ.get("S3_DATA_BUCKET", "s3-bucket-for-data")
        # train_cfg["s3_models_bucket"] = os.environ.get("S3_MODELD_BUCKET", "s3-bucket-for-models")
        pass

    config["train"] = train_cfg
    return config

def fetch_data(config):
    """
    if use_s3_for_data is True, then we fetch data from S3.
    """
    train_cfg = config.get("train", {})
    use_s3_for_data = train_cfg.get("use_s3_for_data", False)

    if use_s3_for_data:
        # import boto3
        # s3_client = boto3.client("s3")
        # bucket = train_cfg["s3_data_bucket"]
        # prefix = train_cfg["s3_data_prefix"]
        # local_data_folder = train_cfg["data_folder"]

        # fetch description.json
        # s3_client.download_file(bucket, prefix + "descriptions.json", os.path.join(local_data_folder, "descriptions.json"))
        pass

def upload_model(ckpt_path, config):
    """
    if upload_to_s3 is True
    """
    train_cfg = config.get("train", {})
    upload_to_s3 = train_cfg.get("upload_to_s3", False)

    if upload_to_s3:
        # import boto3
        # s3_client = boto3.client("s3")
        # bucket = train_cfg["s3_models_bucket"]
        # prefix = train_cfg["s3_models_prefix"]

        # remote_path = prefix + os.path.basename(ckpt_path)
        # s3_client.upload_file(ckpt_path, bucket, remote_path)
        pass
    else:
        # local usage and do nothing
        pass

def train_clip_model(config):
    """
    Main fucntion that trains a CLIP-like model
    """

    logger = setup_logger()
    # Get the configs
    train_params = config["train"]

    log_level = train_params.get("monitoring", {}).get("log_level", "INFO")
    logger.setLevel(log_level)

    # mlflow config
    mlflow_enable = train_params.get("monitoring", {}).get("mlflow_enable", False)
    mlflow_tracking_uri = train_params.get("monitoring", {}).get("mlflow_tracking_uri", "http://localhost:5000")
    mlflow_experiment_name = train_params.get("monitoring", {}).get("mlflow_experiment_name", "default")
    mlflow_run_name = train_params.get("monitoring", {}).get("mlflow_run_name", "clip-finetune-run")

    if mlflow_enable:
        logger.info("mlflow is enable...")
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(mlflow_experiment_name)
    else:
        logger.info("mlflow is disabled...")



    data_folder = train_params["data_folder"]
    images_subfolder = train_params["images_subfolder"]
    descriptions_filename = train_params["descriptions_filename"]
    model_arch = train_params["model_arch"]
    pretrained_dataset = train_params["pretrained_dataset"]
    batch_size = train_params["batch_size"]
    num_epochs = train_params["num_epochs"]
    max_lr = train_params["max_learning_rate"]
    output_folder = train_params["model_output_folder"]

    logger.info(f"Starting training with arc={model_arch}, pretrained={pretrained_dataset}, epochs={num_epochs}")

    # fetch data if use S3 
    # fetch_data(config)

    # Device config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Running on device: {device}")

    # Load the model and transformers 
    logger.info("Loading CLIP model and transformers.")
    model, _, preprocess_fn = open_clip.create_model_and_transforms(
        model_arch,
        pretrained_dataset
    )
    tokenizer_fn = open_clip.get_tokenizer(model_arch)

    # Get out the text encoder and only finetune visual part
    model.transformer.eval()
    model = model.to(device)

    # Create DAtaLoders
    logger.info("Preparing data loader.")
    train_loader, val_loader = create_dataloaders(
        data_folder,
        images_subfolder,
        descriptions_filename,
        preprocess_fn,
        tokenizer_fn,
        batch_size,
        split_ratio=0.7
    )

    # Setup optimizer, scheduler and loss
    logger.info("Setting up optimizer and scheduler.")
    optimizer = AdamW(model.visual.parameters())
    loss_fn = nn.CrossEntropyLoss()

    # Use CUDA for GradScaler is GPU avalible
    scaler = GradScaler("cuda", enabled=(device == "cuda"))

    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lr,
        steps_per_epoch=len(train_loader),
        epochs=num_epochs,
        pct_start=0.1
    )

    logger.info("Start training loop.")

    if mlflow_enable:
        with mlflow.start_run(run_name=mlflow_run_name) as run:
            mlflow.log_param("model_arch", train_params["model_arch"])
            mlflow.log_param("batch_size", train_params["batch_size"])
            mlflow.log_param("max_lr", train_params["max_learning_rate"])
            mlflow.log_param("num_epochs", train_params["num_epochs"])

            for epoch in range(num_epochs):
                # mlflow.log_metric("train_loss", loss.item(), step(epoch * len(train_loader) + step))

                # after each epoch:
                mlflow.log_metric("avg_train_loss", avg_train_loss, step=epoch)
                mlflow.log_metric("avg_val_loss", avg_val_loss, step=epoch)

            mlflow.log_artifact(ckpt_path, artifact_path="checkpoints")

    else:
        for epoch in range(num_epochs):
            model.visual.train()
            epoch_losses = []

            for step, (images, text_tokens) in enumerate(train_loader, start=1):
                images = images.to(device)
                text_tokens = text_tokens.squeeze(1).to(device)

                # if training is on GPU, pass CUDA as the first argument
                with autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                    img_feats = model.encode_image(images)
                    text_feats = model.encode_text(text_tokens)

                    # normlize
                    img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
                    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

                    # calculate similarity 
                    similarity = 100.0 * img_feats @ text_feats.T
                    loss = loss_fn(similarity, torch.arange(len(images), device=device))

                # scale and step
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                epoch_losses.append(loss.item())

                # Log every step
                logger.info(f"Epoch {epoch+1} / {num_epochs}, Step {step}, Loss: {loss.item():.4f}")

            # eval at the end of epoch
            model.visual.eval()
            val_losses = []
            with torch.no_grad():
                for images, text_tokens in val_loader:
                    images = images.to(device)
                    text_tokens = text_tokens.squeeze(1).to(device)

                    with autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                        vf = model.encode_image(images)
                        tf = model.encode_text(text_tokens)
                        vf = vf / vf.norm(dim=-1, keepdim=True)
                        tf = tf / tf.norm(dim=-1, keepdim=True)
                        val_similarity = 100.0 * vf @tf.T
                        val_loss = loss_fn(val_similarity, torch.arange(len(images), device=device))

                    val_losses.append(val_loss.item())

            avg_train_loss = sum(epoch_losses) / len(epoch_losses)
            avg_val_loss = sum(val_losses) / len(val_losses)
            logger.info(f"Epoch {epoch+1} done. Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")

    # model versioning and saving
    ensure_dir_exists(output_folder)
    version_tag = generate_version_tag()
    ckpt_path = os.path.join(output_folder, f"clip_visual_{version_tag}.chkpt")

    logger.info(f"Saving checkpoint to {ckpt_path}")
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "config": config,
    }
    torch.save(checkpoint, ckpt_path)

    # if we uload to S3
    # upload_model(ckpt_path, config)

    logger.info("All Done")
    return ckpt_path


def main():
    parser = argparse.ArgumentParser(description="Finetune CLIP on new image + text.")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
    train_clip_model(config)


if __name__ == "__main__":
    main()