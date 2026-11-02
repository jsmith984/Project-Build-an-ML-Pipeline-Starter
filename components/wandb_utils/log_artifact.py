import os
import wandb


def log_artifact(
    artifact_name,
    artifact_type,
    artifact_description,
    filename,
    wandb_run,
):
    """
    Register a local file as a W&B artifact.
    """

    if not os.path.isfile(filename):
        raise FileNotFoundError(
            f"Artifact file not found: {filename}"
        )

    artifact = wandb.Artifact(
        name=artifact_name,
        type=artifact_type,
        description=artifact_description,
    )

    artifact.add_file(filename)

    logged_artifact = wandb_run.log_artifact(
        artifact,
        aliases=["latest"],
    )

    # Wait for artifact registration.
    logged_artifact.wait()

    return logged_artifact
