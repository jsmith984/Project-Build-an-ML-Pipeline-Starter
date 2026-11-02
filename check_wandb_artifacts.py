import sys
import wandb

api = wandb.Api()

ENTITY = "jsm394-western-governors-university"
PROJECT = "nyc_airbnb"

artifacts_to_check = [
    ("sample.csv", "raw_data"),
    ("clean_data.csv", "clean_data"),
    ("trainval_data.csv", "trainval_data"),
    ("test_data.csv", "test_data"),
    ("model_export", "model"),
    ("test_predictions", "evaluation"),
]

print("\nNYC AIRBNB W&B ARTIFACT VERIFICATION")
print("=" * 55)

failed = []

for artifact_name, expected_type in artifacts_to_check:
    name = f"{ENTITY}/{PROJECT}/{artifact_name}:latest"

    try:
        artifact = api.artifact(name)

        print(f"\nFOUND: {artifact.name}")
        print(f"Version: {artifact.version}")
        print(f"Type: {artifact.type}")
        print(f"Aliases: {artifact.aliases}")

        if artifact.type != expected_type:
            print(f"WARNING: Expected type {expected_type}")
            failed.append(artifact_name)

    except Exception as exc:
        print(f"\nNOT FOUND: {artifact_name}")
        print(f"Details: {exc}")
        failed.append(artifact_name)

print("\n" + "=" * 55)

if failed:
    print("VERIFICATION FAILED")
    print("Artifacts requiring attention:", failed)
    sys.exit(1)

print("SUCCESS: All six W&B artifacts verified.")