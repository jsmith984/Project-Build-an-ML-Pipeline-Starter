import wandb

api = wandb.Api()

ENTITY = "jsm394-western-governors-university"
PROJECT = "nyc_airbnb"

for alias in ("latest", "reference"):
    name = f"{ENTITY}/{PROJECT}/sample.csv:{alias}"

    try:
        artifact = api.artifact(name)

        print(f"\nFOUND: {name}")
        print("Version:", artifact.version)
        print("Type:", artifact.type)
        print("Aliases:", artifact.aliases)

    except Exception as exc:
        print(f"\nFAILED: {name}")
        print(f"{type(exc).__name__}: {exc}")