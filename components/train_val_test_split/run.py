import argparse
import logging
import pandas as pd
import numpy as np
import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args):
    run = wandb.init(job_type="split_data")
    
    logger.info("Reading clean_sample.csv")
    df = pd.read_csv("clean_sample.csv")

    logger.info("Splitting data into trainval and test sets")
    
    # Stratified split using pandas and numpy
    if args.stratify_by != "none" and args.stratify_by in df.columns:
        train_list, test_list = [], []
        for _, group in df.groupby(args.stratify_by):
            shuffled = group.sample(frac=1.0, random_state=args.random_seed)
            n_test = int(len(group) * args.test_size)
            test_list.append(shuffled.iloc[:n_test])
            train_list.append(shuffled.iloc[n_test:])
        test = pd.concat(test_list).sample(frac=1.0, random_state=args.random_seed).reset_index(drop=True)
        trainval = pd.concat(train_list).sample(frac=1.0, random_state=args.random_seed).reset_index(drop=True)
    else:
        shuffled = df.sample(frac=1.0, random_state=args.random_seed).reset_index(drop=True)
        n_test = int(len(df) * args.test_size)
        test = shuffled.iloc[:n_test]
        trainval = shuffled.iloc[n_test:]

    trainval.to_csv("trainval_data.csv", index=False)
    test.to_csv("test_data.csv", index=False)

    logger.info("Uploading trainval_data.csv dataset")
    artifact = wandb.Artifact(
        "trainval_data.csv",
        type="trainval_data",
        description="Train/val dataset"
    )
    artifact.add_file("trainval_data.csv")
    run.log_artifact(artifact)

    logger.info("Uploading test_data.csv dataset")
    artifact = wandb.Artifact(
        "test_data.csv",
        type="test_data",
        description="Test dataset"
    )
    artifact.add_file("test_data.csv")
    run.log_artifact(artifact)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset")
    parser.add_argument("--base_csv", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--test_size", type=float, required=True)
    parser.add_argument("--random_seed", type=int, required=True)
    parser.add_argument("--stratify_by", type=str, required=True)

    args = parser.parse_args()
    go(args)
