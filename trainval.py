import pandas as pd
import argparse
import os, torch

import datasets, models

from haven import haven_wizard as hw
from haven import haven_utils as hu


def trainval(exp_dict, savedir, args):
    """
    exp_dict: dictionary defining the hyperparameters of the experiment
    savedir: the directory where the experiment will be saved
    args: arguments passed through the command line
    """
    # Get Pytorch datasets
    train_set = datasets.get_dataset(
        name=exp_dict["dataset"],
        split="train"
    )
    val_set = datasets.get_dataset(
        name=exp_dict["dataset"],
        split="val"
    )

    # Tip: Test dataset to avoid bugs later (Tip: visualize if possible in tmp folder)
    sample = train_set[0]
    # hu.save_image('.tmp/tmp.png', sample[0].reshape((8,8)))

    # Create data loader (Tip: increase number of threads & drop last batch)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=256, drop_last=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=256, num_workers=0)

    # Get Model
    model = models.get_model(exp_dict=exp_dict, device=args.device)
    
    # Train and Validate
    score_list = []
    for epoch in range(exp_dict['epochs']):
        # Train for one epoch
        train_dict = model.train_on_loader(train_loader)

        # Validate
        val_dict = model.val_on_loader(val_loader)

        # Get Metrics (Tip: add time and size)
        score_dict = {
            "epoch": epoch,
            "train_acc": train_dict["train_acc"],
            "train_loss": train_dict["train_loss"],
            "val_acc": val_dict["val_acc"],
        }

        # Save Metrics
        score_list += [score_dict]
        hu.save_pkl(os.path.join(savedir, "score_list.pkl"), score_list)

        # Report scores
        print(pd.DataFrame(score_list).tail())
        print()

    print("Experiment done\n")


if __name__ == "__main__":
    # Specify arguments regarding save directory and job scheduler
    parser = argparse.ArgumentParser()

    # Get list of experiments
    parser.add_argument(
        "-e",
        "--exp_group"
    )

    # Define directory where experiments are saved
    parser.add_argument(
        "-sb",
        "--savedir_base",
        required=True
    )

    # Reset or resume experiment
    parser.add_argument(
        "-r", "--reset", default=0, type=int
    )

    # Select device (important for those without GPU)
    parser.add_argument(
        "-d", "--device", default='cuda'
    )
    args, others = parser.parse_known_args()

    # Define a list of experiments (Tip: use python to get different hyperparameters)
    if args.exp_group == "baselines":
        exp_list = []
        # enumerate different hyperparameters - greedy search is usually desirable
        for lr in [1e-1, 1e-2, 1e-3]:
            exp_list += [{"dataset": "digits", "model": "linear", "lr": lr, "opt":'adam', "epochs":10}]

    # Run experiments and create results file
    hw.run_wizard(
        func=trainval,
        exp_list=exp_list,
        savedir_base=args.savedir_base,
        reset=args.reset,
        results_fname="results.ipynb",
        args=args,
    )