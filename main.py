# main.py

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import wandb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from torch.utils.data import DataLoader

from dataset import WineDataset
from model import WineClassifier
from train import train_epoch, eval_epoch


# ==================================================
# Seed
# ==================================================

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)


# ==================================================
# Fonction principale
# ==================================================

def run_experiment(optimizer_name="SGD"):

    # ----------------------------------------------
    # Seed
    # ----------------------------------------------

    set_seed()

    # ----------------------------------------------
    # wandb
    # ----------------------------------------------

    wandb.init(

        project="wine-classification",

        name=optimizer_name

    )

    # ----------------------------------------------
    # Chargement CSV
    # ----------------------------------------------

    df = pd.read_csv("data/wine.csv")

    print(df.head())

    print(df.describe())

    print(df.isnull().sum())

    # ----------------------------------------------
    # Features et labels
    # ----------------------------------------------
    target_col = "Wine"

    X = df.drop(columns=[target_col]).values
    y = df[target_col].values

    # Classes : 1 2 3
    # devient : 0 1 2

    y = y - 1

    # ----------------------------------------------
    # Normalisation
    # ----------------------------------------------

    scaler = StandardScaler()

    X = scaler.fit_transform(X)

    # ----------------------------------------------
    # Split train / eval
    # ----------------------------------------------

    X_train, X_eval, y_train, y_eval = train_test_split(

        X,
        y,

        test_size=0.2,

        stratify=y,

        random_state=42

    )

    # ----------------------------------------------
    # Dataset
    # ----------------------------------------------

    train_dataset = WineDataset(
        X_train,
        y_train
    )

    eval_dataset = WineDataset(
        X_eval,
        y_eval
    )

    # ----------------------------------------------
    # DataLoader
    # ----------------------------------------------

    train_loader = DataLoader(

        train_dataset,

        batch_size=16,

        shuffle=True

    )

    eval_loader = DataLoader(

        eval_dataset,

        batch_size=16,

        shuffle=False

    )

    # ----------------------------------------------
    # Device
    # ----------------------------------------------

    device = torch.device(

        "cuda"
        if torch.cuda.is_available()
        else "cpu"

    )

    print("Device :", device)

    # ----------------------------------------------
    # Modèle
    # ----------------------------------------------

    model = WineClassifier().to(device)

    # ----------------------------------------------
    # Loss
    # ----------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # ----------------------------------------------
    # Optimizer
    # ----------------------------------------------

    if optimizer_name == "SGD":

        optimizer = optim.SGD(

            model.parameters(),

            lr=0.01

        )

    else:

        optimizer = optim.Adam(

            model.parameters(),

            lr=0.01

        )

    # ----------------------------------------------
    # Hyperparamètres wandb
    # ----------------------------------------------

    wandb.config.update({

        "learning_rate": 0.01,

        "batch_size": 16,

        "epochs": 50,

        "optimizer": optimizer_name

    })

    # ----------------------------------------------
    # Sauvegarde meilleur modèle
    # ----------------------------------------------

    best_eval_loss = float("inf")

    # ----------------------------------------------
    # Boucle entraînement
    # ----------------------------------------------

    for epoch in range(50):

        train_loss, train_acc = train_epoch(

            model,

            train_loader,

            criterion,

            optimizer,

            device

        )

        eval_loss, eval_acc = eval_epoch(

            model,

            eval_loader,

            criterion,

            device

        )

        # ------------------------------------------
        # Affichage
        # ------------------------------------------

        print(f"\nEpoch {epoch+1}")

        print(

            f"Train Loss : {train_loss:.4f} | "
            f"Train Acc : {train_acc:.4f}"

        )

        print(

            f"Eval Loss : {eval_loss:.4f} | "
            f"Eval Acc : {eval_acc:.4f}"

        )

        # ------------------------------------------
        # wandb log
        # ------------------------------------------

        wandb.log({

            "train_loss": train_loss,

            "eval_loss": eval_loss,

            "train_accuracy": train_acc,

            "eval_accuracy": eval_acc

        })

        # ------------------------------------------
        # Sauvegarde meilleur modèle
        # ------------------------------------------

        if eval_loss < best_eval_loss:

            best_eval_loss = eval_loss

            torch.save(

                model.state_dict(),

                f"models/best_model_{optimizer_name}.pth"

            )

    wandb.finish()


# ==================================================
# Main
# ==================================================

if __name__ == "__main__":

    print("\n===== EXPERIMENT SGD =====")

    run_experiment("SGD")

    print("\n===== EXPERIMENT ADAM =====")

    run_experiment("Adam")