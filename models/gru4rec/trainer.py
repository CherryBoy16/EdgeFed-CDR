"""
trainer.py

Training pipeline for GRU4Rec.
Resumes training from the latest completed checkpoint.
"""

import os
import pickle
import torch
from torch.utils.data import DataLoader

from models.gru4rec.config import (
    DEVICE,
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
)

from models.gru4rec.dataset import GRU4RecDataset
from models.gru4rec.collate import collate_fn
from models.gru4rec.model import GRU4Rec
from models.gru4rec.loss import get_loss_function


CHECKPOINT_DIR = "checkpoints"
RESUME_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "gru4rec_epoch_2.pth"
)


def build_training_components():

    print("=" * 60)
    print("Loading Training Sequences...")
    print("=" * 60)

    with open(
        "datasets/processed/train_sequences.pkl",
        "rb"
    ) as f:
        train_sequences = pickle.load(f)

    print(f"Training Samples : {len(train_sequences)}")

    dataset = GRU4RecDataset(train_sequences)

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    print("DataLoader Created.")

    num_items = max(
        sample["target"]
        for sample in train_sequences
    ) + 1

    print(f"Number of Movies : {num_items}")

    model = GRU4Rec(num_items)
    model.to(DEVICE)

    print(f"Model Loaded on {DEVICE}")

    criterion = get_loss_function()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-5
    )

    print("Loss Function Loaded.")
    print("Optimizer Loaded.")

    print("=" * 60)
    print("Training Pipeline Ready")
    print("=" * 60)

    return (
        dataloader,
        model,
        criterion,
        optimizer
    )


def load_checkpoint(model, optimizer):

    if not os.path.exists(RESUME_CHECKPOINT):
        print("\n[WARNING] Resume checkpoint not found!")
        print(f"Expected: {RESUME_CHECKPOINT}")
        print("Starting from Epoch 1.")
        return 0

    print("\n" + "=" * 60)
    print("Loading Resume Checkpoint")
    print("=" * 60)

    checkpoint = torch.load(
        RESUME_CHECKPOINT,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    completed_epoch = checkpoint["epoch"]
    saved_loss = checkpoint.get("loss", None)

    print(
        f"Checkpoint Loaded : {RESUME_CHECKPOINT}"
    )

    print(
        f"Completed Epoch   : {completed_epoch}"
    )

    if saved_loss is not None:
        print(
            f"Saved Avg Loss    : {saved_loss:.4f}"
        )

    print(
        f"Resuming from     : Epoch {completed_epoch + 1}"
    )

    print("=" * 60)

    return completed_epoch


def train():

    dataloader, model, criterion, optimizer = (
        build_training_components()
    )

    completed_epoch = load_checkpoint(
        model,
        optimizer
    )

    print("\nStarting Training...\n")

    model.train()

    for epoch in range(
        completed_epoch,
        EPOCHS
    ):

        print(
            f"\n========== Epoch "
            f"{epoch + 1}/{EPOCHS} ==========\n"
        )

        running_loss = 0.0

        total_batches = len(dataloader)

        for batch_idx, (
            inputs,
            targets,
            lengths
        ) in enumerate(dataloader):

            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(
                inputs,
                lengths
            )

            loss = criterion(
                outputs,
                targets
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5
            )

            optimizer.step()

            running_loss += loss.detach().item()

            if (
                (batch_idx + 1) % 100 == 0
                or (batch_idx + 1) == total_batches
                or batch_idx == 0
            ):

                print(
                    f"Batch [{batch_idx + 1}/"
                    f"{total_batches}] | "
                    f"Loss = {loss.item():.4f}"
                )

        avg_loss = (
            running_loss /
            (batch_idx + 1)
        )

        print("=" * 60)
        print(
            f"Epoch {epoch + 1} Completed"
        )
        print(
            f"Average Loss : {avg_loss:.4f}"
        )
        print("=" * 60)

        os.makedirs(
            CHECKPOINT_DIR,
            exist_ok=True
        )

        checkpoint_path = (
            f"{CHECKPOINT_DIR}/"
            f"gru4rec_epoch_{epoch + 1}.pth"
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            },
            checkpoint_path,
        )

        print(
            f"Checkpoint Saved -> "
            f"{checkpoint_path}"
        )

    print(
        "\n[+] Training Finished Successfully!"
    )


if __name__ == "__main__":
    train()