"""
evaluate.py

Evaluation runner for GRU4Rec.
Evaluates the trained Epoch 10 checkpoint on the test set.
"""

import os
import pickle
import torch
from torch.utils.data import DataLoader

from models.gru4rec.config import DEVICE, BATCH_SIZE
from models.gru4rec.dataset import GRU4RecDataset
from models.gru4rec.collate import collate_fn
from models.gru4rec.model import GRU4Rec
from models.gru4rec.evaluation import hit_rate, mrr, ndcg


CHECKPOINT_PATH = "checkpoints/gru4rec_epoch_10.pth"
TEST_DATA_PATH = "datasets/processed/test_sequences.pkl"


def evaluate():

    print("=" * 60)
    print("GRU4Rec Evaluation")
    print("=" * 60)

    # --------------------------------------------------
    # Check files
    # --------------------------------------------------

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    if not os.path.exists(TEST_DATA_PATH):
        raise FileNotFoundError(
            f"Test dataset not found: {TEST_DATA_PATH}"
        )

    # --------------------------------------------------
    # Load test sequences
    # --------------------------------------------------

    print("\nLoading Test Sequences...")

    with open(TEST_DATA_PATH, "rb") as f:
        test_sequences = pickle.load(f)

    print(f"Testing Samples : {len(test_sequences)}")

    dataset = GRU4RecDataset(test_sequences)

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    print(f"Test Batches    : {len(dataloader)}")

    # --------------------------------------------------
    # Determine number of items
    # --------------------------------------------------

    num_items = max(
        sample["target"]
        for sample in test_sequences
    ) + 1

    print(f"Number of Movies: {num_items}")

    # --------------------------------------------------
    # Load model
    # --------------------------------------------------

    model = GRU4Rec(num_items)
    model.to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(f"Checkpoint      : {CHECKPOINT_PATH}")
    print(f"Model Device    : {DEVICE}")

    # --------------------------------------------------
    # Metric accumulators
    # --------------------------------------------------

    hit5_total = 0.0
    hit10_total = 0.0

    mrr5_total = 0.0
    mrr10_total = 0.0

    ndcg5_total = 0.0
    ndcg10_total = 0.0

    total_samples = 0

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------

    print("\nStarting Evaluation...\n")

    with torch.no_grad():

        for batch_idx, (
            inputs,
            targets,
            lengths
        ) in enumerate(dataloader):

            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            outputs = model(
                inputs,
                lengths
            )

            top5 = torch.topk(
                outputs,
                k=5,
                dim=1
            ).indices

            top10 = torch.topk(
                outputs,
                k=10,
                dim=1
            ).indices

            for i in range(targets.size(0)):

                target = targets[i].item()

                predictions5 = (
                    top5[i].cpu().tolist()
                )

                predictions10 = (
                    top10[i].cpu().tolist()
                )

                hit5_total += hit_rate(
                    predictions5,
                    target
                )

                hit10_total += hit_rate(
                    predictions10,
                    target
                )

                mrr5_total += mrr(
                    predictions5,
                    target
                )

                mrr10_total += mrr(
                    predictions10,
                    target
                )

                ndcg5_total += ndcg(
                    predictions5,
                    target
                )

                ndcg10_total += ndcg(
                    predictions10,
                    target
                )

            total_samples += targets.size(0)

            if (
                (batch_idx + 1) % 50 == 0
                or (batch_idx + 1) == len(dataloader)
            ):
                print(
                    f"Batch [{batch_idx + 1}/"
                    f"{len(dataloader)}] evaluated"
                )

    # --------------------------------------------------
    # Final metrics
    # --------------------------------------------------

    hit5 = hit5_total / total_samples
    hit10 = hit10_total / total_samples

    mrr5 = mrr5_total / total_samples
    mrr10 = mrr10_total / total_samples

    ndcg5 = ndcg5_total / total_samples
    ndcg10 = ndcg10_total / total_samples

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("GRU4Rec Evaluation Results")
    print("=" * 60)

    print(f"Test Samples : {total_samples}")

    print("\nRanking Metrics:")
    print(f"Hit@5   : {hit5:.4f}")
    print(f"Hit@10  : {hit10:.4f}")

    print(f"MRR@5   : {mrr5:.4f}")
    print(f"MRR@10  : {mrr10:.4f}")

    print(f"NDCG@5  : {ndcg5:.4f}")
    print(f"NDCG@10 : {ndcg10:.4f}")

    print("=" * 60)
    print("Evaluation Completed Successfully!")
    print("=" * 60)


if __name__ == "__main__":
    evaluate()