import torch
from torch.utils.data import Dataset


class GRU4RecDataset(Dataset):
    """
    PyTorch Dataset for GRU4Rec.

    Each sample contains:
    - input sequence (previous interactions)
    - target item (next interaction)
    """

    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sample = self.sequences[index]

        input_sequence = torch.tensor(
            sample["input"],
            dtype=torch.long
        )

        target = torch.tensor(
            sample["target"],
            dtype=torch.long
        )

        return input_sequence, target