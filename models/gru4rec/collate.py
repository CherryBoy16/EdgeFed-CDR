import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    """
    Custom collate function for GRU4Rec.

    Pads sequences within a batch to the same length.
    """

    inputs, targets = zip(*batch)

    lengths = torch.tensor(
        [len(seq) for seq in inputs],
        dtype=torch.long
    )

    padded_inputs = pad_sequence(
        inputs,
        batch_first=True,
        padding_value=0
    )

    targets = torch.stack(targets)

    return padded_inputs, targets, lengths