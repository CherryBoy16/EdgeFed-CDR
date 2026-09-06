import torch

EMBEDDING_DIM = 128
HIDDEN_DIM = 128
NUM_LAYERS = 1
DROPOUT = 0.2
BATCH_SIZE = 256
LEARNING_RATE = 0.001
EPOCHS = 10
MAX_SEQUENCE_LENGTH = 50
PADDING_IDX = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")