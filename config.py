from torch import nn

args = {
        "batch_size": 64,
        "epochs": 10,
        "loss_fn": nn.CrossEntropyLoss(),
        "learning_rate": 1e-3
        }