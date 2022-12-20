from torch import nn

args = {
        "batch_size": 64,
        "epochs": 1,
        "loss_fn": nn.CrossEntropyLoss(),
        "learning_rate": 1e-3,
        "jax_layer_sizes" : [784, 512, 512, 10],
        "jax_step_size": 0.01,
        "jax_num_epochs": 8,
        "jax_batch_size": 128,
        "jax_n_targets": 10
        }