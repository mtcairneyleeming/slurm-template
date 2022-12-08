import jax.numpy as jnp
from jax import random


# Set up all the parameters for the model

n= 400;
x = jnp.arange(0, 1, 1/n)

args = {"num_epochs": 100,  # will change back to 100 soon
        "learning_rate": 1.0e-3, 
        "batch_size": 10, 
        "hidden_dim1": 35,
        "hidden_dim2": 30,
        "z_dim": 30,
        "x": x,
        "n": n,
        "gp_kernel": "exp_sq_kernel",
        "rng_key": random.PRNGKey(1),
        "num_warmup": 1000,
        "num_samples": 1000,
        "num_chains": 4,
        "thinning": 1
        }

