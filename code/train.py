# for saving model parameters/etc.
import pickle
import os

# for the actual task

# PyTorch
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# JAX
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax.scipy.special import logsumexp

import numpy as np
import time

####### PyTorch


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
        

def training_step(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()

    losses = []
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            losses.append(loss)
    return losses


def testing_step(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    return test_loss


################ JAX

# A helper function to randomly initialize weights and biases
# for a dense neural network layer
def random_layer_params(m, n, key, scale=1e-2):
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(sizes, key):
    keys = random.split(key, len(sizes))
    return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]



def relu(x):
    return jnp.maximum(0, x)

def predict(params, image):
    # per-example predictions
    activations = image
    for w, b in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = relu(outputs)

    final_w, final_b = params[-1]
    logits = jnp.dot(final_w, activations) + final_b
    return logits - logsumexp(logits)


def one_hot(x, k, dtype=jnp.float32):
    """Create a one-hot encoding of x of size k."""
    return jnp.array(x[:, None] == jnp.arange(k), dtype)



def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)

class NumpyLoader(DataLoader):
    def __init__(self, dataset, batch_size=1,
                shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0,
                pin_memory=False, drop_last=False,
                timeout=0, worker_init_fn=None):
        super(self.__class__, self).__init__(dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=numpy_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn)

class FlattenAndCast(object):
    def __call__(self, pic):
        return np.ravel(np.array(pic, dtype=jnp.float32))


############### Running both together

# the function run on ARC
def train(args):
    
    ######## PyTorch
    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # Download training data from open datasets.
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=False,
        transform=ToTensor(),
    )
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=False,
        transform=ToTensor(),
    )

    
    train_dataloader = DataLoader(training_data, batch_size=args["batch_size"])
    
    test_dataloader = DataLoader(test_data, batch_size=args["batch_size"])
    
    for X, y in train_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    model = NeuralNetwork().to(device)
    print(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=args["learning_rate"])
    
    epoch_train_losses = []
    epoch_test_losses = []

    for t in range(args["epochs"]):
        print(f"Epoch {t+1}\n-------------------------------")
        train_losses = training_step(train_dataloader, model, args["loss_fn"], optimizer, device)
        test_losses = testing_step(test_dataloader, model, args["loss_fn"], device)
        epoch_train_losses.append(train_losses)
        epoch_test_losses.append(test_losses)
    print("Done PyTorch!")
    
    ## JAX stuff
    print(jax.default_backend())
    print(jax.devices())
    
    params = init_network_params(args["jax_layer_sizes"], random.PRNGKey(0))

    # This works on single examples
    random_flattened_image = random.normal(random.PRNGKey(1), (28 * 28,))
    preds = predict(params, random_flattened_image)
    print(preds.shape)


    # Let's upgrade it to handle batches using `vmap`
    random_flattened_images = random.normal(random.PRNGKey(1), (10, 28 * 28))
    # Make a batched version of the `predict` function
    batched_predict = vmap(predict, in_axes=(None, 0))

    # `batched_predict` has the same call signature as `predict`
    batched_preds = batched_predict(params, random_flattened_images)
    print(batched_preds.shape)

    # Define our dataset, using torch datasets
    mnist_dataset = datasets.MNIST('data', download=False, transform=FlattenAndCast())
    training_generator = NumpyLoader(mnist_dataset, batch_size=args["jax_batch_size"], num_workers=0)
    
    # Get the full train dataset (for checking accuracy while training)
    train_images = np.array(mnist_dataset.train_data).reshape(len(mnist_dataset.train_data), -1)
    train_labels = one_hot(np.array(mnist_dataset.train_labels), args["jax_n_targets"])

    # Get full test dataset
    mnist_dataset_test = datasets.MNIST('data', download=False, train=False)
    test_images = jnp.array(mnist_dataset_test.test_data.numpy().reshape(len(mnist_dataset_test.test_data), -1), dtype=jnp.float32)
    test_labels = one_hot(np.array(mnist_dataset_test.test_labels), args["jax_n_targets"])

      
    def accuracy(params, images, targets):
        target_class = jnp.argmax(targets, axis=1)
        predicted_class = jnp.argmax(batched_predict(params, images), axis=1)
        return jnp.mean(predicted_class == target_class)

    def loss(params, images, targets):
        preds = batched_predict(params, images)
        return -jnp.mean(preds * targets)


    @jit
    def update(params, x, y, step_size):
        grads = grad(loss)(params, x, y)
        return [(w - step_size * dw, b - step_size * db)
              for (w, b), (dw, db) in zip(params, grads)]


    
    for epoch in range(args["jax_num_epochs"]):
        start_time = time.time()
        for x, y in training_generator:
            y = one_hot(y, args["jax_n_targets"])
            params = update(params, x, y, args["jax_step_size"])
        epoch_time = time.time() - start_time

        train_acc = accuracy(params, train_images, train_labels)
        test_acc = accuracy(params, test_images, test_labels)
        print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
        print("Training set accuracy {}".format(train_acc))
        print("Test set accuracy {}".format(test_acc))

    
    if os.path.isdir("output"): # running in an ARC job (using my submission script)
        save_path = "output"
    elif os.path.isdir("outputs/manual"): # running from the root directory of the git repo, not in a job
        save_path = "outputs/manual"
    else: # somewhere else?
        save_path = "."

    torch.save(model, save_path + "/model.pth")
    print("Saved PyTorch Model State to model.pth")

    with open(save_path +'/train_loss_list', 'wb+') as file:
        pickle.dump(epoch_train_losses, file)

    with open(save_path + '/test_loss_list', 'wb+') as file:
        pickle.dump(epoch_test_losses, file)

    
    with open(save_path + '/jax_params', 'wb+') as file:
        pickle.dump(params, file)
        
    print("Saved all outputs")
    
        


