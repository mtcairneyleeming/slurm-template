# for saving model parameters/etc.
import pickle
import os

# for the actual task
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor





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


# the function run on ARC
def train(args):
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
    print("Done!")


    
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

    print("Saved all outputs")
    
        


