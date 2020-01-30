# External package imports
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import tqdm

# Custom module imports
from data_utils.quickdraw_dataloader import QuickDrawDataset
from models.listener import ListeNet

# Native Python imports
import os

# Specify hyperparameters here
EPOCHS = 10
BATCH_SIZE = 32
TRAIN_TEST_SPLIT = 0.8
LEARNING_RATE = 0.001
SAVE_INTERVAL = 5

# Specify directories here
CWD = os.getcwd()
DATA_DIR = os.path.join(CWD, "data")
SAVE_DIR = os.path.join(CWD, "saved_models", "listener")

# If directories don't exist, make them
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR, exist_ok=True)


def train_listener(data_dir):
    # Ingest dataset
    files = os.listdir(data_dir)
    # TODO: Load data

    # Split data into training and testing
    # TODO: Shuffle Data

    # Find index of shuffled data to split
    N = len(data)
    split = int(N * TRAIN_TEST_SPLIT)

    # Now split data
    train_data = data[:split]
    val_data = data[split:split + N // 10]
    test_data = data[split + N // 10:]

    # Now create Datasets - TODO: Format input arguments correctly
    train_dataset = QuickDrawDataset(train_data)
    val_dataset = QuickDrawDataset(val_data)
    test_dataset = QuickDrawDataset(test_data)

    # Now create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Initialize listener network with default parameters
    listener = ListeNet()

    # Define optimizer
    opt = optim.Adam(listener.parameters(), lr=LEARNING_RATE)

    # Define loss function
    loss_obj = F.nll_loss

    # Create dictionary for logging
    history = {'loss_train': [], 'loss_val': []}

    # Now create training loop
    for e in range(EPOCHS):

        # Reset counters
        batch_number = 0
        losses = []

        # Batched training loop
        listener = listener.train()
        for points, classes in tqdm(train_dataloader):
            # Compute predictions on training set
            preds = listener(points)

            # Compute losses on training set
            loss = loss_obj(preds, classes)

            # Append training loss values to history
            history['loss_train'].append(loss.cpu().data.numpy())

            # Clear previous gradients
            opt.zero_grad()

            # Compute new gradients and step
            loss.backward()
            opt.step()

        # Batched evaluation loop
        listener = listener.eval()
        for points, classes in tqdm(val_dataloader):
            # Compute predictions on validation set
            preds = listener(points)

            # Compute losses on validation set
            loss = loss_obj(preds, classes)

            # Append valdation loss values to history
            history['loss_val'].append(loss.cpu().data.numpy())

        # If on a save interval, save model
        if e % SAVE_INTERVAL == 0 & e != 0:
            # TODO: Specify save path/fname
            listener.save()


def main():
    train_listener(DATA_DIR)


if __name__ == '__main__':
    main()
