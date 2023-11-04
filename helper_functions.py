import torch

from tqdm.auto import tqdm
from typing import Dict, List, Tuple

# Function to train and evaluate the model
def train_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              criterion: torch.nn.Module,
              optimizer: torch.optim.Optimizer,
              device: torch.device):
  """
  This function performs a training step for a single epoch.
  Turns a pytoch model into training mode and then runs it 
  through all of the required training steps.

  Args:
    model: a Pytorch model
    train_loader: A DataLoader instance to train the model on.
    criterion: A pytorch loss function to minimize.
    optimizer: A pytorch optimizer to help minimize the loss function.
    device: a target device to compute on. e.g.("cpu" or "cuda")

  Returns:
    The Training loss
  """
  # put the model into traing mode
  model.train()
  # set the train loss value
  train_loss = 0.0

  # loop throught the batches of the DataLoader and train
  for batch, (X,y) in enumerate(dataloader):

    # send the data to target device
    X, y = X.to(device) , y.to(device)
    
    # forward pass through the model
    y_pred = model(X)
    
    # calculate and accumulate the loss
    loss = criterion(y_pred,y)
    train_loss += loss.item()

    # set optimizer zero grad
    optimizer.zero_grad()

    # loss backwards
    loss.backward()

    # optimizer step
    optimizer.step()

  # get average loss per batch 
  train_loss = train_loss / len(dataloader)

  return train_loss

def test_step(model: torch.nn.Module, 
              test_data: torch.Tensor, 
              criterion: torch.nn.Module,
              device: torch.device):
  """
  Tests a PyTorch model for a single epoch.
  Turns a target PyTorch model to "eval" mode and then performs
  a forward pass on a testing dataset.

  Args:
    model: A PyTorch model to be tested.
    test_data: A tuple of  torch.Tensor instance for the model to be tested on.
    criterion: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

  Returns:
    The test loss
  """
  # Put model in eval mode
  model.eval() 

  # Turn on inference context manager
  with torch.inference_mode():

      X, y = test_data
      # Send data to target device
      X, y = X.to(device), y.to(device)

      # Forward pass
      y_pred = model(X)

      # Calculate and accumulate loss
      loss = criterion(y_pred, y)
      test_loss = loss.item()

  return test_loss

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_data: torch.Tensor, 
          optimizer: torch.optim.Optimizer,
          criterion: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List]:
  """Trains and tests a PyTorch model.

  Passes a target PyTorch models through train_step() and test_step()
  functions for a number of epochs, training and testing the model
  in the same epoch loop.

  Calculates, prints and stores evaluation metrics throughout.

  Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_data: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    criterion: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").

  Returns:
    A dictionary of training and testing loss.
    Each metric has a value in a list for each epoch.
    In the form: {train_loss: [...],
                  test_loss: [...],
    For example if training for epochs=2: 
                 {train_loss: [2.0616, 1.0537],
                  test_loss: [1.2641, 1.5706],
  """
  # Create empty results dictionary
  results = {"train_loss": [],"test_loss": []}

  # Loop through training and testing steps for a number of epochs
  for epoch in tqdm(range(epochs)):
      train_loss = train_step(model=model,
                              dataloader=train_dataloader,
                              criterion=criterion,
                              optimizer=optimizer,
                              device=device)
      
      test_loss = test_step(model=model,
                            test_data=test_data,
                            criterion=criterion,
                            device=device)

      # Print out what's happening
      print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"test_loss: {test_loss:.4f} | "
      )

      # Update results dictionary
      results["train_loss"].append(train_loss)
      results["test_loss"].append(test_loss)

  # Return the filled results at the end of the epochs
  return results
