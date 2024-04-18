import torch
import torch.nn as nn
import torch.nn.functional as F
# from _____ import data loader

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the LSTM model.
        Args:
        - input_size (int): Number of input features (e.g., number of pixels in an image)
        - hidden_size (int): Number of hidden units in the LSTM layer
        - output_size (int): Number of output classes or features (0-9 for each number and 10 for idk)
        """
        super(LSTMModel, self).__init__()
        # Define LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        # Define fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
      
    def forward(self, x, hidden):
        """
        Forward pass of the LSTM model.
        Args:
        - x (torch.Tensor): Input tensor with shape (batch_size, seq_length, input_size)
        - hidden (tuple): Tuple containing the initial hidden state and cell state
                          (h0, c0), both with shape (num_layers, batch_size, hidden_size)
        Returns:
        - out (torch.Tensor): Output tensor with shape (batch_size, output_size)
        - hidden (tuple): Updated hidden state and cell state after processing the input
                          (h_out, c_out), both with shape (num_layers, batch_size, hidden_size)
        """
        # Forward pass through LSTM layer
        out, hidden = self.lstm(x, hidden)
        # Extract the output from the last time step
        out = self.fc(out[:, -1, :])
        # Apply softmax activation function to make output a PMF
        out = F.softmax(out, dim=1)
        return out, hidden
    
    def init_hidden(self, batch_size):
        """
        Initialize the hidden state and cell state for the LSTM.
        Args:
        - batch_size (int): Number of sequences in a batch
        Returns:
        - hidden (tuple): Tuple containing the initial hidden state and cell state
                          (h0, c0), both with shape (num_layers, batch_size, hidden_size)
        """
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(1, batch_size, self.lstm.hidden_size).to(next(self.parameters()).device)
        c0 = torch.zeros(1, batch_size, self.lstm.hidden_size).to(next(self.parameters()).device)
        return (h0, c0)
      
    def train_model(self, dataloader, criterion, optimizer, num_epochs, save_path = None):
        """
        Train the LSTM model.
        Args:
        - dataloader (DataLoader): DataLoader object for batching and shuffling the data
        - criterion (nn.Module): Loss function (for this project, the criterion will be the KL divergence)
        - optimizer (torch.optim.Optimizer): Optimizer for updating model parameters
        - num_epochs (int): Number of training epochs
        - save_path: path for where to store the model state after trained
        """
        for epoch in range(num_epochs):
            self.train()  # Set the model to training mode
            total_loss = 0.0
            for batch_inputs, batch_targets in dataloader:
                # Initialize hidden state
                hidden = self.init_hidden(batch_inputs.size(0))
                # Forward pass
                outputs, hidden = self(batch_inputs, hidden)
                # Compute loss
                loss = criterion(outputs, batch_targets)
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            # Print average loss for the epoch
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}")
        if save_path:
            torch.save(self.state_dict(), save_path)
    def predict(self, new_data):
        """
        Perform predictions on new data.
        Args:
        - new_data (torch.Tensor): New input data with shape (batch_size, seq_length, input_size)
        Returns:
        - predicted_classes (torch.Tensor): Predicted class labels
        """
        self.eval()  # Set the model to evaluation mode
        # Initialize hidden state
        hidden = self.init_hidden(new_data.size(0))
        # Forward pass
        with torch.no_grad():  # No need to track gradients during inference
            outputs, _ = self(new_data, hidden)
        # Get predicted class labels
        _, predicted_classes = torch.max(outputs, 1)
        return predicted_classes

  
