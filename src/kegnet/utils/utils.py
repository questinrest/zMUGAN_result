"""
Knowledge Extraction with No Observable Data (NeurIPS 2019)

Authors:
- Jaemin Yoo (jaeminyoo@snu.ac.kr), Seoul National University
- Minyong Cho (chominyong@gmail.com), Seoul National University
- Taebum Kim (k.taebum@snu.ac.kr), Seoul National University
- U Kang (ukang@snu.ac.kr), Seoul National University

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.
"""
import os

import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def set_seed(seed):
    """
    Set a random seed for numpy and PyTorch.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def save_checkpoints(model, path):
    """
    Save a trained model.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(dict(model_state=model.state_dict()), path)


def load_checkpoints(model, path, device):
    """
    Load a saved model.

    
    """
    checkpoint = torch.load(path, map_location=DEVICE)
    classifier.load_state_dict(checkpoint)
    classifier = classifier.cuda()
    #model_state = torch.load(path, map_location=device)
    #checkpoint = torch.load(path, map_location=device)
    #model_state = checkpoint.get('model_state', None)
    #model.load_state_dict(model_state)
    #model.load_state_dict(model_state)
    # model.load_state_dict(torch.load(path, map_location=device))
    #org_model = AllCNN()  # Assuming AllCNN is defined elsewhere in your code
    #checkpoint = torch.load(path, map_location = device)
    #model.load_state_dict(checkpoint['state_dict'])




#def load_checkpoints(model, path, device):
    """
    Load a saved model checkpoint and restore the model state.
    
    Parameters:
    - model: the model object to load the state into.
    - path: the file path of the checkpoint.
    - device: the device on which the model is loaded (e.g., 'cpu' or 'cuda').
    
    Returns:
    - model: the model with the loaded state.
    """
    # Load the checkpoint from the path
   # checkpoint = torch.load(path, map_location=device)

    # Extract the model state dictionary from the checkpoint
   # model_state = checkpoint.get('state_dict', None)  # Using 'state_dict' if it's saved with this key

   # if model_state is None:
        #raise ValueError("Checkpoint doesn't contain 'state_dict' key.")
    
    # Load the state dictionary into the model
    #model.load_state_dict(model_state)

    # Return the model with the loaded weights
    #return model

    
