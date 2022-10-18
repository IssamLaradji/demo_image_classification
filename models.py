import torch
import tqdm
import pandas as pd


def get_model(exp_dict, device):
    """
    Tip: only use this function for getting models for modularity
    """
    model_name = exp_dict['model']

    # Tip: use if/elif/else to switch between models
    if model_name == "linear":
        # acquire Pytorch model (it could be anything but has to be consistent!)
        model = Linear(opt=exp_dict['opt'], lr=exp_dict['lr'], device=device)

    else:
        # Tip: include this to avoid silent bugs
        raise ValueError(f'{model_name} not found')

    return model


# =====================================================
class Linear(torch.nn.Module):
    def __init__(self, opt, lr, device):
        super().__init__()

        # Get the model (Tip: make sure you define the device before opt)
        self.model = torch.nn.Linear(64, 10).to(device)

        # choose between optimizers
        if opt == "adam":
            self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        elif opt == 'sgd':
            self.opt = torch.optim.SGD(self.model.parameters(), lr=lr)
        else:
            raise ValueError(f'{opt} not found')

        self.device = device

    def train_on_loader(self, loader):
        # Set model to training mode (important)
        self.train()

        train_list = []
        for batch in tqdm.tqdm(loader, desc="Training"):
            # Extract the data
            images, labels = batch
            images, labels = images.to(self.device), labels.to(self.device)
            logits = self.model.forward(images.view(images.shape[0], -1))

            # Compute loss
            criterion = torch.nn.CrossEntropyLoss(reduction="mean")
            loss = criterion(logits, labels.view(-1))

            # Perform optimization step
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            # Add score
            train_list += [{
                "train_loss": loss.item(),
                "train_acc": (logits.argmax(dim=1) == labels).float().mean().item(),
            }]

        # compute the average across train_list
        train_dict = pd.DataFrame(train_list).mean().to_dict()
        return train_dict

    def val_on_loader(self, loader):
        # Set model to eval mode (important)
        self.eval()

        val_list = []
        for batch in tqdm.tqdm(loader, desc="Validating"):
            # Extract the data
            images, labels = batch
            images, labels = images.to(self.device), labels.to(self.device)
            logits = self.model.forward(images.view(images.shape[0], -1))

            # Add score
            val_list += [{"val_acc": (logits.argmax(dim=1) == labels).float().mean().item()}]
        
        # compute the average across val_list
        val_dict = pd.DataFrame(val_list).mean().to_dict()
        return val_dict