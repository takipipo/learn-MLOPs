from torch.optim import Adam
from model import CifarClassifier
import argparse
from tqdm import tqdm, trange
import wandb


import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Trainer:
    def __init__(
        self,
        args: argparse.ArgumentParser,
        model: CifarClassifier,
        train_dataloader: DataLoader,
        device: torch.device,
        wandb: wandb,
    ) -> None:
        self.model = model
        self.num_train_epoch = args.num_train_epochs
        self.device = device
        self.optimizer = Adam(model.parameters(), lr=args.learning_rate)
        self.train_dataloader = train_dataloader

        self.wandb = wandb

    def train(self):

        for current_epoch in trange(self.num_train_epoch, desc="Epochs"):
            running_loss = 0.0

            for step, batch_data in enumerate(tqdm(self.train_dataloader, desc="Step")):

                # training mode
                self.model.train()

                # get the inputs
                batch_input_tensors, batch_input_targets = batch_data
                batch_input_tensors, batch_input_targets = batch_input_tensors.to(
                    self.device
                ), batch_input_targets.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                emissions = self.model(batch_input_tensors)
                loss = self.model.compute_loss(emissions, batch_input_targets)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            print(
                current_epoch + 1,
                step + 1,
                "loss:",
                running_loss / len(self.train_dataloader),
            )
            self.wandb.log(
                {
                    "epoch": current_epoch + 1,
                    "train_loss": running_loss / len(self.train_dataloader),
                }
            )

        print("Finished Training")
        
        return self.model
