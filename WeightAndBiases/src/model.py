import torch
import torch.nn as nn
import torch.nn.functional as F


class CifarClassifier(nn.Module):
    # TODO make CNN tunable by args
    def __init__(self, output_hidden_states: int) -> None:
        super(CifarClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.linear1 = nn.Linear(16 * 5 * 5, 120)
        self.linear2 = nn.Linear(120, output_hidden_states)
        self.linear3 = nn.Linear(output_hidden_states, 10)
        self.flatten = nn.Flatten()

        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, batch_input_tensors: torch.tensor) -> torch.tensor:
        x = self.max_pool(F.relu(self.conv1(batch_input_tensors)))
        x = self.max_pool(F.relu(self.conv2(x)))
        x = self.flatten(x)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        emissions = self.linear3(x)

        return emissions

    def predict(self, emissions: torch.tensor) -> torch.tensor:
        return torch.argmax(emissions, dim=1)

    def compute_loss(
        self, emissions: torch.tensor, labels: torch.tensor
    ) -> torch.tensor:
        return self.cross_entropy_loss(emissions, labels)
