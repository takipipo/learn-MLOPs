import argparse
import logging
import torchvision
import torch
from torchvision import transforms
from model import CifarClassifier
from tqdm import tqdm, trange
from trainer import Trainer
import wandb


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)กs - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a model on CIFAR10 dataset")
    parser.add_argument("--input_dir", type=str, default=".", help="Input directory")
    parser.add_argument("--do_train", action="store_true", help="Whether to train the model")
    parser.add_argument("--do_val", action="store_true", help="Whether to validate the model")
    parser.add_argument("--do_test", action="store_true", help="Whether to validate the model")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--output_hidden_states", type=int, default=100, help="Number of hidden states in the output layer")
    parser.add_argument("--name", type=str, help="Experiment name to log in wandb")

    return parser  


def main():
    parser = get_parser()
    args = parser.parse_args()

    config = {
        "learning_rate":args.learning_rate, 
        "output_hidden_states":args.output_hidden_states, 
        "num_train_epochs":args.num_train_epochs,
        "batch_size":args.batch_size
    }
    wandb.init(project='experiment-CifarClassifier', config = config, name=args.name)



    # TODO move these module to data preprocessor module
    image_preprocessor = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # load the dataset
    train_dataset = torchvision.datasets.CIFAR10(root='./datasets', train=True,
                                            download=True, transform=image_preprocessor)
    dataset = [train_dataset[i] for i in range(100)]
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=4,
                                            shuffle=True, num_workers=2)
    test_dataset = torchvision.datasets.CIFAR10(root='./datasets', train=False,
                                        download=True, transform=image_preprocessor)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4,
                                            shuffle=False, num_workers=2)
    
    # Init the model
    model = CifarClassifier(output_hidden_states=args.output_hidden_states)
    model.to(device)

    trainer = Trainer(args, model, train_dataloader, device, wandb)
    model = trainer.train()


if __name__ == "__main__":
    main()

