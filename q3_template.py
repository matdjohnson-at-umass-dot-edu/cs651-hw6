import os
import torch
import numpy as np
import torch.utils.data as utils
import torch.nn as nn
import torch.optim as optim
import random
import json
import matplotlib.pyplot as plt
from torchprofile import profile_macs


from sklearn.metrics import accuracy_score

# data dir is the mnist folder
"""
Load data to pytorch data loader
"""
def get_dataloader(data_dir, batch_size=32):
    """
    :param data_dir: folder directory of the data
    :param batch_size: batch size
    :return: a list of [trainloader, valloader, testloader]
    """
    def get_split_loader(split):
        data_images = np.load(os.path.join(data_dir, f"{split}_data_merged.npy"))

        # labels[:, 0]: figures of the first image
        # labels[:, 1]: figures of the second image
        # labels[:, 2]: figures of the mul of the previous two
        labels = np.load(os.path.join(data_dir, f"{split}_labels_merged.npy"))  # (B, 3)
        labels[:, 2] = labels[:, 0] * labels[:, 1]  # do the multiplication

        images1 = data_images[:, 0, :]  # (B, 784), 0-1 scale flattened image
        images2 = data_images[:, 1, :]  # (B, 784), 0-1 scale flattened image
        images1 = images1.reshape([-1, 1, 28, 28])
        images2 = images2.reshape([-1, 1, 28, 28])

        torch_dataset = utils.TensorDataset(
            torch.from_numpy(images1).float(),
            torch.from_numpy(images2).float(),
            torch.from_numpy(labels).long()
        )

        if split == "train":
            torch_loader = utils.DataLoader(torch_dataset, batch_size=batch_size, shuffle=True)
        else:
            torch_loader = utils.DataLoader(torch_dataset, batch_size=batch_size)

        return torch_loader

    return get_split_loader("train"), get_split_loader("val"), get_split_loader("test")


def create_model(fusion):
    if fusion == "early_fusion":
        return EarlyFusion()
    elif fusion == "late_fusion":
        return LateFusion()
    else:
        raise ValueError("Invalid fusion type")


def train(model, device, train_loader, val_loader, optimizer, epoch):
    model.train()

    train_loss = 0

    loss_fn = nn.CrossEntropyLoss()

    for batch_idx, (image1, image2, merged_labels) in enumerate(train_loader):
        image1, image2, mul_labels = image1.to(device), image2.to(device), merged_labels[:, 2].to(device)

        optimizer.zero_grad()

        output = model(image1, image2)

        loss = loss_fn(output, mul_labels)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:  # Print loss every 100 batch
            print('Train Epoch: {}\tLoss: {:.6f}'.format(
                epoch, loss.item()))

        train_loss += loss.item()

    train_loss /= len(train_loader)

    val_loss, val_acc = test(model, device, val_loader)
    _, train_acc = test(model, device, train_loader)

    return train_loss, train_acc, val_loss, val_acc


def test(model, device, torch_loader):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    epoch_loss = 0
    gt_labels = []
    pred_labels = []

    with torch.no_grad():
        for image1, image2, merged_labels in torch_loader:
            image1, image2, mul_labels = image1.to(device), image2.to(device), merged_labels[:, 2].to(device)
            output = model(image1, image2)
            loss = loss_fn(output, mul_labels)
            epoch_loss += loss.item()

            gt_labels.append(mul_labels.detach().cpu().numpy())
            pred_labels.append(np.argmax(output.detach().cpu().numpy(), axis=1))

    epoch_loss /= len(torch_loader)
    gt_labels = np.concatenate(gt_labels)
    pred_labels = np.concatenate(pred_labels)

    return epoch_loss, accuracy_score(gt_labels, pred_labels)


def main(fusion, optimizer_types):
    """
    :param fusion: str, early_fusion or late_fusion
    :param optimizer_types: list, names of the optimizer you plan to use
    :return:
    """
    seed = 42
    """
    Fix the random seed for reproducibility
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    """
    Alternatively, you can also use the following code to select the device
    Init the device based on your hardware:

    1.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # <- if your device support cuda

    2.
    device = torch.device("cpu") # <- if you want to force the device to use cpu

    3.
    device = "mps" if torch.backends.mps.is_available() else "cpu" # <- if you are running on mac with m chip
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    for optimizer_type in optimizer_types:
        """
        Training settings
        """
        NumEpochs = 10
        batch_size = 32
        os.makedirs(f"{fusion}_{optimizer_type}_report", exist_ok=True)  # make relative path
        output_dir = f"{fusion}_{optimizer_type}_report"

        model = create_model(fusion=fusion).to(device)

        complexity(model)

        # An example could be:
        #
        # if optimizer_type == "optimizer_type_1":
        #     optimizer = ...
        # elif optimizer_type == "optimizer_type_2":
        #     optimizer = ...
        # else:
        #     raise ValueError("Invalid optimizer type")
        if optimizer_type == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=0.001, betas=[0.999, 0.999])
        elif optimizer_type == "Adadelta":
            optimizer = optim.Adadelta(model.parameters(), rho=0.99, eps=1e-5)
        elif optimizer_type == "Adagrad":
            optimizer = optim.Adagrad(model.parameters(), lr=0.01, weight_decay=0.001)
        elif optimizer_type == "RMSprop":
            optimizer = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9, centered=False, weight_decay=0.001)
        elif optimizer_type == "AdamW":
            optimizer = optim.AdamW(model.parameters(), lr=0.001, betas=[0.99, 0.999], weight_decay=0.01)
        elif optimizer_type == "Adamax":
            optimizer = optim.Adamax(model.parameters(), lr=0.001, betas=[0.9, 0.999], eps=1e-5, weight_decay=0)
        else:
            raise ValueError("Invalid optimizer type")

        train_loader, val_loader, test_loader = get_dataloader(data_dir="mnist_data", batch_size=batch_size)

        best_val_acc = 0

        # historical performance
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        for epoch in range(NumEpochs):
            print("############# epoch: ", epoch)

            train_loss, train_acc, val_loss, val_acc = train(model, device, train_loader, val_loader, optimizer, epoch)

            print('\nTrain loss: {:.6f}, acc: {:.6f}\n'.format(train_loss, train_acc))
            print('\nVal loss: {:.6f}, acc: {:.6f}\n'.format(val_loss, val_acc))
            print("############# End of epoch: ", epoch)
            print()

            # record epoch train mse
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            np.save(os.path.join(output_dir, "train_losses.npy"), train_losses)
            np.save(os.path.join(output_dir, "train_accuracies.npy"), train_accuracies)

            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            np.save(os.path.join(output_dir, "val_losses.npy"), val_losses)
            np.save(os.path.join(output_dir, "val_accuracies.npy"), val_accuracies)

            # save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "val_acc": val_acc,
                        "val_loss": val_loss
                    },
                    os.path.join(output_dir, "mnist_model.pt"))

            """
            Do not report these plots, report plots by implementing plot function to visualize all optimizers performance
            """
            plt.plot(range(epoch + 1), train_losses, label='train loss')
            plt.plot(range(epoch + 1), val_losses, label='val loss')
            plt.legend()
            plt.savefig(os.path.join(output_dir, "train_val_loss.png"))
            plt.close()

            plt.plot(range(epoch + 1), train_accuracies, label='train acc')
            plt.plot(range(epoch + 1), val_accuracies, label='val acc')
            plt.legend()
            plt.savefig(os.path.join(output_dir, "train_val_acc.png"))
            plt.close()

        # test on test split
        best_cp = torch.load(os.path.join(output_dir, "mnist_model.pt"))
        model.load_state_dict(best_cp["state_dict"])
        test_loss, test_acc = test(model, device, test_loader)
        print("Best epoch: {}, Test loss:  {:.6f}, Test acc:  {:.6f}\n".format(best_cp["epoch"], test_loss, test_acc))
        with open(os.path.join(output_dir, "test_summary.json"), "w") as f:
            json.dump({"test_loss": test_loss, "test_acc": test_acc,
                       "val_acc": best_cp["val_acc"], "val_loss": best_cp["val_loss"],
                       "epoch": best_cp["epoch"]}, f, indent=4)


class EarlyFusion(nn.Module):
    def __init__(self):
        super(EarlyFusion, self).__init__()
        self.c1 = torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(3, 3), stride=1, padding=0)
        self.a1 = torch.nn.ReLU()
        self.c2 = torch.nn.Conv2d(in_channels=2, out_channels=4, kernel_size=(3, 3), stride=1, padding=0)
        self.a2 = torch.nn.ReLU()
        self.p1 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
        self.d1 = torch.nn.Dropout(p=0.25)
        self.l1 = torch.nn.Linear(in_features=1248, out_features=128)
        self.a3 = torch.nn.ReLU()
        self.l2 = torch.nn.Linear(in_features=128, out_features=82)

    def forward(self, x1, x2):
        input = torch.concatenate([x1, x2], dim=-1)
        output_1 = self.a1(self.c1(input))
        output_2 = self.a2(self.c2(output_1))
        output_3 = self.d1(self.p1(output_2))
        output = self.l2(self.a3(self.l1(output_3.flatten(start_dim=1))))
        return output


class LateFusion(nn.Module):
    def __init__(self):
        super(LateFusion, self).__init__()
        self.i1_c1 = torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(3, 3), stride=1, padding=0)
        self.i1_a1 = torch.nn.ReLU()
        self.i1_c2 = torch.nn.Conv2d(in_channels=2, out_channels=4, kernel_size=(3, 3), stride=1, padding=0)
        self.i1_a2 = torch.nn.ReLU()
        self.i1_p1 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
        self.i1_d1 = torch.nn.Dropout(p=0.25)
        self.i2_c1 = torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(3, 3), stride=1, padding=0)
        self.i2_a1 = torch.nn.ReLU()
        self.i2_c2 = torch.nn.Conv2d(in_channels=2, out_channels=4, kernel_size=(3, 3), stride=1, padding=0)
        self.i2_a2 = torch.nn.ReLU()
        self.i2_p1 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
        self.i2_d1 = torch.nn.Dropout(p=0.25)
        self.l1 = torch.nn.Linear(in_features=1152, out_features=128)
        self.a3 = torch.nn.ReLU()
        self.l2 = torch.nn.Linear(in_features=128, out_features=82)

    def forward(self, x1, x2):
        output_i1_1 = self.i1_a1(self.i1_c1(x1))
        output_i1_2 = self.i1_a2(self.i1_c2(output_i1_1))
        output_i1_3 = self.i1_d1(self.i1_p1(output_i1_2))
        output_i2_1 = self.i2_a1(self.i2_c1(x2))
        output_i2_2 = self.i2_a2(self.i2_c2(output_i2_1))
        output_i2_3 = self.i2_d1(self.i2_p1(output_i2_2))
        output = self.l2(self.a3(self.l1(
            torch.concatenate([output_i1_3.flatten(start_dim=1), output_i2_3.flatten(start_dim=1)], dim=-1))))
        return output


def plot(model_name, list_of_optimizers):
    # hint: you can load the train_accuracies.npy and val_accuracies.npy from the report dirs to avoid rerun training
    for optimizer in list_of_optimizers:
        train_accuracies = np.load(f"{model_name}_{optimizer}_report/train_accuracies.npy")
        plt.plot(range(len(train_accuracies)), train_accuracies, label=f"{optimizer} - train accuracy")
    plt.legend()
    plt.savefig(f"{model_name}-train-acc.png")
    plt.close()
    for optimizer in list_of_optimizers:
        train_losses = np.load(f"{model_name}_{optimizer}_report/train_losses.npy")
        plt.plot(range(len(train_losses)), train_losses, label=f"{optimizer} - train loss")
    plt.legend()
    plt.savefig(f"{model_name}-train-loss.png")
    plt.close()
    for optimizer in list_of_optimizers:
        val_accuracies = np.load(f"{model_name}_{optimizer}_report/val_accuracies.npy")
        plt.plot(range(len(val_accuracies)), val_accuracies, label=f"{optimizer} - val accuracy")
    plt.legend()
    plt.savefig(f"{model_name}-val-acc.png")
    plt.close()
    for optimizer in list_of_optimizers:
        val_losses = np.load(f"{model_name}_{optimizer}_report/val_losses.npy")
        plt.plot(range(len(val_losses)), val_losses, label=f"{optimizer} - val loss")
    plt.legend()
    plt.savefig(f"{model_name}-val-loss.png")
    plt.close()


def complexity(model):
    sample_data_1 = torch.rand([1, 1, 28, 28])
    sample_data_2 = torch.rand([1, 1, 28, 28])
    macs = profile_macs(model, (sample_data_1, sample_data_2))
    parameters = model.parameters(recurse=True)
    parameter_count = 0
    for parameter in parameters:
        parameter_count = parameter_count + parameter.numel()
    print(f"mac_ct:{macs}, parameter_ct:{parameter_count}")


if __name__ == '__main__':
    """
    str "early_fusion" inits the EarlyFusion model
    str "late_fusion" inits the LateFusion model
    """
    main("early_fusion", ["Adam", "Adadelta", "Adagrad", "RMSprop", "AdamW", "Adamax"])
    main("late_fusion", ["Adam", "Adadelta", "Adagrad", "RMSprop", "AdamW", "Adamax"])
    plot("early_fusion", ["Adam", "Adadelta", "Adagrad", "RMSprop", "AdamW", "Adamax"])
    plot("late_fusion", ["Adam", "Adadelta", "Adagrad", "RMSprop", "AdamW", "Adamax"])
    plot("early_fusion", ["Adam", "Adadelta", "AdamW"])
    plot("late_fusion", ["Adam", "Adadelta", "AdamW"])

