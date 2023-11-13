import pandas as pd
import numpy as np
import random
import os
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
from torchvision.models import resnet18
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

# Torch version
print('Torch version: ', torch.__version__)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


random_seed = 42
seed_everything(random_seed)

cudnn.benchmark = True

# Paths
IMAGES_PATH = "images"
LABELS = "icm.csv"
OUTPUTS = "outputs"
LOSS_FIG = "loss_fig"
ACCURACY_FIG = "accuracy_fig"

# Tuning parameters
IMG_SIZE = (224, 227)
BATCH_SIZE = 4
LEARNING_RATE = 1E-4
EPOCHS = 20


class SmartEmbryo(Dataset):
    """Image dataset"""

    def __init__(self, labels, img_dir, transform=None):
        """
        :param labels: class labels
        :param img_dir: the directory of the image folder
        :param transform: applied transforms
        """
        self.labels = labels
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        image = Image.open(self.img_dir[item])
        label = self.labels[item]

        if self.transform:
            return self.transform(image), label

        return image, label


class DeepLearningModel:
    def __init__(self, model, optimizer, loss_fn):
        self.batch_size = BATCH_SIZE
        self.epochs = EPOCHS
        self.learning_rate = LEARNING_RATE
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.best_vloss = 1_000_000
        self.model_path = os.path.join(OUTPUTS, f"simple_convolutional.model")
        self.model_saved = False

    def train_one_epoch(self, trainloader):

        self.model.train(True)

        running_loss = 0.
        avg_loss = 0.
        correct = 0.
        total = 0.

        for i, (inputs, labels) in enumerate(trainloader):

            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            if i % 20 == 19:
                avg_loss = running_loss / 20
                running_loss = 0.

        return avg_loss, (correct / total * 100)

    def validate(self, validloader):

        self.model.train(False)

        i = 0
        val_correct = 0.
        total = 0.

        with torch.no_grad():

            running_vloss = 0.

            for i, (inputs, labels) in enumerate(validloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                vloss = self.loss_fn(outputs, labels)

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                running_vloss += vloss.item()

            avg_vloss = running_vloss / (i + 1)

            if avg_vloss < self.best_vloss:
                self.best_vloss = avg_vloss
                self.model_saved = True
                torch.save(self.model.state_dict(), self.model_path)

        return avg_vloss, (val_correct / total * 100)

    def train(self, trainloader, validloader):

        train_history = []
        valid_history = []

        accuracy_history = [0.]
        accuracy_val_history = [0.]

        for epoch in range(EPOCHS):

            # Training
            avg_loss, avg_acc = self.train_one_epoch(trainloader)
            train_history.append(avg_loss)
            accuracy_history.append(avg_acc)

            # Validation
            avg_val_loss, avg_val_acc = self.validate(validloader)
            valid_history.append(avg_val_loss)
            accuracy_val_history.append(avg_val_acc)

            print(
                f"EPOCH: {epoch + 1} / {EPOCHS} -> LOSS -> train: {avg_loss:.4f}, valid: {avg_val_loss:.4f}, train_acc:"
                f" {avg_acc:.2f} %, valid_acc: {avg_val_acc:.2f} %")
            if self.model_saved:
                print("MODEL SAVED")
                self.model_saved = False

        path = os.path.join(OUTPUTS, f'{LOSS_FIG}.png')
        plt.plot(range(EPOCHS), train_history, label='Train')
        plt.plot(range(EPOCHS), valid_history, label='Validation')
        plt.xticks(range(EPOCHS + 1))
        plt.legend()
        plt.savefig(path)
        plt.close()

        path = os.path.join(OUTPUTS, f'{ACCURACY_FIG}.png')
        plt.plot(range(EPOCHS + 1), accuracy_history, label='Train')
        plt.plot(range(EPOCHS + 1), accuracy_val_history, label='Validation')
        plt.xticks(range(EPOCHS + 2))
        plt.ylim(0, 100)
        plt.legend()
        plt.savefig(path)
        plt.close()

    def test(self, testloader):
        test_model = resnet18(weights='ResNet18_Weights.DEFAULT')
        num_ftrs = test_model.fc.in_features
        test_model.fc = torch.nn.Linear(num_ftrs, 3)

        test_model.load_state_dict(torch.load(self.model_path))

        test_model.to(self.device)

        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = test_model(inputs)

                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Accuracy of the network on the test images: {(100 * correct / total):.2f} %')


if __name__ == '__main__':

    torch.cuda.empty_cache()

    # Choose GPU
    # torch.cuda.set_device(GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not OUTPUTS in os.listdir():
        os.mkdir(OUTPUTS)

    images_names = sorted(os.listdir(IMAGES_PATH))
    images_paths = [os.path.join(IMAGES_PATH, path) for path in images_names]

    df = pd.read_csv(LABELS)
    labels = df['grade'].values
    print(len(labels))
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.295, 0.295, 0.295), (0.201, 0.206, 0.209))  # Normalization based on Embryoset
    ])

    embryoset = SmartEmbryo(labels, images_paths, transform)

    train_size = int(0.85 * len(embryoset))
    test_size = len(embryoset) - train_size
    trainset, testset = random_split(embryoset, [train_size, test_size])

    train_size = int(0.8 * len(trainset))
    val_size = len(trainset) - train_size
    trainset, validset = random_split(trainset, [train_size, val_size])

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    validloader = DataLoader(validset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # ResNet
    model = resnet18(weights='ResNet18_Weights.DEFAULT')
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 4)

    print("MODEL SUMMARY")
    print(model)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    print("------- DataLoaders Length --------")
    print(f"TrainLoader: {len(trainloader)}")
    print(f"ValidLoader: {len(validloader)}")
    print(f"TestLoader: {len(testloader)}")

    dl_model = DeepLearningModel(model, optimizer, loss_fn)
    dl_model.train(trainloader, validloader)
    dl_model.test(testloader)
