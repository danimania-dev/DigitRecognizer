import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader

trans_train = transforms.Compose([
    transforms.Grayscale(num_output_channels = 1),
    transforms.RandomAffine(degrees = 15, translate = (0.1, 0.1), scale = (0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trans_eval = transforms.Compose([
    transforms.Grayscale(num_output_channels = 1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Flatten(),
            nn.Linear(7 * 7 * 32, 128),
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)

    def predict(self, image_path):
        image = Image.open(image_path)
        image_trans = trans_eval(image)
        tensor_image = image_trans.unsqueeze(0)

        with torch.no_grad():
            result = self(tensor_image) 
            class_res = torch.argmax(result).item()
            
            return class_res


    def test_model(self, dataset_path):
        test_dataset = torchvision.datasets.ImageFolder(root = dataset_path, transform = trans_eval)
        test_loader = DataLoader(test_dataset, batch_size = 8, shuffle = True)

        correct = 0
        samples = 0
        test_loss = 0.0

        conf_mat = [[0 for _ in range(10)] for _ in range(10)]

        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                y_pred = self(batch_x)
                
                loss = criterion(y_pred, batch_y)
                test_loss += loss.item()
                
                _, predicted_classes = torch.max(y_pred, 1)

                y_list = batch_y.tolist()
                y_pred_list = predicted_classes.tolist()

                for i in range(len(y_list)):
                    if y_list[i] == y_pred_list[i]:
                        correct += 1

                    conf_mat[y_list[i]][y_pred_list[i]] += 1
                    samples += 1

        avg_test_loss = test_loss / len(test_loader)
        accuracy = (correct / samples) * 100

        print("Testing model with dataset: " + str(dataset_path))
        print("Average loss: " + str(avg_test_loss))
        print("Accuracy: " + str(accuracy) + "(" + str(correct) + " / " + str(samples) + ")")
        print("Confusion matrix: ")

        for i in range(10):
            for j in range(10):
                print(str(conf_mat[i][j]) + " ", end = '')
            print()

        print("\n\n")
