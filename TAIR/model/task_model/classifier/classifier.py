import logging
import torch.nn as nn
from torchmetrics.classification import Accuracy

################################################
#  This is the base class for all classifiers  #
################################################

class Classifier(nn.Module):

    def __init__(self, n_classes: int):
        super(Classifier, self).__init__()
        
        self.train_acc = Accuracy(task="multiclass", num_classes=n_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=n_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=n_classes)
        
        self.accuracies = {
            "train": self.train_acc,
            "val": self.val_acc,
            "test": self.test_acc,
        }

    def forward(self, x):
        logging.error("The forward method must be implemented.")
        raise NotImplementedError()
    
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        return self

##############################################
#  Below are the implemation of classifiers  #
##############################################

from torchvision.models.resnet import resnet50, ResNet50_Weights
class ResNet50(Classifier):

    def __init__(self, n_classes: int):
        super(ResNet50, self).__init__(n_classes)
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet.fc = nn.Linear(in_features=self.resnet.fc.in_features, out_features=n_classes)

    def forward(self, x):
        return self.resnet(x)
    
from torchvision.models.resnet import resnet18, ResNet18_Weights
class ResNet18(Classifier):

    def __init__(self, n_classes: int):
        super(ResNet18, self).__init__(n_classes)
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Linear(in_features=self.resnet.fc.in_features, out_features=n_classes)

    def forward(self, x):
        return self.resnet(x)