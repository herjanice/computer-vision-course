
# Modelzoo for usage 
# Feel free to add any model you like for your final result
# Note : Pretrained model is allowed iff it pretrained on ImageNet

import torch
import torch.nn as nn

class myLeNet(nn.Module):
    def __init__(self, num_out):
        super(myLeNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3,6,kernel_size=5, stride=1),
                             nn.ReLU(),
                             nn.MaxPool2d(kernel_size=2, stride=2),
                             )
        self.conv2 = nn.Sequential(nn.Conv2d(6,16,kernel_size=5),
                             nn.ReLU(),
                             nn.MaxPool2d(kernel_size=2, stride=2),)
        
        self.fc1 = nn.Sequential(nn.Linear(400, 120), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(120,84), nn.ReLU())
        self.fc3 = nn.Linear(84,num_out)

    def forward(self, x):   

        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        
        # It is important to check your shape here so that you know how manys nodes are there in first FC in_features
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)      
        out = x
        return out

# reference = https://github.com/Brother-Lee/WRN-28-10-CIFAR10/blob/master/wideResNet.py

class residual_block(nn.Module):
    expansion = 1

    def __init__(self, c_in, c_out, stride=1, dropout_rate=0.3, downsample=None):
        super(residual_block, self).__init__()

        self.bn1 = nn.BatchNorm2d(c_in)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1, bias=False)
        
        self.dropout = nn.Dropout(p=dropout_rate)
        self.downsample = downsample
        
        self.bn2 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.stride = stride

    def forward(self, x):
        #TODO: DONE 
        # Perform residual network. 
        # You can refer to our ppt to build the block. It's ok if you want to do much more complicated one. 
        # i.e. pass identity to final result before activation function 
        
        identity = x
        out = self.bn1(x)
        out = self.relu(out)
        
        out = self.conv1(out)
        out = self.dropout(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        return out


class wideResNet(nn.Module):

    def __init__(self, block, layers, widen, dropout_rate=0.3, num_classes=10):
        super(wideResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)

        self.layer1 = self.make_layer(block, 16, 16*widen, layers[0], dropout_rate)
        self.layer2 = self.make_layer(block, 16*widen, 32*widen, layers[1], dropout_rate, stride=2)
        self.layer3 = self.make_layer(block, 32*widen, 64*widen, layers[2], dropout_rate, stride=2)

        self.batch_norm = nn.BatchNorm2d(64*widen)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64*widen, num_classes)

        for m in self.modules():

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0.1)


    def make_layer(self, block, c_in, c_out, blocks, dropout_rate, stride=1):
        downsample = None

        if c_in != c_out or stride != 1:
            downsample = nn.Sequential(nn.Conv2d(c_in*block.expansion, c_out*block.expansion, kernel_size=1, stride=stride, bias=False))

        layers = []
        layers.append(block(c_in*block.expansion, c_out, stride, dropout_rate, downsample))

        for _ in range(1, blocks):
            layers.append(block(c_out*block.expansion, c_out, dropout_rate=dropout_rate))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        out = x

        return out

def myResnet():
    """Constructs a wideResnet28_10 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = wideResNet(residual_block, [1, 1, 1], 4)
    return model
