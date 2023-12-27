"""
This code is based on the Torchvision repository, which was licensed under the BSD 3-Clause.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

def resnet18(**kwargs):
    return {'backbone': ResNet(BasicBlock, [2, 2, 2, 2], **kwargs), 'dim': 512}

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        model = torchvision.models.resnet18()
        self.model = torch.nn.Sequential(*(list(model.children())[:-1]))
    def forward(self,x):
        out = self.model(x)
        out = out.view(out.shape[0],out.shape[1])
        return out
def resnet18(**kwargs):
    return {'backbone': ResNet(), 'dim': 512}
if __name__ == '__main__':
    model = ResNet()
    # model = ResNet(BasicBlock, [2, 2, 2, 2])
    img = torch.FloatTensor(1,3,224,224)
    print(model(img).shape)