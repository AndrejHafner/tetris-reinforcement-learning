import torch.nn as nn
import torch.nn.functional as F

class DQCNN(nn.Module):

    def __init__(self, screen_w, screen_h, outputs, device):
        super(DQCNN, self).__init__()
        self.device = device

        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=1)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size = 5, stride = 1):
            return (size - (kernel_size - 1) - 1) // stride  + 1

        conv_w = conv2d_size_out(conv2d_size_out(conv2d_size_out(screen_w)))
        conv_h = conv2d_size_out(conv2d_size_out(conv2d_size_out(screen_h)))

        print("conv_w: ", conv_w)
        print("conv_h: ", conv_h)

        linear_input_size = conv_w * conv_h * 32
        print("linear_input_size: ", linear_input_size)


        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        return self.head(x.view(x.size(0), -1))
