import torch
import torch.nn as nn
from torch.autograd import Function


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class CNR2d(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)):
        super().__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(ch_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))

        return x


class ResidualBlock(nn.Module):
    def __init__(self, dim=128):
        super(ResidualBlock, self).__init__()
        self.pw1 = CNR2d(dim, dim, kernel_size=1)
        self.dw = CNR2d(dim, dim, kernel_size=3, padding=1)
        self.pw2 = CNR2d(dim, dim, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        inp = x.clone()
        x = self.relu(self.pw1(x))
        x = self.relu(self.dw(x))
        x = self.pw2(x)

        return self.relu(x + inp)


class Feature_Encoder(nn.Module):
    def __init__(self):
        super(Feature_Encoder, self).__init__()
        self.conv1 = CNR2d(8, 32, kernel_size=(1, 7), stride=(1, 3))
        self.conv2 = CNR2d(32, 128, kernel_size=(1, 5), stride=(1, 2))
        self.residual_blocks = nn.Sequential(ResidualBlock(128), ResidualBlock(128))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)  # [2, 128, 7, 54]
        x = self.residual_blocks(x)

        return x


class DOA_Estimator(nn.Module):
    def __init__(self):
        super(DOA_Estimator, self).__init__()
        self.residual_blocks = nn.Sequential(ResidualBlock(128), ResidualBlock(128), ResidualBlock(128))
        self.pw1 = CNR2d(128, 360, kernel_size=1)
        self.pw2 = CNR2d(54, 500, kernel_size=1)
        self.conv = CNR2d(500, 1, kernel_size=(7, 5), padding=(0, 2))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x:[2, 128, 7, 54]
        x = self.residual_blocks(x)
        x = self.pw1(x)
        # swap axes  C<--->F
        x = x.permute(0, 3, 2, 1)  # [2, 54, 7, 360]
        x = self.pw2(x)  # [2, 500, 7, 360]
        x = self.conv(x)
        x = self.sigmoid(x.squeeze())  # [2, 360]

        return x


class Domain_Classifier(nn.Module):
    def __init__(self):
        super(Domain_Classifier, self).__init__()
        self.ResidualBlock = ResidualBlock(128)
        self.conv1 = CNR2d(128, 16, kernel_size=3, stride=(2, 3))
        self.conv2 = CNR2d(16, 2, kernel_size=3, stride=(2, 3))
        self.fc = nn.Linear(12, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x:[2, 128, 7, 54]
        x = self.ResidualBlock(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.sigmoid(x)

        return x


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.feature_encoder = Feature_Encoder()
        self.doa_estimator = DOA_Estimator()
        self.domain_classifier = Domain_Classifier()

    def forward(self, input_data, alpha):
        feature = self.feature_encoder(input_data)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        estimate_output = self.doa_estimator(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return estimate_output, domain_output


if __name__ == '__main__':
    x = torch.rand(2, 8, 7, 337)  # [B, C, T, F]
    model = ResNet()
    estimate_output, domain_output = model(x, alpha=0.1)
    print("estimate_output:", estimate_output.size(), "domain_output:", domain_output.size())