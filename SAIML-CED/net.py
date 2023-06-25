import copy
import torch.nn as nn
import torch.nn.functional as F
import torch


class RPS_net(nn.Module):

    def __init__(self, num_class):
        super(RPS_net, self).__init__()
        self.final_layers = []
        self.init(num_class)

    def init(self, num_class):
        self.conv = []
        self.conv0 = []
        self.conv.append(self.conv0)
        self.M = 1
        self.L = 9
        self.num_class = num_class

        self.blocks = [2, 2, 2, 2]
        self.stride = [0, 1, 1, 0]
        self.maps = [21, 42, 85, 170]

        self.out_dim = self.maps[-1]

        for i in range(self.L):
            exec("self.conv" + str(i + 1) + " = []")
            exec("self.conv.append(self.conv" + str(i + 1) + ")")

        for i in range(self.M):
            exec("self.m_0_" + str(i) + " = nn.Sequential(nn.Conv2d(3, " + str(
                self.maps[0]) + ", kernel_size=3, stride=1, padding=1, bias=True),nn.BatchNorm2d(" + str(
                self.maps[0]) + "))")
            exec("self.conv[0].append(self.m_0_" + str(i) + ")")

        layer_num = 1
        for a, b in enumerate(self.blocks):

            for l in range(b):

                if (l == 0 and a != 0):
                    sf = self.maps[a - 1]
                else:
                    sf = self.maps[a]
                ef = self.maps[a]

                if (l == b - 1 and self.stride[a] == 1):
                    st = 1
                else:
                    st = 1

                for i in range(self.M):
                    exec("self.m_" + str(layer_num) + "_" + str(i) + " = nn.Sequential(nn.Conv2d(" + str(
                        sf) + ", " + str(ef) + ", kernel_size=3, stride=" + str(
                        st) + ", padding=1, bias=True),nn.BatchNorm2d(" + str(ef) + "),nn.ReLU(), nn.Conv2d(" + str(
                        ef) + ", " + str(ef) + ", kernel_size=3, stride=1, padding=1, bias=True),nn.BatchNorm2d(" + str(
                        ef) + "))")
                    exec("self.conv[" + str(layer_num) + "].append(self.m_" + str(layer_num) + "_" + str(i) + ")")

                if (l == 0 and a != 0):
                    exec("self.m_" + str(layer_num) + "_" + str("x") + " = nn.Sequential(nn.Conv2d(" + str(
                        sf) + ", " + str(ef) + ", kernel_size=1, stride=" + str(
                        st) + ", padding=0),nn.BatchNorm2d(" + str(ef) + "),nn.ReLU())")
                    exec("self.conv[" + str(layer_num) + "].append(self.m_" + str(layer_num) + "_" + str("x") + ")")

                layer_num += 1

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.prelu = nn.PReLU()
        self.relu = nn.ReLU()
        self.fc = nn.Linear(self.maps[-1], self.num_class, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def simple(self, x, conv_list, increse_dim=False, has_skip=True):

        modules = []
        if (not (has_skip)):
            px = conv_list[0](x)
            start = 1
        else:
            px = conv_list[-1](x)
            start = 0
        modules.append(px)
        for j in range(start, self.M):
            px = conv_list[j](x)
            modules.append(px)

        y = torch.stack(modules, dim=0)
        y = y.sum(dim=0)

        if (not (has_skip)):
            x = y + x
        else:
            x = y

        if increse_dim:
            x = self.pool(x)
        return x

    def forward(self, x):

        y = self.conv[0][0](x)
        for j in range(1, self.M):
            if (path[0][j] == 1):
                y += self.conv[0][j](x)
        x = F.relu(y)

        l_num = 1
        for j, b in enumerate(self.blocks):
            for l in range(b):

                if (l == 0 and j != 0):
                    has_skip = True
                else:
                    has_skip = False

                if (l == b - 1 and self.stride[j] == 1):
                    increse_dim = True
                else:
                    increse_dim = False

                x = self.simple(x, self.conv[l_num], increse_dim=increse_dim, has_skip=has_skip)
                l_num += 1
                x = F.relu(x)

        x = F.avg_pool2d(x.clamp(min=1e-6).pow(3), (x.size(-2), x.size(-1))).pow(1. / 3)
        x = x.view(-1, self.maps[-1])
        x1 = self.fc(x)
        x2 = self.fc(x)

        return x2, x1


class BasicNet1(nn.Module):

    def __init__(
        self, args, use_bias=False, init="kaiming", use_multi_fc=False, device=None
    ):
        super(BasicNet1, self).__init__()

        self.use_bias = use_bias
        self.init = init
        self.use_multi_fc = use_multi_fc
        self.args = args

        self.convnet = RPS_net(self.args.num_class)

    
        
        self.classifier = None

        self.n_classes = 0
        self.device = device
        self.cuda()
        
    def forward(self, x):
        x1, x2 = self.convnet(x)
        return x1, x2
        
    def _add_classes_multi_fc(self, n_classes):
        if self.classifier is None:
            self.classifier = []

        new_classifier = self._gen_classifier(n_classes)
        name = "_clf_{}".format(len(self.classifier))
        self.__setattr__(name, new_classifier)
        self.classifier.append(name)

    def _add_classes_single_fc(self, n_classes):
        if self.classifier is not None:
            weight = copy.deepcopy(self.classifier.weight.data)
            if self.use_bias:
                bias = copy.deepcopy(self.classifier.bias.data)

        classifier = self._gen_classifier(self.n_classes + n_classes)

        if self.classifier is not None:
            classifier.weight.data[:self.n_classes] = weight
            if self.use_bias:
                classifier.bias.data[:self.n_classes] = bias

        del self.classifier
        self.classifier = classifier

    def _gen_classifier(self, n_classes):
        classifier = nn.Linear(self.convnet.out_dim, n_classes, bias=self.use_bias).cuda()
        if self.init == "kaiming":
            nn.init.kaiming_normal_(classifier.weight, nonlinearity="linear")
        if self.use_bias:
            nn.init.constant_(classifier.bias, 0.)

        return classifier