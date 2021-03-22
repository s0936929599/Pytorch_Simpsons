class VGGNet(nn.Module):
    def __init__(self, vgg_arch, num_classes):
        super(VGGNet, self).__init__()
        self.features = self.vgg_block(vgg_arch)

        self.fc1 = nn.Linear(in_features=512*7*7, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=1024)
        self.fc3 = nn.Linear(in_features=1024, out_features=num_classes)

    def vgg_block(self, vgg_arch):
        layers = []
        in_channels = 3 

        for v in vgg_arch:
            if v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

            else:
                layers.append(nn.Conv2d(in_channels=in_channels, out_channels=v, kernel_size=3, padding=1))
                layers.append(nn.ReLU(inplace=True)) 
                in_channels = v

        return nn.Sequential(*layers)       

    def forward(self, x):
        x = self.features(x)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    


model = VGGNet([64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],1000)
print(model)
