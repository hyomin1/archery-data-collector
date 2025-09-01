import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from app import config

class CornerRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.fc1   = nn.Linear(128*16*16, 512)
        self.fc2   = nn.Linear(512, 8)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

def load_corner_regressor():
    state = torch.load(config.CORNER_MODEL_PATH, map_location="cuda")
    model = CornerRegressor().to("cuda")
    model.load_state_dict(state)
    model.eval()
    return model



transform = transforms.ToTensor()