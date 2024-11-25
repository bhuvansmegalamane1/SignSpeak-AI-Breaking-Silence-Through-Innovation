import torch
import torch.nn as nn
import torch.nn.functional as F

class GestureNet(nn.Module):
    def __init__(self, num_classes=27):  # Default to ASL alphabet + space
        super(GestureNet, self).__init__()
        
        # 3D CNN layers
        self.conv1 = nn.Conv3d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm3d(64)
        self.bn2 = nn.BatchNorm3d(128)
        self.bn3 = nn.BatchNorm3d(256)
        
        # Pooling
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 4 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Input shape: (batch_size, channels, frames, height, width)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class TemporalModule(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super(TemporalModule, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
        
    def forward(self, x):
        # Input shape: (batch_size, sequence_length, input_size)
        output, (hidden, cell) = self.lstm(x)
        return output, hidden

def load_gesture_model(model_path):
    """Load a trained gesture recognition model"""
    model = GestureNet()
    try:
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def preprocess_frame(frame):
    """Preprocess a single frame for the model"""
    # Convert to tensor and normalize
    tensor = torch.from_numpy(frame).float()
    tensor = tensor / 255.0  # Normalize to [0, 1]
    
    # Add batch and channel dimensions if needed
    if len(tensor.shape) == 3:
        tensor = tensor.permute(2, 0, 1)  # Change to (C, H, W)
        tensor = tensor.unsqueeze(0)  # Add batch dimension
    
    return tensor

def get_gesture_prediction(model, frame_tensor):
    """Get prediction for a single frame"""
    with torch.no_grad():
        output = model(frame_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)
    return predicted_class.item(), probabilities[0]
