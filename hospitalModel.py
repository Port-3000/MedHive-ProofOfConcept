import torch
import flwr as fl
from torch import nn, optim
from torchvision import models

# Define a CNN for medical images (X-rays, CT scans)
class MedicalCNN(nn.Module):
    def __init__(self):
        super(MedicalCNN, self).__init__()
        self.base_model = models.resnet18(pretrained=True)
        self.base_model.fc = nn.Linear(512, 2)  # Binary Classification (Disease / No Disease)
    
    def forward(self, x):
        return self.base_model(x)

# Training function for hospital
def train(model, train_loader, epochs=3):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        for images, labels in train_loader:
            optimizer.zero_grad()
            output = model(images)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()

    return model.state_dict()  # Return model weights

# Federated Learning Client (Hospital)
class HospitalClient(fl.client.NumPyClient):
    def __init__(self, model):
        self.model = model
    
    def get_parameters(self):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]
    
    def fit(self, parameters, config):
        # Load global model weights
        params_dict = zip(self.model.state_dict().keys(), parameters)
        self.model.load_state_dict({k: torch.Tensor(v) for k, v in params_dict})
        
        # Train locally
        self.model.train()
        train(self.model, train_loader)
        
        return self.get_parameters(), len(train_loader), {}

    def evaluate(self, parameters, config):
        # Load global model
        params_dict = zip(self.model.state_dict().keys(), parameters)
        self.model.load_state_dict({k: torch.Tensor(v) for k, v in params_dict})
        
        # Evaluate on test data
        self.model.eval()
        accuracy = 0.85  # Placeholder accuracy
        return accuracy, len(test_loader), {"accuracy": accuracy}

# Run hospital as federated client
model = MedicalCNN()
fl.client.start_numpy_client(server_address="localhost:8080", client=HospitalClient(model))
