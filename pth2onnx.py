import torch
import torch.onnx

# Define model architecture again
class LinearNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4)  # 3 inputs → 4 outputs

    def forward(self, x):
        return self.linear(x)

# Create model and load trained weights
model = LinearNN()
model.load_state_dict(torch.load("Cluster_Model.pth"))
model.eval()
# Dummy input (e.g., one sample with 3 features: SpO2, HR, Temp)
dummy_input = torch.randn(1, 3)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "Cluster_Model.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},  # Optional for batch flexibility
    opset_version=11
)

print("Model successfully exported as Cluster_Model.onnx")
