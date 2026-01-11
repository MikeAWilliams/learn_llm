from huggingface_hub import hf_hub_download
import torch
import importlib.util

# Download model code
model_code_path = hf_hub_download(repo_id="mikeawilliams/gpt2", filename="model.py")

# Import the model module
spec = importlib.util.spec_from_file_location("model", model_code_path)
model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_module)

# Download checkpoint
checkpoint_path = hf_hub_download(
    repo_id="mikeawilliams/gpt2",
    filename="model_19072-nq_00149.pt"
)

# Load checkpoint
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

# Create model
model = model_module.GPT(checkpoint['config'])

# Handle state dict with different prefixes
state_dict = checkpoint['model']

# Remove _orig_mod. prefix if present (from torch.compile)
if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
    state_dict = {key.replace("_orig_mod.", ""): value for key, value in state_dict.items()}

# Remove module. prefix if present (from DDP)
if any(key.startswith("module.") for key in state_dict.keys()):
    state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}

# Load the cleaned state dict
model.load_state_dict(state_dict)
model.eval()

print("Model loaded successfully!")
print(f"Config: {checkpoint['config']}")
