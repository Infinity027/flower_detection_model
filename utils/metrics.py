import torch
import json


def save_checkpoint(state, filename="weights/my_checkpoint.pth"):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    
def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

def print_gpu_memory():
    allocated = torch.cuda.memory_allocated() / 1024**2  # MB
    cached = torch.cuda.memory_reserved() / 1024**2      # MB
    total = torch.cuda.get_device_properties(0).total_memory / 1024**2
    print(f"GPU Memory: Allocated: {allocated:.2f} MB | Cached: {cached:.2f} MB | Total: {total:.2f} MB")

def get_data_memory(batch, dtype=torch.float32):
    bytes_per_element = torch.tensor([], dtype=dtype).element_size()
    data_mem = batch.element_size() * batch.numel() / (1024**2)  # MB
    return data_mem

def get_model_memory(model):
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size = (param_size + buffer_size) / (1024**2)  # Convert to MB
    return total_size

def cxcywh_to_x1y1x2y2(boxes):
        x1 = boxes[..., 0] - boxes[..., 2]/2
        y1 = boxes[..., 1] - boxes[..., 3]/2
        x2 = boxes[..., 0] + boxes[..., 2]/2
        y2 = boxes[..., 1] + boxes[..., 3]/2
        return torch.stack((x1, y1, x2, y2), dim=-1)

def save_loss_log(result, filename="results/loss_log.json"):
    try:
        with open(filename, "r") as f:
            old_logs = json.load(f)
    except FileNotFoundError:
        old_logs = {key: [] for key in result.keys()}
    
    # Append new values to the old logs
    for key in result.keys():
        old_logs[key].extend(result[key])
    
    with open(filename, "w") as f:
        json.dump(old_logs, f, indent=4)
