import torch
import os
import csv
from tqdm import tqdm

def find_top_1_percent_by_layer(grad_param_dir, output_dir, top_percent=0.01, exclude_layers=range(3, 30), exclude_feature_idx=3968):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Loading and processing grad-mul-param tensors layer by layer...")

    for file_name in tqdm(os.listdir(grad_param_dir), desc="Processing layers"):
        if file_name.endswith(".pt")
            try:
                # Adjust based on actual filename format
                parts = file_name.split('_')
                # The layer index is in the fourth part
                layer_str = parts[4]
                layer_number = int(layer_str)
            except ValueError:
                print(f"Unable to parse layer index from filename: {file_name}, skipping this file.")
                continue

            # Load the grad-mul-param tensor for the current layer and move it to GPU
            grad_param = torch.load(os.path.join(grad_param_dir, file_name)).to(device)
            
            # If the current layer needs to exclude a specific Feature Index, set its corresponding row to zero
            if layer_number in exclude_layers:
                if grad_param.dim() == 2:
                    if exclude_feature_idx < grad_param.size(0):
                        grad_param[exclude_feature_idx, :] = 0
                        print(f"Layer {layer_number}: Feature Index {exclude_feature_idx} has been excluded.")
                        # Verify if the exclusion was successful
                        if torch.all(grad_param[exclude_feature_idx, :] == 0):
                            print(f"Verification passed: Feature Index {exclude_feature_idx} in Layer {layer_number} has been successfully set to zero.")
                        else:
                            print(f"Verification failed: Feature Index {exclude_feature_idx} in Layer {layer_number} was not completely set to zero.")
                    else:
                        print(f"Layer {layer_number}: Feature Index {exclude_feature_idx} exceeds tensor row count {grad_param.size(0)}, unable to exclude.")
                else:
                    print(f"Layer {layer_number}: Expected a 2D tensor but received a {grad_param.dim()}D tensor, skipping exclusion.")

            grad_param_flat = grad_param.flatten()
            top_k = max(1, int(top_percent * grad_param_flat.numel()))
            topk_values, topk_indices = torch.topk(grad_param_flat.abs(), top_k)
            bool_mask = torch.zeros_like(grad_param_flat, dtype=torch.bool, device=device)
            bool_mask[topk_indices] = True
            bool_mask = bool_mask.view(grad_param.shape).cpu()
            save_file_name = file_name.replace('grad-mul-param_', '').replace('.pt', '_bool.pt')
            save_path = os.path.join(output_dir, save_file_name)
            torch.save(bool_mask, save_path)
            print(f"Boolean mask for Layer {layer_number} has been saved to {save_path}")

    print("All boolean mask files have been saved.")

grad_param_dir = "grad_mul_param"
output_dir = "bool_mask"


os.makedirs(output_dir, exist_ok=True)
find_top_1_percent_by_layer(grad_param_dir, output_dir, top_percent=0.01)
