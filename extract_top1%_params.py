import torch
import os
import csv
from tqdm import tqdm

def find_top_1_percent_by_layer(grad_param_dir, output_dir, top_percent=0.01, exclude_layers=range(3, 30), exclude_feature_idx=3968):
    """
    For each layer's grad-mul-param tensor, extract the Top 1% of parameters and save them as a boolean mask.
    For the specified layers (exclude_layers), exclude a specific Feature Index (exclude_feature_idx).

    Args:
        grad_param_dir (str): Directory storing grad-mul-param tensors.
        output_dir (str): Directory to save boolean mask files.
        top_percent (float, optional): Top percentage. Default is 0.01 (1%).
        exclude_layers (iterable, optional): Layer indices (0-based) where a specific Feature Index should be excluded. 
                                             Default is layers 4 to 30 (index 3 to 29).
        exclude_feature_idx (int, optional): Feature Index to be excluded. Default is 3968.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Loading and processing grad-mul-param tensors layer by layer...")

    for file_name in tqdm(os.listdir(grad_param_dir), desc="Processing layers"):
        if file_name.endswith(".pt"):
            # Assume the layer index is embedded in the filename, e.g., 'grad-mul-param_layer_3.pt'
            # Extract the layer index
            try:
                # Adjust based on actual filename format
                # Example: avg_grad-mul-param_model_layers_25_self_attn_q_proj_weight.pt
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

            # Flatten the tensor for easier Top K processing
            grad_param_flat = grad_param.flatten()
            
            # Compute the Top 1% count for this layer
            top_k = max(1, int(top_percent * grad_param_flat.numel()))
            
            # Find the Top 1% parameters (absolute values) on GPU
            topk_values, topk_indices = torch.topk(grad_param_flat.abs(), top_k)
            
            # Create a boolean mask and set Top K positions to True on GPU
            bool_mask = torch.zeros_like(grad_param_flat, dtype=torch.bool, device=device)
            bool_mask[topk_indices] = True
            
            # Reshape boolean mask to original tensor shape, move to CPU, and save
            bool_mask = bool_mask.view(grad_param.shape).cpu()
            
            # Modify filename, e.g., 'grad-mul-param_layer_3.pt' -> 'layer_3_bool.pt'
            save_file_name = file_name.replace('grad-mul-param_', '').replace('.pt', '_bool.pt')
            save_path = os.path.join(output_dir, save_file_name)
            torch.save(bool_mask, save_path)
            print(f"Boolean mask for Layer {layer_number} has been saved to {save_path}")

    print("All boolean mask files have been saved.")

# Define directory storing grad*param values
grad_param_dir = "/scratch/ResearchGroups/lt_jixingli/aphasia/model/grad_mul_param/avg"
output_dir = "/scratch/ResearchGroups/lt_jixingli/aphasia/model/boolean_masks_top1_lyr/avg_wo_mass"

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Find the Top 1% parameters in each layer and save boolean masks
find_top_1_percent_by_layer(grad_param_dir, output_dir, top_percent=0.01)
