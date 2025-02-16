import visualcla
import torch
import os
import csv
import random
import numpy as np
from tqdm import tqdm
import copy
from PIL import Image
from transformers import GenerationConfig

DIR = '/scratch/ResearchGroups/lt_jixingli/aphasia'
os.chdir(DIR)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

image_path = 'analysis/cookie_theft.png'

generation_config = GenerationConfig(
    min_new_tokens=200,    # Minimum generated length
)


def load_model(device):
    print(f"Loading model on device: {device}")

    # Load model, tokenizer, and image processor
    model, tokenizer, image_processor = visualcla.get_model_and_tokenizer_and_processor(
        visualcla_model="visualcla",
        torch_dtype=torch.float16,
        load_in_8bit=True,
        default_device=device
    )
    
    if model is None:
        print("Failed to load model.")
        return None

    # Move model to the specified device
    model.to(device)
    model.eval()
    print(f"Model loaded on device: {device}")

    # Store initial model weights
    original_weights = {name: param.data.clone() for name, param in model.named_parameters()}

    # Create a deep copy of the model
    model_copy = copy.deepcopy(model)

    return model, model_copy, tokenizer, image_processor, original_weights

def reconstruct_param_path(param_name):
    parts = param_name.split('_')
    try:
        layer_num = int(parts[3])
        module_name = parts[4]
        sub_module_name = parts[5]
        param_type = parts[6]

        if module_name == 'mlp':
            param_path = f"text_model.model.layers.{layer_num}.mlp.{sub_module_name}_proj.weight"
        elif module_name == 'self':
            param_path = f"text_model.model.layers.{layer_num}.self_attn.{param_type}_proj.weight"
        else:
            print(f"Unknown module name: {module_name} in file {param_name}")
            return None

        return param_path
    except Exception as e:
        print(f"Error reconstructing param_path from {param_name}: {e}")
        return None

def zero_out_parameters(model, param_path, mask_tensor):
    params_dict = dict(model.named_parameters())
    target_param = params_dict.get(param_path, None)

    if target_param is None:
        print(f"Parameter path {param_path} not found in the model.")
        return False

    if target_param.shape != mask_tensor.shape:
        print(f"Mask shape {mask_tensor.shape} does not match parameter shape {target_param.shape} for {param_path}.")
        return False

    # Apply the mask and set masked positions to zero
    with torch.no_grad():
        target_param.data[mask_tensor] = 0.0

    return True

def zero_out_and_process(avg_folder, output_csv_path, model, model_copy, original_weights):
    
    # Get all .pt files in the folder
    all_pt_files = [f for f in os.listdir(avg_folder) if f.endswith('.pt')]
    
    results = []
    for pt_file in tqdm(all_pt_files, desc="Processing pt files"):
        print(f"Processing parameter: {pt_file}")
        
        # Reconstruct parameter path
        param_path = reconstruct_param_path(pt_file)
        if not param_path:
            print(f"Parameter Path Reconstruction Error for {pt_file}")
            results.append((pt_file, "Parameter Path Reconstruction Error"))
            continue
        
        # Load mask tensor
        mask_path = os.path.join(avg_folder, pt_file)
        try:
            mask_tensor = torch.load(mask_path, map_location='cpu')
            if not isinstance(mask_tensor, torch.Tensor):
                print(f"Mask in {mask_path} is not a tensor.")
                results.append((pt_file, "Invalid Mask Format"))
                continue
            if mask_tensor.dtype != torch.bool:
                mask_tensor = mask_tensor.bool()
        except Exception as e:
            print(f"Error loading mask tensor from {mask_path}: {e}")
            results.append((pt_file, "Mask Loading Error"))
            continue
        
        model = copy.deepcopy(model_copy)
        
        # Apply the mask to zero out the parameters
        success = zero_out_parameters(model, param_path, mask_tensor)
        if not success:
            print(f"Failed to zero out parameters for {param_path}.")
            results.append((pt_file, "Zero-Out Failed"))
            continue
        
        try:
            history = []
            response, history = visualcla.chat(
                model=model, 
                image=image_path, 
                text="请描述这张图片。", 
                history=history, 
                generation_config=generation_config
            )
            print(f"Response for {pt_file}: {response}")
            results.append((pt_file, response))
        except Exception as e:
            print(f"Exception in visualcla.chat for {pt_file}: {e}")
            results.append((pt_file, f"Processing Error: {e}"))
    
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Parameter Name", "Response"])
        for result in results:
            param_name, response = result
            csv_writer.writerow([param_name, response])
    
    print(f"Results saved to {output_csv_path}.")

set_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model, model_copy, tokenizer, image_processor, original_weights = load_model(device)

avg_folder = "bool_mask"
output_csv_path = "model/top_1%/perf/results.csv"

zero_out_and_process(
    avg_folder=avg_folder,
    output_csv_path=output_csv_path,
    model=model,
    model_copy=model_copy,
    original_weights=original_weights
)

print("Processing completed.")
