import sys
import visualcla
import torch
import re
from tqdm import tqdm  # Add progress bar
from transformers import GenerationConfig
import os
import copy

device = 'cuda' if torch.cuda.is_available() else 'cpu'

try:
    model, tokenizer, image_processor = visualcla.get_model_and_tokenizer_and_processor(
        visualcla_model="/scratch/ResearchGroups/lt_jixingli/aphasia/model/visualcla",
        torch_dtype=torch.float16,
        load_in_8bit=True,
        default_device=device
    )
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)


num_layers = 32

image_path = '/scratch/ResearchGroups/lt_jixingli/aphasia/analysis/cookie_theft.png'

results = []

original_weights = {}
for name, param in model.text_model.named_parameters():
    original_weights[name] = param.data.clone()

model_copy = copy.deepcopy(model)

pattern = re.compile(r"text_model\.model\.layers\.(\d+)\..+$")

model.eval()


for layer_index in tqdm(range(num_layers), desc="Disabling layers"):
    try:
        model = copy.deepcopy(model_copy)
        for name, param in model.named_parameters():
            if "embedding" in name:  # Skip embedding layers
                continue

            match = pattern.match(name)
            if match:
                current_layer = int(match.group(1))
                if current_layer == layer_index:
                    param.data.zero_()  # Set all parameters in the layer to zero

        response = visualcla.chat(
            model=model, 
            image=image_path, 
            text="Please describe all the details of this image as thoroughly as possible in about 200 words.", 
            history=[], 
            # generation_config=generation_config
            )

        results.append(response[0])
    except Exception as e:
        print(f"Error processing layer {layer_index + 1}: {e}")
        results.append(None)

output_path = '/scratch/ResearchGroups/lt_jixingli/aphasia/model/disabled_lyr.txt'
output_dir = os.path.dirname(output_path)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

try:
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, response in enumerate(results, 1):
            f.write(f"Disabled_{i}_Layers: {response}\n")
    print("Results have been saved to TXT.")
except Exception as e:
    print(f"Error saving results to TXT: {e}")
