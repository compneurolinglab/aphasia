import sys
import visualcla
import torch
import re
from tqdm import tqdm  
import os
from transformers import GenerationConfig
import pandas as pd


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model and related components
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

# Set random seed to ensure consistent results
seed = 42
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

num_heads = 32   # Total of 32 heads
head_size = model.text_model.config.hidden_size // model.text_model.config.num_attention_heads

# Image path
image_path = '/scratch/ResearchGroups/lt_jixingli/aphasia/analysis/cookie_theft.png'

# Initialize results list
results = ["" for _ in range(num_heads)]

# Save initial weights
original_weights = {}
for name, param in model.text_model.named_parameters():
    if "q_proj.weight" in name or "k_proj.weight" in name or "v_proj.weight" in name:
        original_weights[name] = param.data.clone()

# Set model to evaluation mode
model.eval()


# Iterate to disable each head
for head_index in tqdm(range(num_heads), desc="Disabling heads"):
    # Empty history for each iteration
    history = []

    # Reset model weights to original
    for name, param in model.text_model.named_parameters():
        if name in original_weights:
            param.data = original_weights[name].clone()

    # Disable a specific head
    for name, param in model.text_model.named_parameters():
        if "q_proj.weight" in name or "k_proj.weight" in name or "v_proj.weight" in name:
            start_index = head_index * head_size
            end_index = start_index + head_size
            param.data[:, start_index:end_index] = 0

    # Generate text response
    try:
        response = visualcla.chat(
            model=model, 
            image=image_path, 
            text="Describe this image.", 
            history=history,
            # generation_config=generation_config
            )


        # Save results
        results[head_index] = response[0]
    except Exception as e:
        print(f"Error processing head {head_index + 1}: {e}")
        results[head_index] = ""


df = pd.DataFrame(results, columns=['response'])
df.to_csv('/scratch/ResearchGroups/lt_jixingli/aphasia/lyr_head/results_head_w_config.csv', index=True, encoding='utf-8')
