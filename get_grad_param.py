import os
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.cuda.amp import autocast, GradScaler

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TextDatasetFromFile(Dataset):
    def __init__(self, file_path, tokenizer, max_length):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.sentences = [line.strip() for line in f.readlines()]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        encoding = self.tokenizer(
            sentence,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze()
        labels = input_ids.clone()
        return {'input_ids': input_ids, 'labels': labels}

def calculate_and_save_grad_mul_param(model, train_dataloader, output_dir, device, accumulation_steps=8):
    model.train()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-6
    )
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()  # Initialize GradScaler
    os.makedirs(output_dir, exist_ok=True)

    total_steps = len(train_dataloader)
    optimizer.zero_grad()  # Zero gradients before accumulating

    for step, batch in enumerate(tqdm(train_dataloader, desc="Training"), 1):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        with autocast():
            outputs = model(input_ids=input_ids)
            logits = outputs.logits
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss = loss / accumulation_steps  # Normalize loss by accumulation steps

        # Accumulate gradients
        scaler.scale(loss).backward()

        if step % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()  # Clear gradients for the next accumulation cycle

            # Clear cache
            torch.cuda.empty_cache()

        if step == total_steps:
            saved_params = []
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if 'mlp' in name or 'attn' in name:
                        grad_mul_param = param.grad * param.data
                        save_path = os.path.join(
                            output_dir,
                            f"grad-mul-param_{name.replace('.', '_')}_step_{step}.pt"
                        )
                        torch.save(grad_mul_param.cpu(), save_path)
                        saved_params.append(name)
                        print(f"Saved grad-mul-param for {name} at step {step} to {save_path}")
                else:
                    print(f"No gradient for {name} at step {step}")
            print(f"Total saved parameters: {len(saved_params)}")


def main():
    # Set random seed
    s = 42
    set_seed(s)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    text_model_name = "chinese-alpaca-plus-7b"
    file_path = "control.txt"
    output_dir = f"grad_mul_param_sent"
    max_length = 32
    batch_size = 8
    
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    model = AutoModelForCausalLM.from_pretrained(text_model_name, torch_dtype=torch.float32)
    
    model.gradient_checkpointing_enable()  
    
    model.to(device)

    for name, param in model.named_parameters():
#        if 'mlp' in name or 'attn' in name:
        param.requires_grad = True
#        else:
#            param.requires_grad = False

    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total = sum(p.numel() for p in model.parameters())
    print(f"Number of trainable parameters: {num_trainable} / {num_total}")

    dataset = TextDatasetFromFile(file_path, tokenizer, max_length)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    calculate_and_save_grad_mul_param(model, train_dataloader, output_dir, device)
    print("Training and parameter saving completed.")

if __name__ == "__main__":
    main()
