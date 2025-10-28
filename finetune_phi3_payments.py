"""
Phi-3 Fine-tuning for Payments Domain
Optimized for NVIDIA RTX 3060 (12GB VRAM)

This script fine-tunes the Phi-3-Mini model using LoRA on synthetic payments data.
"""

import os
import json
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import numpy as np
from typing import Dict

# Configuration
CONFIG = {
    'base_model': "microsoft/Phi-3-mini-4k-instruct",
    'dataset_dir': './payments_dataset',
    'output_dir': './phi3-payments-finetuned',
    'max_seq_length': 512,
    
    # LoRA parameters
    'lora_r': 16,  # Rank - higher = more parameters but better quality
    'lora_alpha': 32,  # Scaling factor (typically 2x rank)
    'lora_dropout': 0.05,
    'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    
    # Training parameters (optimized for 12GB VRAM)
    'per_device_train_batch_size': 1,  # Small batch size for memory
    'per_device_eval_batch_size': 1,
    'gradient_accumulation_steps': 8,  # Simulates batch size of 8
    'num_train_epochs': 3,
    'learning_rate': 2e-4,
    'warmup_steps': 50,
    'logging_steps': 10,
    'eval_steps': 50,
    'save_steps': 100,
    'max_grad_norm': 0.3,
    'weight_decay': 0.01,
}

def print_gpu_memory():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

def load_model_and_tokenizer():
    """Load Phi-3 model with 8-bit quantization"""
    print("\n" + "="*80)
    print("LOADING MODEL AND TOKENIZER")
    print("="*80)
    
    # Configure 8-bit quantization for reduced memory usage
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_quant_type="nf4",
        bnb_8bit_compute_dtype=torch.float16,
        bnb_8bit_use_double_quant=True,
    )
    
    print(f"\nLoading model: {CONFIG['base_model']}")
    print("Using 8-bit quantization to reduce VRAM usage...")
    
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG['base_model'],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        use_cache=False,  # Disable KV cache for training
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG['base_model'],
        trust_remote_code=True,
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    print(f"✓ Model loaded successfully")
    print_gpu_memory()
    
    return model, tokenizer

def setup_lora(model):
    """Configure LoRA for efficient fine-tuning"""
    print("\n" + "="*80)
    print("CONFIGURING LORA")
    print("="*80)
    
    lora_config = LoraConfig(
        r=CONFIG['lora_r'],
        lora_alpha=CONFIG['lora_alpha'],
        target_modules=CONFIG['target_modules'],
        lora_dropout=CONFIG['lora_dropout'],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    trainable_percent = 100 * trainable_params / all_params
    
    print(f"\nLoRA Configuration:")
    print(f"  Rank (r): {CONFIG['lora_r']}")
    print(f"  Alpha: {CONFIG['lora_alpha']}")
    print(f"  Dropout: {CONFIG['lora_dropout']}")
    print(f"  Target modules: {len(CONFIG['target_modules'])}")
    print(f"\nTrainable Parameters:")
    print(f"  Trainable: {trainable_params:,} ({trainable_percent:.2f}%)")
    print(f"  Total: {all_params:,}")
    print(f"  Memory Savings: ~{100-trainable_percent:.1f}% reduction")
    
    return model

def load_and_prepare_datasets(tokenizer):
    """Load and tokenize the payments dataset"""
    print("\n" + "="*80)
    print("LOADING DATASET")
    print("="*80)
    
    # Load JSON files
    train_path = f"{CONFIG['dataset_dir']}/train.json"
    val_path = f"{CONFIG['dataset_dir']}/validation.json"
    
    print(f"\nLoading training data from: {train_path}")
    with open(train_path, 'r') as f:
        train_data = json.load(f)
    
    print(f"Loading validation data from: {val_path}")
    with open(val_path, 'r') as f:
        val_data = json.load(f)
    
    # Convert to HuggingFace datasets
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    print(f"\n✓ Loaded {len(train_dataset)} training examples")
    print(f"✓ Loaded {len(val_dataset)} validation examples")
    
    # Tokenization function
    def generate_prompt(data_point: Dict) -> str:
        """Format the data into a prompt for the model"""
        return f"""<|system|>
You are a financial services assistant that explains payment transactions in clear, customer-friendly language.<|end|>
<|user|>
Convert the following structured payment information into a natural explanation:

{data_point['meaning_representation']}<|end|>
<|assistant|>
{data_point['target']}<|end|>"""
    
    def tokenize_function(examples):
        """Tokenize the prompts"""
        prompts = [generate_prompt(ex) for ex in examples]
        
        result = tokenizer(
            prompts,
            truncation=True,
            max_length=CONFIG['max_seq_length'],
            padding="max_length",
            return_tensors=None,
        )
        
        # For causal LM, labels are the same as input_ids
        result["labels"] = result["input_ids"].copy()
        
        return result
    
    # Process datasets
    print("\nTokenizing datasets...")
    
    # Need to convert to list of dicts for mapping
    train_dicts = [{"meaning_representation": ex["meaning_representation"], 
                    "target": ex["target"]} 
                   for ex in train_data]
    val_dicts = [{"meaning_representation": ex["meaning_representation"], 
                  "target": ex["target"]} 
                 for ex in val_data]
    
    # Tokenize in batches
    train_tokenized = []
    for ex in train_dicts:
        prompt = generate_prompt(ex)
        tokens = tokenizer(
            prompt,
            truncation=True,
            max_length=CONFIG['max_seq_length'],
            padding="max_length",
        )
        tokens["labels"] = tokens["input_ids"].copy()
        train_tokenized.append(tokens)
    
    val_tokenized = []
    for ex in val_dicts:
        prompt = generate_prompt(ex)
        tokens = tokenizer(
            prompt,
            truncation=True,
            max_length=CONFIG['max_seq_length'],
            padding="max_length",
        )
        tokens["labels"] = tokens["input_ids"].copy()
        val_tokenized.append(tokens)
    
    train_dataset = Dataset.from_list(train_tokenized)
    val_dataset = Dataset.from_list(val_tokenized)
    
    print("✓ Tokenization complete")
    
    # Show sample
    print("\nSample tokenized example:")
    print(f"  Input length: {len(train_tokenized[0]['input_ids'])} tokens")
    sample_text = tokenizer.decode(train_tokenized[0]['input_ids'][:100])
    print(f"  First 100 tokens: {sample_text[:200]}...")
    
    return train_dataset, val_dataset

def setup_training_arguments():
    """Configure training parameters"""
    training_args = TrainingArguments(
        output_dir=CONFIG['output_dir'],
        per_device_train_batch_size=CONFIG['per_device_train_batch_size'],
        per_device_eval_batch_size=CONFIG['per_device_eval_batch_size'],
        gradient_accumulation_steps=CONFIG['gradient_accumulation_steps'],
        num_train_epochs=CONFIG['num_train_epochs'],
        learning_rate=CONFIG['learning_rate'],
        fp16=True,  # Use mixed precision for memory efficiency
        logging_steps=CONFIG['logging_steps'],
        eval_strategy="steps",
        eval_steps=CONFIG['eval_steps'],
        save_steps=CONFIG['save_steps'],
        save_total_limit=3,  # Keep only last 3 checkpoints
        warmup_steps=CONFIG['warmup_steps'],
        max_grad_norm=CONFIG['max_grad_norm'],
        weight_decay=CONFIG['weight_decay'],
        logging_dir=f"{CONFIG['output_dir']}/logs",
        report_to="none",  # Disable wandb/tensorboard for simplicity
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
    )
    
    return training_args

def train_model():
    """Main training pipeline"""
    print("\n" + "="*80)
    print("PHI-3 PAYMENTS FINE-TUNING")
    print("Optimized for RTX 3060 (12GB VRAM)")
    print("="*80)
    
    # Check GPU availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available! This script requires a GPU.")
    
    print(f"\n✓ GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"✓ Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    print_gpu_memory()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Setup LoRA
    model = setup_lora(model)
    
    # Load datasets
    train_dataset, val_dataset = load_and_prepare_datasets(tokenizer)
    
    # Setup training arguments
    training_args = setup_training_arguments()
    
    print("\n" + "="*80)
    print("TRAINING CONFIGURATION")
    print("="*80)
    print(f"\nTraining Parameters:")
    print(f"  Epochs: {CONFIG['num_train_epochs']}")
    print(f"  Batch size: {CONFIG['per_device_train_batch_size']}")
    print(f"  Gradient accumulation: {CONFIG['gradient_accumulation_steps']}")
    print(f"  Effective batch size: {CONFIG['per_device_train_batch_size'] * CONFIG['gradient_accumulation_steps']}")
    print(f"  Learning rate: {CONFIG['learning_rate']}")
    print(f"  Max sequence length: {CONFIG['max_seq_length']}")
    print(f"\nTotal training steps: {len(train_dataset) * CONFIG['num_train_epochs'] // (CONFIG['per_device_train_batch_size'] * CONFIG['gradient_accumulation_steps'])}")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Start training
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    print("\nThis will take approximately 30-60 minutes depending on your GPU...")
    print("Monitor GPU memory usage and training loss below.\n")
    
    try:
        result = trainer.train()
        
        print("\n" + "="*80)
        print("TRAINING COMPLETE!")
        print("="*80)
        print(f"\n✓ Final training loss: {result.training_loss:.4f}")
        
        # Save the final model
        print(f"\nSaving model to: {CONFIG['output_dir']}")
        trainer.save_model()
        tokenizer.save_pretrained(CONFIG['output_dir'])
        
        print("\n✓ Model saved successfully!")
        print(f"\nModel location: {CONFIG['output_dir']}")
        print("\nYou can now use this model with:")
        print("  python test_payments_model.py")
        
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        raise

def main():
    """Entry point"""
    # Check if dataset exists
    if not os.path.exists(CONFIG['dataset_dir']):
        print(f"❌ Dataset not found at {CONFIG['dataset_dir']}")
        print("\nPlease run the dataset generator first:")
        print("  python generate_payments_dataset.py")
        return
    
    # Run training
    train_model()

if __name__ == "__main__":
    main()
