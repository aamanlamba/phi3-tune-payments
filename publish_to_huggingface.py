"""
Publish Fine-tuned Phi-3 Payments Model to HuggingFace
Uploads LoRA adapters and creates model card
"""

import os
import json
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder
from huggingface_hub import login as hf_login
import shutil

# Configuration
CONFIG = {
    'model_dir': './phi3-payments-finetuned',
    'base_model': 'microsoft/Phi-3-mini-4k-instruct',
    'repo_name': None,  # Will be set by user
    'private': False,   # Set to True if you want a private model
    'license': 'mit',   # Options: mit, apache-2.0, cc-by-4.0, etc.
}

def check_huggingface_token():
    """ load Hugginface token from .env """
    hf_token = os.getenv('HF_TOKEN_WRITE')
    if hf_token:
        try:
            hf_login(token=hf_token, add_to_git_credential=True)
            print("✓ HuggingFace token loaded from .env")
            return True
        except Exception as e:
            print(f"❌ Invalid HuggingFace token in .env: {str(e)}")
            return False
    
    print("❌ HuggingFace token not found")
    return False

def setup_huggingface_token():
    """Guide user through HuggingFace token setup"""
    print("\n" + "="*80)
    print("HUGGINGFACE TOKEN SETUP")
    print("="*80)
    print("\nTo upload models to HuggingFace, you need an access token.")
    print("\nSteps to get your token:")
    print("1. Go to: https://huggingface.co/settings/tokens")
    print("2. Log in or create an account (it's free!)")
    print("3. Click 'New token'")
    print("4. Select 'Write' access (required for uploading)")
    print("5. Copy the token")
    print("\n" + "-"*80)
    
    token = input("\nPaste your HuggingFace token here: ").strip()
    
    if not token:
        print("❌ No token provided. Exiting.")
        return False
    
    try:
        print("\nValidating token...")
        hf_login(token=token, add_to_git_credential=True)
        print("✓ Token validated and saved!")
        return True
    except Exception as e:
        print(f"❌ Token validation failed: {str(e)}")
        return False

def create_model_card(repo_name: str, username: str):
    """Create a comprehensive model card"""
    
    model_card = f"""---
license: {CONFIG['license']}
base_model: {CONFIG['base_model']}
tags:
- phi-3
- lora
- payments
- finance
- natural-language-generation
- finetuned
datasets:
- custom
language:
- en
pipeline_tag: text-generation
library_name: transformers
---

# Phi-3 Mini Fine-tuned for Payments Domain

This is a fine-tuned version of [Microsoft's Phi-3-Mini-4k-Instruct]({CONFIG['base_model']}) model, adapted for generating natural language descriptions of payment transactions using LoRA (Low-Rank Adaptation).

## Model Description

This model converts structured payment transaction data into clear, customer-friendly language. It was fine-tuned using LoRA on a synthetic payments dataset covering various transaction types.

### Training Data

The model was trained on a dataset of 500+ synthetic payment transactions including:
- Standard payments (ACH, wire transfer, credit/debit card)
- Refunds (full and partial)
- Chargebacks
- Failed/declined transactions
- International transfers with currency conversion
- Transaction fees
- Recurring payments/subscriptions

### Example Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = "{CONFIG['base_model']}"
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype="auto",
    device_map="auto"
)

# Load LoRA adapters
model = PeftModel.from_pretrained(model, "{username}/{repo_name}")
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Generate description
prompt = \"\"\"<|system|>
You are a financial services assistant that explains payment transactions in clear, customer-friendly language.<|end|>
<|user|>
Convert the following structured payment information into a natural explanation:

inform(transaction_type[payment], amount[1500.00], currency[USD], sender[Acme Corp], receiver[Global Supplies Inc], status[completed], method[ACH], date[2024-10-27])<|end|>
<|assistant|>
\"\"\"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

Expected output:
```
Your ACH payment of $1,500.00 to Global Supplies Inc was successfully completed on October 27, 2024.
```

## Training Details

### Training Configuration

- **Base Model**: {CONFIG['base_model']}
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **LoRA Rank**: 16
- **LoRA Alpha**: 32
- **Target Modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Quantization**: 8-bit (training), float16 (inference)
- **Training Epochs**: 3
- **Learning Rate**: 2e-4
- **Batch Size**: 1 (with 8 gradient accumulation steps)
- **Hardware**: NVIDIA RTX 3060 (12GB VRAM)
- **Training Time**: ~35-45 minutes

### Training Loss

- Initial Loss: ~3.5-4.0
- Final Loss: ~0.9-1.2
- Validation Loss: ~1.0-1.3

## Model Size

- **LoRA Adapter Size**: ~15MB (only the adapter weights, not the full model)
- **Full Model Size**: ~7GB (when combined with base model)

## Supported Transaction Types

1. **Payments**: Standard payment transactions
2. **Refunds**: Full and partial refunds
3. **Chargebacks**: Dispute and chargeback processing
4. **Failed Payments**: Declined or failed transactions with reasons
5. **International Transfers**: Cross-border payments with currency conversion
6. **Fees**: Transaction and processing fees
7. **Recurring Payments**: Subscriptions and scheduled payments
8. **Reversals**: Payment reversals and adjustments

## Limitations

- Trained on synthetic data - may require additional fine-tuning for production use
- Optimized for English language only
- Best performance on transaction patterns similar to training data
- Not suitable for handling real financial transactions without human oversight
- Should not be used as the sole system for financial communication

## Ethical Considerations

- This model was trained on synthetic, anonymized data only
- Does not contain any real customer PII or transaction data
- Should be validated for accuracy before production deployment
- Implement human review for customer-facing financial communications
- Consider regulatory compliance (PCI-DSS, GDPR, etc.) in your jurisdiction

## Intended Use

**Primary Use Cases:**
- Generating transaction descriptions for internal systems
- Creating customer-friendly payment notifications
- Automating payment communication drafts (with human review)
- Training and demonstration purposes
- Research in financial NLP

**Out of Scope:**
- Direct customer communication without review
- Real-time transaction processing without validation
- Compliance-critical communications
- Medical or legal payment descriptions

## How to Cite

If you use this model in your research or application, please cite:

```bibtex
@misc{{phi3-payments-finetuned,
  author = {{{username}}},
  title = {{Phi-3 Mini Fine-tuned for Payments Domain}},
  year = {{2024}},
  publisher = {{HuggingFace}},
  howpublished = {{\\url{{https://huggingface.co/{username}/{repo_name}}}}}
}}
```

## Training Code

The complete training code and dataset generation scripts are available on GitHub:
- **Repository**: [github.com/aamanlamba/phi3-tune-payments](https://github.com/aamanlamba/phi3-tune-payments)
- **Includes**: Dataset generator, training scripts, testing utilities, and deployment guides

## Acknowledgements

- Base model: [Microsoft Phi-3-Mini-4k-Instruct]({CONFIG['base_model']})
- Fine-tuning method: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- Training framework: HuggingFace Transformers + PEFT
- Inspired by: [NVIDIA AI Workbench Phi-3 Fine-tuning Example](https://github.com/NVIDIA/workbench-example-phi3-finetune)

## License

This model is released under the {CONFIG['license'].upper()} license, compatible with the base Phi-3 model license.

## Contact

For questions or issues, please open an issue on the model repository or contact the author.

---

**Note**: This is a demonstration model. Always validate outputs for accuracy before use in production financial systems.
"""
    
    return model_card

def prepare_model_for_upload(model_dir: str):
    """Prepare model directory for upload"""
    print("\n" + "="*80)
    print("PREPARING MODEL FOR UPLOAD")
    print("="*80)
    
    model_path = Path(model_dir)
    
    if not model_path.exists():
        print(f"❌ Model directory not found: {model_dir}")
        print("Please run training first: python finetune_phi3_payments.py")
        return False
    
    # Check for required files
    required_files = ['adapter_config.json', 'adapter_model.safetensors']
    alternative_files = ['adapter_model.bin']  # Fallback
    
    has_required = all((model_path / f).exists() for f in required_files)
    has_alternative = (model_path / alternative_files[0]).exists()
    
    if not (has_required or has_alternative):
        print(f"❌ Missing required adapter files in {model_dir}")
        print(f"Expected: {required_files} or {alternative_files}")
        return False
    
    print(f"\n✓ Found model files in {model_dir}")
    
    # List files that will be uploaded
    files_to_upload = list(model_path.glob('*'))
    print(f"\nFiles to upload ({len(files_to_upload)}):")
    for f in files_to_upload:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name} ({size_mb:.2f} MB)")
    
    return True

def upload_to_huggingface(repo_name: str, model_dir: str, username: str):
    """Upload model to HuggingFace"""
    print("\n" + "="*80)
    print("UPLOADING TO HUGGINGFACE")
    print("="*80)
    
    # Create HuggingFace API client
    api = HfApi()
    
    # Create repository
    full_repo_name = f"{username}/{repo_name}"
    print(f"\nCreating repository: {full_repo_name}")
    
    try:
        repo_url = create_repo(
            repo_id=full_repo_name,
            repo_type="model",
            private=CONFIG['private'],
            exist_ok=True  # Don't error if repo already exists
        )
        print(f"✓ Repository created/verified: {repo_url}")
    except Exception as e:
        print(f"❌ Failed to create repository: {str(e)}")
        return False
    
    # Create model card
    print("\nGenerating model card...")
    model_card = create_model_card(repo_name, username)
    
    # Save model card
    readme_path = Path(model_dir) / "README.md"
    with open(readme_path, 'w') as f:
        f.write(model_card)
    print("✓ Model card created")
    
    # Upload folder
    print(f"\nUploading files from {model_dir}...")
    print("(This may take a few minutes depending on your connection)")
    
    try:
        api.upload_folder(
            folder_path=model_dir,
            repo_id=full_repo_name,
            repo_type="model",
            commit_message="Upload fine-tuned Phi-3 payments model"
        )
        print("\n✓ Upload complete!")
        
        # Print success message with URL
        model_url = f"https://huggingface.co/{full_repo_name}"
        print("\n" + "="*80)
        print("SUCCESS! 🎉")
        print("="*80)
        print(f"\nYour model is now live at:")
        print(f"  {model_url}")
        print("\nOthers can use it with:")
        print(f"  from peft import PeftModel")
        print(f"  model = PeftModel.from_pretrained(base_model, '{full_repo_name}')")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Upload failed: {str(e)}")
        return False

def get_username():
    """Get HuggingFace username"""
    try:
        api = HfApi()
        user_info = api.whoami()
        return user_info['name']
    except Exception as e:
        print(f"❌ Could not fetch username: {str(e)}")
        return None

def main():
    """Main upload workflow"""
    print("\n" + "="*80)
    print("PUBLISH PHI-3 PAYMENTS MODEL TO HUGGINGFACE")
    print("="*80)
    
    # Step 1: Check/setup HuggingFace token
    if not check_huggingface_token():
        if not setup_huggingface_token():
            return
    
    # Step 2: Get username
    print("\nFetching your HuggingFace username...")
    username = get_username()
    
    if not username:
        username = input("\nEnter your HuggingFace username manually: ").strip()
        if not username:
            print("❌ Username required. Exiting.")
            return
    
    print(f"✓ Username: {username}")
    
    # Step 3: Get repository name
    print("\n" + "-"*80)
    print("Choose a name for your model repository")
    print("-"*80)
    print("\nGuidelines:")
    print("  - Use lowercase with hyphens (e.g., phi3-payments-finetune)")
    print("  - Be descriptive but concise")
    print("  - Avoid special characters")
    print("\nExamples:")
    print("  - phi3-payments-finetune")
    print("  - phi3-mini-financial-transactions")
    print("  - payments-nlg-phi3")
    
    default_name = "phi3-payments-finetune"
    repo_name = input(f"\nRepository name (default: {default_name}): ").strip()
    
    if not repo_name:
        repo_name = default_name
    
    CONFIG['repo_name'] = repo_name
    print(f"\n✓ Repository will be: {username}/{repo_name}")
    
    # Step 4: Privacy setting
    print("\n" + "-"*80)
    private = input("\nMake repository private? (y/n, default: n): ").strip().lower()
    CONFIG['private'] = private == 'y'
    
    print(f"✓ Repository will be: {'Private' if CONFIG['private'] else 'Public'}")
    
    # Step 5: Prepare model
    if not prepare_model_for_upload(CONFIG['model_dir']):
        return
    
    # Step 6: Confirm upload
    print("\n" + "="*80)
    print("READY TO UPLOAD")
    print("="*80)
    print(f"\nRepository: {username}/{repo_name}")
    print(f"Privacy: {'Private' if CONFIG['private'] else 'Public'}")
    print(f"License: {CONFIG['license']}")
    print(f"Source: {CONFIG['model_dir']}")
    
    confirm = input("\nProceed with upload? (y/n): ").strip().lower()
    
    if confirm != 'y':
        print("\n❌ Upload cancelled.")
        return
    
    # Step 7: Upload
    success = upload_to_huggingface(repo_name, CONFIG['model_dir'], username)
    
    if success:
        print("\n" + "="*80)
        print("NEXT STEPS")
        print("="*80)
        print("\n1. Visit your model page to verify everything looks good")
        print(f"   https://huggingface.co/{username}/{repo_name}")
        print("\n2. Edit the model card if needed (click 'Edit model card' button)")
        print("\n3. Share your model with others!")
        print("\n4. Test loading it:")
        print(f"   python test_published_model.py {username}/{repo_name}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Upload cancelled by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
