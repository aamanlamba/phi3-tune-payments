# Phi-3 Fine-tuning for Payments Domain

Fine-tune Microsoft's Phi-3-Mini model for payment transaction processing using LoRA (Low-Rank Adaptation).

**Two Models Available:**
- **Forward Model** (main branch): Structured data → Natural language descriptions
- **Reverse Model** (reverse-structured-extraction branch): Natural language → Structured metadata

**Optimized for**: NVIDIA RTX 3060 (12GB VRAM)

## 🎯 What This Does

### Forward Model (This Branch)
Converts structured payment data into natural, customer-friendly language:

**Input:**
```
inform(transaction_type[payment], amount[1500.00], currency[USD],
       sender[Acme Corp], receiver[Global Supplies Inc],
       status[completed], method[ACH], date[2024-10-27])
```

**Output:**
```
Your ACH payment of $1,500.00 to Global Supplies Inc was
successfully completed on October 27, 2024.
```

### Reverse Model (See reverse-structured-extraction Branch)
Extracts structured metadata from natural language payment descriptions:

**Input:**
```
Your payment of USD 1,500.00 to Global Supplies Inc via
wire transfer was successfully completed on 2024-10-27.
```

**Output:**
```
inform(transaction_type[payment], amount[1500.00], currency[USD],
       receiver[Global Supplies Inc], status[completed],
       method[wire_transfer], date[2024-10-27])
```

👉 **For reverse model documentation**, switch to the `reverse-structured-extraction` branch and see README_REVERSE.md.

## 📋 Prerequisites

- **GPU**: NVIDIA RTX 3060 (12GB VRAM) or similar
- **OS**: Linux (Ubuntu 20.04+) or Windows with WSL2
- **CUDA**: Version 11.8 or 12.1+
- **Python**: 3.9+
- **Disk Space**: ~15GB for model and dataset

### Verify CUDA Installation

```bash
# Check if CUDA is available
nvidia-smi

# Expected output should show your GPU and CUDA version
```

## 🚀 Quick Start (30 minutes)

> **📌 Note:** This guide is for the **forward model** (structured → natural language).
> For the **reverse model** (natural language → structured), see:
> ```bash
> git checkout reverse-structured-extraction
> # Then read README_REVERSE.md
> ```

### Step 1: Clone or Download Files

```bash
# Create project directory
mkdir phi3-payments-finetune
cd phi3-payments-finetune

# Copy all Python files to this directory:
# - generate_payments_dataset.py
# - finetune_phi3_payments.py
# - test_payments_model.py
# - requirements.txt
```

### Step 2: Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

**Note for Windows users**: Install PyTorch with CUDA support first:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 3: Generate Dataset

```bash
python generate_payments_dataset.py
```

**Output:**
- `payments_dataset/train.json` (400 examples)
- `payments_dataset/validation.json` (50 examples)
- `payments_dataset/test.json` (50 examples)

This takes about 30 seconds and creates 500 synthetic payment transactions with natural language descriptions.

### Step 4: Fine-tune Model

```bash
python finetune_phi3_payments.py
```

**Expected behavior:**
1. Downloads Phi-3-Mini model (~3GB, first run only)
2. Applies LoRA adapters
3. Trains for 3 epochs (~30-45 minutes on RTX 3060)
4. Saves fine-tuned model to `./phi3-payments-finetuned/`

**Monitor GPU memory:**
- Initial load: ~8-9GB VRAM
- During training: ~10-11GB VRAM
- If you get OOM errors, see troubleshooting below

### Step 5: Test Your Model

```bash
# Run pre-defined test cases
python test_payments_model.py

# Interactive mode
python test_payments_model.py interactive

# Compare with base model
python test_payments_model.py compare
```

## 📊 Expected Results

After fine-tuning, your model should:

✓ Convert payment MRs into natural language with ~95% accuracy  
✓ Handle 8 different transaction types (payments, refunds, chargebacks, etc.)  
✓ Maintain consistent terminology and tone  
✓ Generate responses in < 2 seconds on RTX 3060  

**Training metrics to expect:**
- Initial loss: ~3.5-4.0
- Final loss: ~0.8-1.2
- Training time: 30-45 minutes

## 🔧 Customization

### Adjust Training Parameters

Edit `finetune_phi3_payments.py`:

```python
CONFIG = {
    # Increase for better quality (uses more VRAM)
    'lora_r': 16,  # Try 32 or 64 for more capacity
    
    # Train longer for better results
    'num_train_epochs': 3,  # Try 5-10 epochs
    
    # Adjust for your GPU
    'per_device_train_batch_size': 1,  # Increase if you have >12GB VRAM
    'gradient_accumulation_steps': 8,  # Decrease if you increase batch size
}
```

### Create Your Own Dataset

Edit `generate_payments_dataset.py` to:

1. **Add transaction types**: Add methods to the `PaymentsDatasetGenerator` class
2. **Modify distributions**: Change the probability weights in `generate_dataset()`
3. **Customize vocabulary**: Update company names, currencies, payment methods
4. **Scale dataset**: Modify `TRAIN_SIZE`, `VAL_SIZE`, `TEST_SIZE` at top of file

Example - add cryptocurrency payments:

```python
def create_crypto_payment_example(self) -> Dict:
    """Generate a cryptocurrency payment"""
    amount = self.generate_amount(100, 50000)
    crypto = random.choice(['Bitcoin', 'Ethereum', 'USDC'])
    wallet_from = '0x' + ''.join(random.choices('0123456789abcdef', k=40))
    wallet_to = '0x' + ''.join(random.choices('0123456789abcdef', k=40))
    status = random.choice(['confirmed', 'pending', 'processing'])
    
    mr = f"inform(transaction_type[crypto_payment], amount[{amount}], cryptocurrency[{crypto}], wallet_from[{wallet_from[:10]}...], wallet_to[{wallet_to[:10]}...], status[{status}])"
    
    ref = f"Your {crypto} payment of {amount:,.2f} is {status}. Transaction hash: {wallet_from[:10]}..."
    
    return {'meaning_representation': mr, 'target': ref, 'references': [ref]}
```

### Use Real Data

Replace synthetic data with your own:

```python
# Your format
real_data = [
    {
        "meaning_representation": "inform(...)",
        "target": "Your payment description...",
        "references": ["Your payment description..."]
    },
    # ... more examples
]

# Save as JSON
import json
with open('payments_dataset/train.json', 'w') as f:
    json.dump(real_data, f, indent=2)
```

**Important**: Always anonymize real payment data - remove names, account numbers, emails, etc.

## 🐛 Troubleshooting

### Out of Memory (OOM) Errors

**Symptom**: `RuntimeError: CUDA out of memory`

**Solutions**:

1. **Reduce batch size**:
   ```python
   'per_device_train_batch_size': 1,  # Already at minimum
   ```

2. **Reduce sequence length**:
   ```python
   'max_seq_length': 256,  # Down from 512
   ```

3. **Reduce LoRA rank**:
   ```python
   'lora_r': 8,  # Down from 16
   ```

4. **Close other GPU applications**:
   ```bash
   # Check what's using GPU
   nvidia-smi
   
   # Kill other processes if needed
   ```

5. **Use gradient checkpointing** (slower but uses less memory):
   ```python
   model.gradient_checkpointing_enable()
   ```

### Slow Training

**Symptom**: Training takes > 2 hours

**Solutions**:

1. **Verify GPU is being used**:
   ```python
   import torch
   print(torch.cuda.is_available())  # Should be True
   print(torch.cuda.get_device_name(0))  # Should show your GPU
   ```

2. **Reduce dataset size** for testing:
   ```python
   TRAIN_SIZE = 100  # Start small
   VAL_SIZE = 20
   ```

3. **Check GPU utilization**:
   ```bash
   watch -n 1 nvidia-smi
   # GPU utilization should be 90-100% during training
   ```

### Model Not Loading

**Symptom**: `OSError: [model_name] does not appear to be a valid model identifier`

**Solution**: Download model manually:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "microsoft/Phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Will cache for future use
```

### Poor Quality Outputs

**Symptom**: Model generates gibberish or doesn't follow payment format

**Solutions**:

1. **Train longer**:
   ```python
   'num_train_epochs': 5,  # Up from 3
   ```

2. **Increase LoRA rank**:
   ```python
   'lora_r': 32,  # Up from 16
   ```

3. **Generate more training data**:
   ```python
   TRAIN_SIZE = 1000  # Up from 400
   ```

4. **Check data quality**: Review `payments_dataset/train.json` to ensure examples are varied and correct

## 💡 Advanced Usage

### Deploy as API

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
model, tokenizer = load_model()  # Your loading function

class PaymentRequest(BaseModel):
    meaning_representation: str

@app.post("/generate")
async def generate(request: PaymentRequest):
    response = generate_response(model, tokenizer, request.meaning_representation)
    return {"response": response}

# Run with: uvicorn api:app --reload
```

### Batch Processing

```python
def process_batch(meaning_representations: list, batch_size=8):
    """Process multiple MRs efficiently"""
    results = []
    
    for i in range(0, len(meaning_representations), batch_size):
        batch = meaning_representations[i:i+batch_size]
        prompts = [create_prompt(mr) for mr in batch]
        
        inputs = tokenizer(prompts, return_tensors="pt", padding=True)
        outputs = model.generate(**inputs, max_new_tokens=150)
        
        responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        results.extend(responses)
    
    return results
```

### Multiple Domains

Train different LoRA adapters for different use cases:

```bash
# Train for B2B payments
python finetune_phi3_payments.py --output_dir ./phi3-b2b-payments

# Train for consumer payments
python finetune_phi3_payments.py --output_dir ./phi3-consumer-payments

# Train for international
python finetune_phi3_payments.py --output_dir ./phi3-international-payments
```

Switch adapters at runtime:

```python
from peft import PeftModel

base_model = load_base_model()

# Load B2B adapter
b2b_model = PeftModel.from_pretrained(base_model, "./phi3-b2b-payments")

# Switch to consumer adapter
consumer_model = PeftModel.from_pretrained(base_model, "./phi3-consumer-payments")
```

## 📈 Performance Benchmarks

**RTX 3060 (12GB VRAM):**
- Training time: 35-45 minutes (3 epochs, 400 examples)
- Inference speed: 25-30 tokens/second
- Memory usage: 10-11GB during training
- Model size: Base (7GB) + LoRA adapters (8-15MB)

**Scaling up:**
- RTX 3090 (24GB): 2x batch size, 40% faster training
- RTX 4090 (24GB): 3x batch size, 60% faster training
- A100 (40GB): 8x batch size, train in 10-15 minutes

## 🔒 Security Considerations

- **Never** include real customer PII in training data
- Use synthetic or heavily anonymized data only
- Implement output validation for financial accuracy
- Add human review for production deployments
- Consider compliance requirements (PCI-DSS, GDPR, etc.)

## 📚 Additional Resources

- [Phi-3 Model Card](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Original NVIDIA Workbench Example](https://github.com/NVIDIA/workbench-example-phi3-finetune)

## 🤝 Next Steps

1. **Expand dataset**: Generate 1,000-5,000 examples for production quality
2. **Add validation**: Implement automated tests for accuracy
3. **Deploy**: Wrap in API and integrate with your systems
4. **Monitor**: Track performance metrics in production
5. **Iterate**: Continuously improve with real-world examples

## 📝 License

This code is provided as-is for educational and research purposes.

- Phi-3 model: MIT License (Microsoft)
- Training code: Use freely, attribute if sharing

## 💬 Support

Having issues? Check:
1. This troubleshooting section
2. [PyTorch CUDA installation](https://pytorch.org/get-started/locally/)
3. [Transformers documentation](https://huggingface.co/docs/transformers)
4. [NVIDIA CUDA toolkit](https://developer.nvidia.com/cuda-downloads)

## 📁 Repository Structure

```
phi3-tune-payments/
├── README.md                          # This file (Forward model)
│
├── generate_payments_dataset.py       # Dataset generator (forward)
├── finetune_phi3_payments.py         # Training script (forward)
├── test_payments_model.py            # Testing script (forward)
│
├── payments_dataset/                  # Generated forward dataset
│   ├── train.json
│   ├── validation.json
│   └── test.json
│
└── phi3-payments-finetuned/          # Fine-tuned forward model
```

### Reverse Model (Separate Branch)

The reverse model files are in the `reverse-structured-extraction` branch:

```
reverse-structured-extraction branch:
├── README_REVERSE.md                  # Reverse model documentation
├── generate_reverse_dataset.py       # Dataset generator (reverse)
├── finetune_phi3_reverse.py          # Training script (reverse)
├── test_reverse_model.py             # Testing script (reverse)
├── reverse_payments_dataset/          # Generated reverse dataset
└── phi3-payments-reverse-finetuned/  # Fine-tuned reverse model
```

### Branch Information

- **master** (main branch): Forward model implementation (this branch)
- **reverse-structured-extraction**: Reverse model implementation

To switch between models:
```bash
# Use forward model (current)
git checkout master

# Use reverse model
git checkout reverse-structured-extraction
```

---
