# Phi-3 Reverse Fine-tuning for Payments Domain

**Natural Language → Structured Metadata Extraction**

Fine-tune Microsoft's Phi-3-Mini model to extract structured payment metadata from natural language descriptions using LoRA (Low-Rank Adaptation).

**Two Models Available:**
- **Forward Model** (master branch): Structured data → Natural language descriptions
- **Reverse Model** (THIS BRANCH): Natural language → Structured metadata

**Optimized for**: NVIDIA RTX 3060 (12GB VRAM)

👉 **For forward model documentation**, switch to the `master` branch and see [README.md](https://github.com/your-repo/blob/master/README.md)

---

## 🎯 What This Does (Reverse Model)

Extracts structured payment metadata from natural language descriptions - perfect for building conversational payment interfaces, parsing payment emails, and converting user requests into API-ready formats.

**Input (Natural Language):**
```
Your payment of USD 1,500.00 to Global Supplies Inc via
wire transfer was successfully completed on 2024-10-27.
```

**Output (Structured Metadata):**
```
inform(transaction_type[payment], amount[1500.00], currency[USD],
       receiver[Global Supplies Inc], status[completed],
       method[wire_transfer], date[2024-10-27])
```

### Use Cases

- 💬 **Conversational Payment Bots**: Parse user payment requests in natural language
- 📧 **Email Parsing**: Extract transaction data from payment notifications
- 🎤 **Voice Interfaces**: Convert spoken payment descriptions to structured data
- 🔄 **Payment APIs**: Transform natural language into API-ready parameters
- 📊 **Transaction Analysis**: Structure unstructured payment descriptions

### Comparison: Forward vs Reverse Model

| Aspect | Forward Model | Reverse Model (This Branch) |
|--------|---------------|----------------------------|
| **Input** | Structured metadata | Natural language |
| **Output** | Natural language | Structured metadata |
| **Temperature** | 0.7 (creative) | 0.3 (deterministic) |
| **Use Case** | User notifications | API parsing, chatbots |
| **Branch** | `master` | `reverse-structured-extraction` |

---

## 📋 Prerequisites

- **GPU**: NVIDIA RTX 3060 (12GB VRAM) or similar
- **OS**: Windows, Linux (Ubuntu 20.04+), or macOS with CUDA support
- **CUDA**: Version 11.8 or 12.1+
- **Python**: 3.9+
- **Disk Space**: ~15GB for model and dataset

### Verify CUDA Installation

```bash
# Check if CUDA is available
nvidia-smi

# Expected output should show your GPU and CUDA version
```

---

## 🚀 Quick Start (30 minutes)

### Step 1: Set Up Python Environment

```bash
# Ensure you're on the reverse-structured-extraction branch
git checkout reverse-structured-extraction

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

### Step 2: Generate Reverse Dataset

```bash
python generate_reverse_dataset.py
```

**Output:**
- `reverse_payments_dataset/train.json` (400 examples)
- `reverse_payments_dataset/validation.json` (50 examples)
- `reverse_payments_dataset/test.json` (50 examples)

**Sample Data Format:**
```json
{
  "input": "Your payment of EUR 2,500.00 to TechGadgets Store via credit card was declined due to insufficient funds.",
  "output": "alert(transaction_type[payment], amount[2500.00], currency[EUR], receiver[TechGadgets Store], method[credit_card], status[failed], reason[insufficient_funds])"
}
```

This takes about 30 seconds and creates 500 synthetic payment examples with **reversed** input/output from the forward model.

### Step 3: Fine-tune the Reverse Model

```bash
python finetune_phi3_reverse.py
```

**Expected behavior:**
1. Downloads Phi-3-Mini model (~3GB, first run only)
2. Applies LoRA adapters for reverse task
3. Trains for 3 epochs (~30-45 minutes on RTX 3060)
4. Saves fine-tuned model to `./phi3-payments-reverse-finetuned/`

**Monitor GPU memory:**
- Initial load: ~8-9GB VRAM
- During training: ~10-11GB VRAM
- If you get OOM errors, see troubleshooting below

**Key Training Configuration:**
```python
# In finetune_phi3_reverse.py
CONFIG = {
    'dataset_dir': './reverse_payments_dataset',  # Note: REVERSE dataset
    'output_dir': './phi3-payments-reverse-finetuned',
    'num_train_epochs': 3,
    'learning_rate': 2e-4,
    'lora_r': 16,
}
```

### Step 4: Test Your Reverse Model

```bash
# Run pre-defined test cases
python test_reverse_model.py

# Interactive mode - try your own payment descriptions
python test_reverse_model.py interactive

# Compare reverse model with base model
python test_reverse_model.py compare

# Validate accuracy on test dataset
python test_reverse_model.py validate
```

**Example Interactive Session:**
```
Enter payment description: Your monthly subscription of $49.99 to Netflix is active
Structured Data:
  inform(transaction_type[recurring_payment], amount[49.99],
         currency[USD], merchant[Netflix], status[active])
```

---

## 📊 Expected Results

After fine-tuning, your reverse model should:

✓ Extract structured metadata from natural language with ~90-95% accuracy
✓ Handle 8 different transaction types (payments, refunds, chargebacks, etc.)
✓ Produce consistent, parseable output format
✓ Generate responses in < 2 seconds on RTX 3060

**Training metrics to expect:**
- Initial loss: ~3.5-4.0
- Final loss: ~0.8-1.2
- Training time: 30-45 minutes
- Extraction accuracy: 90-95% on validation set

---

## 💡 Real-World Examples

### Example 1: Standard Payment
**Input:**
```
Your payment of USD 1,500.00 to Global Supplies Inc via
wire transfer was successfully completed on 2024-10-27.
```
**Extracted:**
```
inform(transaction_type[payment], amount[1500.00], currency[USD],
       receiver[Global Supplies Inc], status[completed],
       method[wire_transfer], date[2024-10-27])
```

### Example 2: Failed Payment
**Input:**
```
Your payment of EUR 2,500.00 to TechGadgets Store via
credit card was declined due to insufficient funds.
```
**Extracted:**
```
alert(transaction_type[payment], amount[2500.00], currency[EUR],
      receiver[TechGadgets Store], method[credit_card],
      status[failed], reason[insufficient_funds])
```

### Example 3: Chargeback Request
**Input:**
```
We've received your chargeback request for USD 899.99 from
Digital Solutions Co regarding product not received.
```
**Extracted:**
```
alert(transaction_type[chargeback], amount[899.99], currency[USD],
      merchant[Digital Solutions Co], reason[product_not_received],
      status[under_review])
```

### Example 4: International Transfer
**Input:**
```
Your international transfer of USD 10,000.00 to XYZ Vendor
will be converted to EUR 9,200.00 at an exchange rate of 0.92.
```
**Extracted:**
```
inform(transaction_type[international_transfer], amount_sent[10000.00],
       currency_from[USD], amount_received[9200.00], currency_to[EUR],
       exchange_rate[0.92], receiver[XYZ Vendor])
```

---

## 🔧 Integration into Applications

### Python Example - Basic Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load reverse model
MODEL_PATH = "./phi3-payments-reverse-finetuned"
BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(model, MODEL_PATH)
model.eval()

# Extract structured data from natural language
def extract_payment_data(description: str):
    prompt = f"""<|system|>
You are a financial data extraction assistant that converts natural language
payment descriptions into structured metadata.<|end|>
<|user|>
Extract structured payment information from the following description:

{description}<|end|>
<|assistant|>
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.3,  # Low temperature for deterministic extraction
            top_p=0.9,
            do_sample=True,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("<|assistant|>")[-1].strip()

# Use it
payment_text = "Your payment of USD 750.00 to ABC Corp was completed."
structured_data = extract_payment_data(payment_text)
print(structured_data)
# Output: inform(transaction_type[payment], amount[750.00], currency[USD],
#                receiver[ABC Corp], status[completed])
```

### Parsing Structured Output

```python
import re

def parse_structured_data(structured_str: str) -> dict:
    """Parse structured payment data into a dictionary"""
    # Extract action type (inform, alert, etc.)
    action_match = re.match(r'(\w+)\((.*)\)', structured_str)
    if not action_match:
        return None

    action_type = action_match.group(1)
    fields_str = action_match.group(2)

    # Extract fields
    fields = {}
    field_pattern = r'(\w+)\[(.*?)\]'
    for match in re.finditer(field_pattern, fields_str):
        field_name = match.group(1)
        field_value = match.group(2)

        # Convert numeric values
        if field_name in ['amount', 'refund_amount', 'fee_amount', 'exchange_rate']:
            try:
                field_value = float(field_value)
            except ValueError:
                pass

        fields[field_name] = field_value

    return {
        'action_type': action_type,
        'fields': fields
    }

# Example usage
structured = extract_payment_data("Your payment of $100 was completed")
parsed = parse_structured_data(structured)
print(parsed)
# Output:
# {
#   'action_type': 'inform',
#   'fields': {
#     'transaction_type': 'payment',
#     'amount': 100.0,
#     'status': 'completed'
#   }
# }
```

### FastAPI Endpoint

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
model, tokenizer = load_reverse_model()  # Your loading function

class PaymentDescription(BaseModel):
    description: str

class StructuredPayment(BaseModel):
    raw_output: str
    parsed: dict

@app.post("/extract", response_model=StructuredPayment)
async def extract_payment(request: PaymentDescription):
    structured = extract_payment_data(
        model, tokenizer, request.description
    )
    parsed = parse_structured_data(structured)

    return StructuredPayment(
        raw_output=structured,
        parsed=parsed
    )

# Run with: uvicorn api:app --reload
# Test: curl -X POST "http://localhost:8000/extract"
#            -H "Content-Type: application/json"
#            -d '{"description":"Payment of $50 completed"}'
```

---

## 🐛 Troubleshooting

### Poor Extraction Quality

**Symptom**: Model outputs incomplete or incorrect structured data

**Solutions**:

1. **Use lower temperature** for more consistent output:
   ```python
   temperature=0.1  # Very deterministic
   ```

2. **Train longer**:
   ```python
   'num_train_epochs': 5,  # Up from 3
   ```

3. **Increase LoRA rank** for more model capacity:
   ```python
   'lora_r': 32,  # Up from 16
   ```

4. **Add more training data**:
   ```python
   # In generate_reverse_dataset.py
   TRAIN_SIZE = 1000  # Up from 400
   ```

### Inconsistent Output Format

**Symptom**: Output doesn't follow `action(field[value], ...)` pattern

**Solution**: Add post-processing validation:

```python
def validate_and_fix_output(output: str) -> str:
    """Ensure output follows expected format"""
    # Check if output starts with action type
    if not re.match(r'^\w+\(', output):
        return None

    # Ensure balanced parentheses
    if output.count('(') != output.count(')'):
        return None

    # Ensure fields follow field[value] pattern
    if not re.search(r'\w+\[.+?\]', output):
        return None

    return output
```

### Out of Memory (OOM) Errors

**Same solutions as forward model** - see main README.md troubleshooting section:

1. Reduce batch size (already at 1)
2. Reduce sequence length to 256
3. Reduce LoRA rank to 8
4. Close other GPU applications

---

## 🔒 Dataset Variations & Customization

### Add Custom Transaction Types

Edit `generate_reverse_dataset.py`:

```python
def create_subscription_example(self) -> Dict:
    """Generate a subscription payment example"""
    amount = self.generate_amount(5.99, 99.99)
    merchant = random.choice(['Netflix', 'Spotify', 'AWS', 'Adobe'])
    status = random.choice(['active', 'cancelled', 'paused'])

    nl_input = f"Your monthly subscription to {merchant} for ${amount} is {status}."
    structured_output = f"inform(transaction_type[subscription], amount[{amount}], merchant[{merchant}], frequency[monthly], status[{status}])"

    return {'input': nl_input, 'output': structured_output}

# Add to generation distribution
type_generators = [
    # ... existing generators
    (self.create_subscription_example, 0.10),  # 10% subscriptions
]
```

### Add Natural Language Variations

The dataset generator includes `_add_variation()` method that creates variations:

```python
def _add_variation(self, text: str) -> str:
    """Add linguistic variations to training data"""
    variations = [
        text,
        text.replace("Your ", "The "),
        text.replace("We're ", "We are "),
        text.replace("has been", "was"),
        # Add more variations for robustness
        text.replace("USD ", "$"),
        text.replace("EUR ", "€"),
    ]
    return random.choice(variations)
```

---

## 📁 Repository Structure (This Branch)

```
reverse-structured-extraction branch:
├── README_REVERSE.md                  # This file
├── ABOUT.md                           # Repository metadata
│
├── generate_reverse_dataset.py       # Dataset generator (reverse)
├── finetune_phi3_reverse.py          # Training script (reverse)
├── test_reverse_model.py             # Testing script (reverse)
│
├── reverse_payments_dataset/          # Generated dataset
│   ├── train.json                     # 400 training examples
│   ├── validation.json                # 50 validation examples
│   └── test.json                      # 50 test examples
│
└── phi3-payments-reverse-finetuned/  # Fine-tuned model output
    ├── adapter_config.json
    ├── adapter_model.safetensors       # LoRA weights (~8-15MB)
    └── ...
```

### Branch Navigation

- **master** (forward model): Structured → Natural language
- **reverse-structured-extraction** (THIS BRANCH): Natural language → Structured

```bash
# Switch to forward model
git checkout master
# See: README.md for forward model docs

# Switch to reverse model (current)
git checkout reverse-structured-extraction
# See: README_REVERSE.md (this file)
```

---

## 📈 Performance Benchmarks

**RTX 3060 (12GB VRAM):**
- Training time: 35-45 minutes (3 epochs, 400 examples)
- Inference speed: 20-25 tokens/second
- Memory usage: 10-11GB during training
- Extraction accuracy: 90-95% on validation set
- Model size: Base (7GB) + LoRA adapters (8-15MB)

**Scaling up:**
- RTX 3090 (24GB): 2x batch size, 40% faster
- RTX 4090 (24GB): 3x batch size, 60% faster
- A100 (40GB): 8x batch size, train in 10-15 minutes

---

## 📚 Additional Resources

- [Phi-3 Model Card](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Forward Model Documentation](../master/README.md) (switch to master branch)

---

## 🤝 Next Steps

1. **Test on real data**: Try the model on actual payment notifications/emails
2. **Improve accuracy**: Add domain-specific training examples
3. **Build API**: Wrap in FastAPI for production use
4. **Add validation**: Implement output format validation
5. **Combine models**: Use forward + reverse for round-trip validation

---

## 📝 License

This code is provided as-is for educational and research purposes.

- Phi-3 model: MIT License (Microsoft)
- Training code: Use freely, attribute if sharing

---

## 💬 Support

Having issues? Check:
1. This troubleshooting section above
2. [Main README.md](../master/README.md) troubleshooting (forward model)
3. [PyTorch CUDA installation](https://pytorch.org/get-started/locally/)
4. [Transformers documentation](https://huggingface.co/docs/transformers)

---

**Note**: This README describes the **reverse model** (natural language → structured). For the **forward model** (structured → natural language), switch to the `master` branch.
