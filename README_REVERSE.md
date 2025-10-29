# Phi-3 Reverse Payments Model

**Natural Language to Structured Metadata Extraction**

This model is the reverse of the original Phi-3 Payments model. While the original converts structured payment data into natural language descriptions, this model does the opposite: it extracts structured metadata from natural language payment descriptions.

## Overview

The reverse model is fine-tuned to convert English transaction statements into structured metadata that can be processed by payment applications. This is useful for:

- Parsing user-written payment descriptions
- Extracting structured data from payment emails or notifications
- Converting natural language payment requests into API-ready formats
- Building conversational payment interfaces

## Model Architecture

- **Base Model**: microsoft/Phi-3-mini-4k-instruct
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Task Type**: Causal Language Modeling
- **Training Data**: 500 synthetic payment examples (English → Structured)
- **Hardware Requirements**: NVIDIA GPU with 12GB+ VRAM (optimized for RTX 3060)

## Dataset Format

The reverse dataset uses the following format:

```json
{
  "input": "Your payment of USD 1,500.00 to Global Supplies Inc via wire transfer was successfully completed on 2024-10-27.",
  "output": "inform(transaction_type[payment], amount[1500.00], currency[USD], sender[Acme Corp], receiver[Global Supplies Inc], status[completed], method[wire_transfer], date[2024-10-27])"
}
```

**Key Difference from Original Model:**
- **Input**: Natural language payment description
- **Output**: Structured metadata in a specific format

## Supported Transaction Types

The model can extract metadata for:

1. **Standard Payments**: credit card, ACH, wire transfer, etc.
2. **Refunds**: full and partial refunds
3. **Chargebacks**: various dispute reasons and statuses
4. **Failed Payments**: with specific failure reasons
5. **Transaction Fees**: various fee types
6. **International Transfers**: with currency conversion
7. **Recurring Payments**: subscriptions and scheduled payments

## Quick Start

### 1. Generate the Reverse Dataset

```bash
python generate_reverse_dataset.py
```

This creates the `reverse_payments_dataset/` directory with:
- `train.json` (400 examples)
- `validation.json` (50 examples)
- `test.json` (50 examples)

### 2. Fine-tune the Model

```bash
python finetune_phi3_reverse.py
```

Training parameters (optimized for 12GB VRAM):
- Batch size: 1 (with gradient accumulation of 8)
- Epochs: 3
- Learning rate: 2e-4
- LoRA rank: 16
- Training time: ~30-60 minutes on RTX 3060

The fine-tuned model will be saved to `phi3-payments-reverse-finetuned/`

### 3. Test the Model

```bash
# Run standard test examples
python test_reverse_model.py

# Interactive mode
python test_reverse_model.py interactive

# Compare with base model
python test_reverse_model.py compare

# Validate on test dataset
python test_reverse_model.py validate
```

## Usage Examples

### Example 1: Standard Payment
**Input:**
```
Your payment of USD 1,500.00 to Global Supplies Inc via wire transfer was successfully completed on 2024-10-27.
```

**Extracted Output:**
```
inform(transaction_type[payment], amount[1500.00], currency[USD], receiver[Global Supplies Inc], status[completed], method[wire_transfer], date[2024-10-27])
```

### Example 2: Failed Payment
**Input:**
```
Your payment of EUR 2,500.00 to TechGadgets Store via credit card was declined due to insufficient funds.
```

**Extracted Output:**
```
alert(transaction_type[payment], amount[2500.00], currency[EUR], receiver[TechGadgets Store], method[credit_card], status[failed], reason[insufficient_funds])
```

### Example 3: Chargeback
**Input:**
```
We've received your chargeback request for USD 899.99 from Digital Solutions Co regarding product not received.
```

**Extracted Output:**
```
alert(transaction_type[chargeback], amount[899.99], currency[USD], merchant[Digital Solutions Co], reason[product_not_received], status[under_review])
```

## Integration into Applications

### Python Example

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load model
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

# Extract structured data
def extract_payment_data(description: str):
    prompt = f"""<|system|>
You are a financial data extraction assistant that converts natural language payment descriptions into structured metadata.<|end|>
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
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("<|assistant|>")[-1].strip()

# Use it
payment_text = "Your payment of USD 750.00 to ABC Corp was completed on 2024-10-20."
structured_data = extract_payment_data(payment_text)
print(structured_data)
```

### Parsing the Structured Output

The output format follows this pattern:
```
action_type(field1[value1], field2[value2], ...)
```

You can parse it with a simple regex or custom parser:

```python
import re

def parse_structured_data(structured_str):
    """Parse structured payment data into a dictionary"""
    # Extract action type
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
        fields[field_name] = field_value

    return {
        'action_type': action_type,
        'fields': fields
    }

# Example usage
structured = "inform(transaction_type[payment], amount[1500.00], currency[USD], status[completed])"
parsed = parse_structured_data(structured)
print(parsed)
# Output:
# {
#   'action_type': 'inform',
#   'fields': {
#     'transaction_type': 'payment',
#     'amount': '1500.00',
#     'currency': 'USD',
#     'status': 'completed'
#   }
# }
```

## Performance Considerations

### Memory Usage
- Model loading: ~6-7GB VRAM
- Inference: ~8GB VRAM
- Training: ~10-11GB VRAM

### Inference Speed
- On RTX 3060: ~2-3 seconds per extraction
- Can be improved with:
  - Batched inference
  - Lower precision (8-bit quantization)
  - Smaller max_new_tokens

### Accuracy
The model has been trained on synthetic data, so:
- High accuracy on payment descriptions similar to training data
- May need additional fine-tuning for domain-specific terminology
- Works best with structured, clear payment descriptions

## Comparison with Original Model

| Feature | Original Model | Reverse Model |
|---------|---------------|---------------|
| **Input** | Structured metadata | Natural language |
| **Output** | Natural language | Structured metadata |
| **Use Case** | Explain transactions to users | Parse user descriptions |
| **Temperature** | 0.7 (more creative) | 0.3 (more deterministic) |
| **Application** | Customer notifications | Payment APIs, chatbots |

## Project Structure

```
phi3-tune-payments/
├── reverse_payments_dataset/           # Generated reverse dataset
│   ├── train.json
│   ├── validation.json
│   └── test.json
├── phi3-payments-reverse-finetuned/    # Fine-tuned model
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── ...
├── generate_reverse_dataset.py         # Dataset generator
├── finetune_phi3_reverse.py           # Training script
├── test_reverse_model.py              # Testing/inference script
└── README_REVERSE.md                  # This file
```

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
- Reduce batch size in config
- Enable gradient checkpointing
- Use 8-bit quantization for inference

**2. Poor Extraction Quality**
- Try lower temperature (0.1-0.3)
- Increase training epochs
- Add more domain-specific training data

**3. Inconsistent Format**
- Use temperature=0.3 or lower for more consistent output
- Post-process with validation rules
- Fine-tune with more examples

## Future Improvements

Potential enhancements:
1. **Multi-format Output**: Support JSON, XML, or other structured formats
2. **Confidence Scores**: Add extraction confidence metrics
3. **Error Detection**: Identify ambiguous or incomplete descriptions
4. **Multi-language**: Support non-English payment descriptions
5. **Real Data**: Fine-tune on actual payment notification data

## License

Same license as the base Phi-3 model (MIT License).

## Acknowledgments

- Microsoft for the Phi-3-mini model
- Hugging Face for transformers and PEFT libraries
- The LoRA paper authors for the efficient fine-tuning technique

## Related Files

- Original model documentation: `README.md` (in parent directory)
- Dataset generator (forward): `generate_payments_dataset.py`
- Training script (forward): `finetune_phi3_payments.py`
