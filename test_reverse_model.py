"""
Test the fine-tuned Phi-3 Reverse Payments Model
Run inference to extract structured metadata from natural language payment descriptions
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json

# Configuration
MODEL_PATH = "./phi3-payments-reverse-finetuned"
BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"

def load_model():
    """Load the fine-tuned reverse model"""
    print("Loading fine-tuned reverse model...")
    print(f"Base model: {BASE_MODEL}")
    print(f"LoRA adapters: {MODEL_PATH}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
    )

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",  # Use eager attention to avoid cache issues
    )

    # Load LoRA weights
    model = PeftModel.from_pretrained(model, MODEL_PATH)
    model.eval()

    print("[OK] Model loaded successfully\n")
    return model, tokenizer

def extract_structured_data(model, tokenizer, natural_language: str, max_new_tokens: int = 200):
    """Extract structured metadata from natural language payment description"""

    prompt = f"""<|system|>
You are a financial data extraction assistant that converts natural language payment descriptions into structured metadata that can be processed by payment applications.<|end|>
<|user|>
Extract structured payment information from the following description:

{natural_language}<|end|>
<|assistant|>
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,  # Lower temperature for more deterministic extraction
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False,  # Disable KV cache to avoid API issues
        )

    # Decode and extract only the assistant's response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Extract just the assistant's response
    if "<|assistant|>" in full_response:
        response = full_response.split("<|assistant|>")[-1].split("<|end|>")[0].strip()
    else:
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()

    return response

def run_test_examples(model, tokenizer):
    """Test with various natural language payment descriptions"""

    test_cases = [
        {
            "name": "Standard Payment",
            "description": "Your wire transfer payment of USD 1,500.00 to Global Supplies Inc was successfully completed on 2024-10-27."
        },
        {
            "name": "Failed Payment",
            "description": "Your payment of EUR 2,500.00 to TechGadgets Store via credit card was declined due to insufficient funds. Please verify your payment information and try again."
        },
        {
            "name": "Chargeback Request",
            "description": "We've received your chargeback request for USD 899.99 from Digital Solutions Co regarding product not received. The case is currently under review and should be resolved within 5-7 business days."
        },
        {
            "name": "Partial Refund",
            "description": "Your partial refund of USD 1,500.00 (from your original USD 5,000.00 purchase) from Premier Services LLC is being processed and should arrive within 3-5 business days."
        },
        {
            "name": "International Transfer",
            "description": "Your international transfer of USD 10,000.00 to XYZ Vendor will be converted to EUR 9,200.00 at an exchange rate of 0.92. A transfer fee of USD 150.00 applies. The transfer should complete within 3-5 business days."
        },
        {
            "name": "Transaction Fee",
            "description": "A wire transfer fee of USD 35.00 (0.23%) has been applied to your transaction of USD 15,000.00 on 2024-10-26."
        },
        {
            "name": "Recurring Payment",
            "description": "Your monthly subscription payment of USD 49.99 to CloudTech Systems is active. Your next payment will be processed on 2024-11-15."
        },
        {
            "name": "Approved Chargeback",
            "description": "Your chargeback claim for GBP 299.99 from Retail Plus has been approved. The funds will be returned to your account within 3-5 business days."
        },
        {
            "name": "Pending ACH Payment",
            "description": "Your payment of CAD 3,250.50 to MegaMart via ACH is currently pending and should be completed within 1-2 business days."
        },
        {
            "name": "Cryptocurrency Payment Processing",
            "description": "We're processing your JPY 125,000.00 payment to Swift Logistics via cryptocurrency. The transaction should be completed shortly."
        },
    ]

    print("="*80)
    print("TESTING REVERSE FINE-TUNED MODEL")
    print("Natural Language -> Structured Metadata")
    print("="*80)

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"Test {i}: {test_case['name']}")
        print(f"{'='*80}")
        print(f"\nInput (Natural Language):")
        print(f"  {test_case['description']}")
        print(f"\nExtracted Structured Data:")

        structured = extract_structured_data(model, tokenizer, test_case['description'])
        print(f"  {structured}")
        print()

def interactive_mode(model, tokenizer):
    """Interactive testing mode"""
    print("\n" + "="*80)
    print("INTERACTIVE MODE")
    print("="*80)
    print("\nEnter natural language payment descriptions to extract structured data.")
    print("Type 'quit' to exit.\n")

    while True:
        print("-" * 80)
        nl_input = input("\nEnter payment description (or 'quit'): ").strip()

        if nl_input.lower() in ['quit', 'exit', 'q']:
            print("\nExiting interactive mode.")
            break

        if not nl_input:
            print("Please enter a valid payment description.")
            continue

        print(f"\nExtracting structured data...")
        structured = extract_structured_data(model, tokenizer, nl_input)
        print(f"\nStructured Data:\n  {structured}\n")

def compare_with_base_model(model, tokenizer):
    """Compare fine-tuned model with base model"""
    print("\n" + "="*80)
    print("COMPARISON: Fine-tuned vs Base Model")
    print("="*80)

    test_description = "Your payment of USD 750.00 to XYZ Vendor via wire transfer was successfully completed on 2024-10-20."

    print(f"\nTest Input:\n  {test_description}\n")

    # Fine-tuned model response
    print("Fine-tuned Model Response:")
    response_finetuned = extract_structured_data(model, tokenizer, test_description)
    print(f"  {response_finetuned}\n")

    # Load base model for comparison
    print("Loading base model for comparison...")
    base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",  # Use eager attention to avoid cache issues
    )
    base_model.eval()

    print("Base Model Response:")
    response_base = extract_structured_data(base_model, base_tokenizer, test_description)
    print(f"  {response_base}\n")

    print("\nNote: The fine-tuned model should produce accurate structured metadata")
    print("in the expected format, while the base model may struggle with this task.\n")

def validate_extraction_accuracy(model, tokenizer):
    """Test accuracy on validation dataset samples"""
    print("\n" + "="*80)
    print("VALIDATION ACCURACY TEST")
    print("="*80)

    validation_path = "./reverse_payments_dataset/validation.json"

    try:
        with open(validation_path, 'r') as f:
            val_data = json.load(f)
    except FileNotFoundError:
        print(f"\nValidation dataset not found at {validation_path}")
        print("Please run generate_reverse_dataset.py first.")
        return

    # Test on a sample of validation data
    sample_size = min(10, len(val_data))
    print(f"\nTesting on {sample_size} validation samples...\n")

    for i, example in enumerate(val_data[:sample_size], 1):
        print(f"\n{'='*80}")
        print(f"Validation Example {i}")
        print(f"{'='*80}")
        print(f"\nInput:")
        print(f"  {example['input']}")
        print(f"\nExpected Output:")
        print(f"  {example['output']}")
        print(f"\nModel Output:")

        extracted = extract_structured_data(model, tokenizer, example['input'])
        print(f"  {extracted}")

        # Simple accuracy check (exact match)
        match = extracted.strip() == example['output'].strip()
        print(f"\nExact Match: {'YES' if match else 'NO'}")

def main():
    """Main entry point"""
    import sys

    # Load model
    try:
        model, tokenizer = load_model()
    except Exception as e:
        print(f"[ERROR] Error loading model: {str(e)}")
        print("\nMake sure you've completed training first:")
        print("  python finetune_phi3_reverse.py")
        return

    # Parse command line arguments
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"

    if mode == "interactive":
        interactive_mode(model, tokenizer)
    elif mode == "compare":
        compare_with_base_model(model, tokenizer)
    elif mode == "validate":
        validate_extraction_accuracy(model, tokenizer)
    else:
        # Run standard test examples
        run_test_examples(model, tokenizer)

        # Offer additional modes
        print("\nAdditional test modes:")
        print("  python test_reverse_model.py interactive  - Try your own descriptions")
        print("  python test_reverse_model.py compare      - Compare with base model")
        print("  python test_reverse_model.py validate     - Test on validation dataset")

        print("\nWant to try interactive mode?")
        choice = input("Enter 'y' for interactive mode, or any other key to exit: ").strip().lower()
        if choice == 'y':
            interactive_mode(model, tokenizer)

if __name__ == "__main__":
    main()
