"""
Test the fine-tuned Phi-3 Payments Model
Run inference on sample payment transactions
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json

# Configuration
MODEL_PATH = "./phi3-payments-finetuned"
BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"

def load_model():
    """Load the fine-tuned model"""
    print("Loading fine-tuned model...")
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
    )
    
    # Load LoRA weights
    model = PeftModel.from_pretrained(model, MODEL_PATH)
    model.eval()
    
    print("✓ Model loaded successfully\n")
    return model, tokenizer

def generate_response(model, tokenizer, meaning_representation: str, max_new_tokens: int = 150):
    """Generate natural language from payment MR"""
    
    prompt = f"""<|system|>
You are a financial services assistant that explains payment transactions in clear, customer-friendly language.<|end|>
<|user|>
Convert the following structured payment information into a natural explanation:

{meaning_representation}<|end|>
<|assistant|>
"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
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
    """Test with various payment scenarios"""
    
    test_cases = [
        {
            "name": "Standard Payment",
            "mr": "inform(transaction_type[payment], amount[1500.00], currency[USD], sender[Acme Corp], receiver[Global Supplies Inc], status[completed], method[ACH], date[2024-10-27])"
        },
        {
            "name": "Failed Payment",
            "mr": "alert(transaction_type[payment], amount[2500.00], currency[EUR], receiver[TechGadgets Store], method[credit_card], status[failed], reason[insufficient_funds])"
        },
        {
            "name": "Chargeback Request",
            "mr": "alert(transaction_type[chargeback], amount[899.99], currency[USD], merchant[Digital Solutions Co], reason[product_not_received], status[under_review])"
        },
        {
            "name": "Partial Refund",
            "mr": "inform(transaction_type[refund], original_amount[5000.00], refund_amount[1500.00], currency[USD], merchant[Premier Services LLC], refund_type[partial], processing_time[3-5_business_days])"
        },
        {
            "name": "International Transfer",
            "mr": "inform(transaction_type[international_transfer], amount_sent[10000.00], currency_from[USD], amount_received[9200.00], currency_to[EUR], exchange_rate[0.92], fee[150.00], sender[ABC Company], receiver[XYZ Vendor], processing_time[3-5_business_days])"
        },
        {
            "name": "Transaction Fee",
            "mr": "inform(transaction_type[fee], fee_type[wire_transfer_fee], fee_amount[35.00], base_amount[15000.00], currency[USD], date[2024-10-26])"
        },
        {
            "name": "Recurring Payment",
            "mr": "inform(transaction_type[recurring_payment], amount[49.99], currency[USD], merchant[CloudTech Systems], frequency[monthly], next_payment_date[2024-11-15], status[active])"
        },
        {
            "name": "Approved Chargeback",
            "mr": "alert(transaction_type[chargeback], amount[299.99], currency[GBP], merchant[Retail Plus], reason[duplicate_charge], status[approved])"
        },
    ]
    
    print("="*80)
    print("TESTING FINE-TUNED MODEL")
    print("="*80)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"Test {i}: {test_case['name']}")
        print(f"{'='*80}")
        print(f"\nInput (Meaning Representation):")
        print(f"  {test_case['mr']}")
        print(f"\nGenerated Response:")
        
        response = generate_response(model, tokenizer, test_case['mr'])
        print(f"  {response}")
        print()

def interactive_mode(model, tokenizer):
    """Interactive testing mode"""
    print("\n" + "="*80)
    print("INTERACTIVE MODE")
    print("="*80)
    print("\nEnter payment meaning representations to test the model.")
    print("Type 'quit' to exit.\n")
    
    while True:
        print("-" * 80)
        mr_input = input("\nEnter MR (or 'quit'): ").strip()
        
        if mr_input.lower() in ['quit', 'exit', 'q']:
            print("\nExiting interactive mode.")
            break
        
        if not mr_input:
            print("Please enter a valid meaning representation.")
            continue
        
        print(f"\nGenerating response...")
        response = generate_response(model, tokenizer, mr_input)
        print(f"\nResponse:\n  {response}\n")

def compare_with_base_model(model, tokenizer):
    """Compare fine-tuned model with base model"""
    print("\n" + "="*80)
    print("COMPARISON: Fine-tuned vs Base Model")
    print("="*80)
    
    test_mr = "inform(transaction_type[payment], amount[750.00], currency[USD], sender[ABC Company], receiver[XYZ Vendor], status[completed], method[wire_transfer], date[2024-10-20])"
    
    print(f"\nTest Input:\n  {test_mr}\n")
    
    # Fine-tuned model response
    print("Fine-tuned Model Response:")
    response_finetuned = generate_response(model, tokenizer, test_mr)
    print(f"  {response_finetuned}\n")
    
    # Load base model for comparison
    print("Loading base model for comparison...")
    base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    base_model.eval()
    
    print("Base Model Response:")
    response_base = generate_response(base_model, base_tokenizer, test_mr, max_new_tokens=150)
    print(f"  {response_base}\n")
    
    print("\nNote: The fine-tuned model should produce more natural, domain-specific")
    print("payment descriptions compared to the base model.\n")

def main():
    """Main entry point"""
    import sys
    
    # Load model
    try:
        model, tokenizer = load_model()
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        print("\nMake sure you've completed training first:")
        print("  python finetune_phi3_payments.py")
        return
    
    # Parse command line arguments
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"
    
    if mode == "interactive":
        interactive_mode(model, tokenizer)
    elif mode == "compare":
        compare_with_base_model(model, tokenizer)
    else:
        # Run standard test examples
        run_test_examples(model, tokenizer)
        
        # Offer interactive mode
        print("\nWant to try your own examples?")
        choice = input("Enter 'y' for interactive mode, or any other key to exit: ").strip().lower()
        if choice == 'y':
            interactive_mode(model, tokenizer)

if __name__ == "__main__":
    main()
