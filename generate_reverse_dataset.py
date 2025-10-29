"""
Reverse Payments Dataset Generator
Generates training data for extracting structured metadata from natural language payment descriptions
This is the reverse of the original dataset - it trains the model to convert English to structured data
"""

import json
import random
from datetime import datetime, timedelta
from typing import List, Dict
import os

# Dataset parameters
TRAIN_SIZE = 400
VAL_SIZE = 50
TEST_SIZE = 50

class ReversePaymentsDatasetGenerator:
    def __init__(self):
        # Transaction types
        self.transaction_types = [
            'payment', 'refund', 'chargeback', 'transfer',
            'fee', 'adjustment', 'reversal', 'pre_authorization'
        ]

        # Payment methods
        self.payment_methods = [
            'credit_card', 'debit_card', 'ACH', 'wire_transfer',
            'PayPal', 'cryptocurrency', 'check', 'direct_debit'
        ]

        # Currencies
        self.currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD']

        # Transaction statuses
        self.statuses = [
            'completed', 'pending', 'failed', 'declined',
            'processing', 'under_review', 'cancelled', 'expired'
        ]

        # Decline/failure reasons
        self.failure_reasons = [
            'insufficient_funds', 'fraud_detection', 'expired_card',
            'invalid_card_number', 'card_declined', 'exceeded_limit',
            'authentication_failed', 'network_error'
        ]

        # Chargeback reasons
        self.chargeback_reasons = [
            'product_not_received', 'product_not_as_described',
            'unauthorized_transaction', 'duplicate_charge',
            'service_not_rendered', 'cancelled_subscription'
        ]

        # Company/merchant names
        self.companies = [
            'Acme Corp', 'Global Supplies Inc', 'TechGadgets Store',
            'Premier Services LLC', 'Digital Solutions Co', 'ABC Company',
            'XYZ Vendor', 'Swift Logistics', 'CloudTech Systems',
            'Retail Plus', 'Enterprise Solutions', 'MegaMart'
        ]

        # Fee types
        self.fee_types = [
            'transaction_fee', 'processing_fee', 'wire_transfer_fee',
            'currency_conversion_fee', 'late_payment_fee',
            'international_transfer_fee', 'chargeback_fee'
        ]

    def generate_amount(self, min_amount=10, max_amount=50000):
        """Generate realistic payment amount"""
        return round(random.uniform(min_amount, max_amount), 2)

    def generate_date(self, days_back=180):
        """Generate random date within past N days (or future if negative)"""
        if days_back < 0:
            # Generate future date
            days_ahead = random.randint(0, abs(days_back))
            date = datetime.now() + timedelta(days=days_ahead)
        else:
            # Generate past date
            days_ago = random.randint(0, days_back)
            date = datetime.now() - timedelta(days=days_ago)
        return date.strftime('%Y-%m-%d')

    def generate_processing_time(self):
        """Generate processing time estimate"""
        options = [
            'instant', '1-2_business_days', '3-5_business_days',
            '5-7_business_days', '7-10_business_days', '2-3_weeks'
        ]
        return random.choice(options)

    def create_payment_example(self) -> Dict:
        """Generate a standard payment transaction - REVERSED"""
        amount = self.generate_amount(100, 25000)
        currency = random.choice(self.currencies)
        sender = random.choice(self.companies)
        receiver = random.choice(self.companies)
        method = random.choice(self.payment_methods)
        status = random.choice(['completed', 'pending', 'processing'])
        date = self.generate_date()

        # The structured output (now the target)
        structured_output = f"inform(transaction_type[payment], amount[{amount}], currency[{currency}], sender[{sender}], receiver[{receiver}], status[{status}], method[{method}], date[{date}])"

        # Generate natural language input (now the input)
        if status == 'completed':
            nl_input = f"Your {method.replace('_', ' ')} payment of {currency} {amount:,.2f} to {receiver} was successfully completed on {date}."
        elif status == 'pending':
            nl_input = f"Your payment of {currency} {amount:,.2f} to {receiver} via {method.replace('_', ' ')} is currently pending and should be completed within 1-2 business days."
        else:  # processing
            nl_input = f"We're processing your {currency} {amount:,.2f} payment to {receiver}. The transaction should be completed shortly."

        # Add variations to the natural language
        nl_input = self._add_variation(nl_input)

        return {
            'input': nl_input,  # Natural language description
            'output': structured_output,  # Structured metadata
        }

    def create_refund_example(self) -> Dict:
        """Generate a refund transaction - REVERSED"""
        original_amount = self.generate_amount(200, 10000)
        refund_amount = original_amount if random.random() > 0.3 else round(original_amount * random.uniform(0.3, 0.9), 2)
        currency = random.choice(self.currencies)
        merchant = random.choice(self.companies)
        processing_time = self.generate_processing_time()
        refund_type = 'full' if refund_amount == original_amount else 'partial'

        structured_output = f"inform(transaction_type[refund], original_amount[{original_amount}], refund_amount[{refund_amount}], currency[{currency}], merchant[{merchant}], refund_type[{refund_type}], processing_time[{processing_time}])"

        if refund_type == 'full':
            nl_input = f"Your full refund of {currency} {refund_amount:,.2f} from {merchant} is being processed and should appear in your account within {processing_time.replace('_', ' ')}."
        else:
            nl_input = f"Your partial refund of {currency} {refund_amount:,.2f} (from your original {currency} {original_amount:,.2f} purchase) from {merchant} is being processed and should arrive within {processing_time.replace('_', ' ')}."

        nl_input = self._add_variation(nl_input)

        return {
            'input': nl_input,
            'output': structured_output,
        }

    def create_chargeback_example(self) -> Dict:
        """Generate a chargeback transaction - REVERSED"""
        amount = self.generate_amount(50, 5000)
        currency = random.choice(self.currencies)
        merchant = random.choice(self.companies)
        reason = random.choice(self.chargeback_reasons)
        status = random.choice(['under_review', 'approved', 'denied', 'pending_merchant_response'])

        structured_output = f"alert(transaction_type[chargeback], amount[{amount}], currency[{currency}], merchant[{merchant}], reason[{reason}], status[{status}])"

        reason_text = reason.replace('_', ' ')

        if status == 'under_review':
            nl_input = f"We've received your chargeback request for {currency} {amount:,.2f} from {merchant} regarding {reason_text}. The case is currently under review and should be resolved within 5-7 business days."
        elif status == 'approved':
            nl_input = f"Your chargeback claim for {currency} {amount:,.2f} from {merchant} has been approved. The funds will be returned to your account within 3-5 business days."
        elif status == 'denied':
            nl_input = f"After review, your chargeback request for {currency} {amount:,.2f} from {merchant} has been denied. The merchant provided sufficient evidence that the transaction was valid."
        else:  # pending_merchant_response
            nl_input = f"Your chargeback request for {currency} {amount:,.2f} from {merchant} is awaiting a response from the merchant. We'll update you within 7 business days."

        nl_input = self._add_variation(nl_input)

        return {
            'input': nl_input,
            'output': structured_output,
        }

    def create_failed_payment_example(self) -> Dict:
        """Generate a failed/declined payment - REVERSED"""
        amount = self.generate_amount(50, 15000)
        currency = random.choice(self.currencies)
        receiver = random.choice(self.companies)
        method = random.choice(self.payment_methods)
        reason = random.choice(self.failure_reasons)

        structured_output = f"alert(transaction_type[payment], amount[{amount}], currency[{currency}], receiver[{receiver}], method[{method}], status[failed], reason[{reason}])"

        reason_text = reason.replace('_', ' ')

        nl_input = f"Your payment of {currency} {amount:,.2f} to {receiver} via {method.replace('_', ' ')} was declined due to {reason_text}. Please verify your payment information and try again."

        nl_input = self._add_variation(nl_input)

        return {
            'input': nl_input,
            'output': structured_output,
        }

    def create_fee_example(self) -> Dict:
        """Generate a fee transaction - REVERSED"""
        base_amount = self.generate_amount(1000, 50000)
        fee_amount = round(base_amount * random.uniform(0.01, 0.05), 2)
        currency = random.choice(self.currencies)
        fee_type = random.choice(self.fee_types)
        date = self.generate_date()

        structured_output = f"inform(transaction_type[fee], fee_type[{fee_type}], fee_amount[{fee_amount}], base_amount[{base_amount}], currency[{currency}], date[{date}])"

        fee_name = fee_type.replace('_', ' ')
        percentage = (fee_amount / base_amount) * 100

        nl_input = f"A {fee_name} of {currency} {fee_amount:,.2f} ({percentage:.2f}%) has been applied to your transaction of {currency} {base_amount:,.2f} on {date}."

        nl_input = self._add_variation(nl_input)

        return {
            'input': nl_input,
            'output': structured_output,
        }

    def create_international_transfer_example(self) -> Dict:
        """Generate an international transfer with currency conversion - REVERSED"""
        amount_sent = self.generate_amount(1000, 100000)
        currency_from = random.choice(self.currencies)
        currency_to = random.choice([c for c in self.currencies if c != currency_from])
        exchange_rate = round(random.uniform(0.5, 2.0), 4)
        amount_received = round(amount_sent * exchange_rate, 2)
        fee = round(amount_sent * 0.02, 2)
        sender = random.choice(self.companies)
        receiver = random.choice(self.companies)
        processing_time = self.generate_processing_time()

        structured_output = f"inform(transaction_type[international_transfer], amount_sent[{amount_sent}], currency_from[{currency_from}], amount_received[{amount_received}], currency_to[{currency_to}], exchange_rate[{exchange_rate}], fee[{fee}], sender[{sender}], receiver[{receiver}], processing_time[{processing_time}])"

        nl_input = f"Your international transfer of {currency_from} {amount_sent:,.2f} to {receiver} will be converted to {currency_to} {amount_received:,.2f} at an exchange rate of {exchange_rate}. A transfer fee of {currency_from} {fee:,.2f} applies. The transfer should complete within {processing_time.replace('_', ' ')}."

        nl_input = self._add_variation(nl_input)

        return {
            'input': nl_input,
            'output': structured_output,
        }

    def create_recurring_payment_example(self) -> Dict:
        """Generate a recurring payment/subscription - REVERSED"""
        amount = self.generate_amount(9.99, 299.99)
        currency = random.choice(self.currencies)
        merchant = random.choice(self.companies)
        frequency = random.choice(['monthly', 'quarterly', 'annually', 'weekly'])
        next_date = self.generate_date(days_back=-30)  # Future date
        status = random.choice(['active', 'cancelled', 'paused', 'payment_failed'])

        structured_output = f"inform(transaction_type[recurring_payment], amount[{amount}], currency[{currency}], merchant[{merchant}], frequency[{frequency}], next_payment_date[{next_date}], status[{status}])"

        if status == 'active':
            nl_input = f"Your {frequency} subscription payment of {currency} {amount:,.2f} to {merchant} is active. Your next payment will be processed on {next_date}."
        elif status == 'cancelled':
            nl_input = f"Your {frequency} subscription payment of {currency} {amount:,.2f} to {merchant} has been cancelled. You will not be charged on {next_date}."
        elif status == 'paused':
            nl_input = f"Your {frequency} subscription to {merchant} ({currency} {amount:,.2f}) is currently paused. No payment will be processed on {next_date}."
        else:  # payment_failed
            nl_input = f"The scheduled {frequency} payment of {currency} {amount:,.2f} to {merchant} failed. Please update your payment method to avoid service interruption."

        nl_input = self._add_variation(nl_input)

        return {
            'input': nl_input,
            'output': structured_output,
        }

    def _add_variation(self, text: str) -> str:
        """Add slight variations to natural language to make the dataset more robust"""
        variations = [
            text,  # Original
            text.replace("Your ", "The "),
            text.replace("We're ", "We are "),
            text.replace("should be", "will be"),
            text.replace("has been", "was"),
        ]
        return random.choice(variations)

    def generate_dataset(self, size: int) -> List[Dict]:
        """Generate a complete dataset with varied transaction types"""
        dataset = []

        # Distribution of transaction types
        type_generators = [
            (self.create_payment_example, 0.30),           # 30% standard payments
            (self.create_refund_example, 0.15),            # 15% refunds
            (self.create_chargeback_example, 0.10),        # 10% chargebacks
            (self.create_failed_payment_example, 0.15),    # 15% failed payments
            (self.create_fee_example, 0.10),               # 10% fees
            (self.create_international_transfer_example, 0.12),  # 12% international
            (self.create_recurring_payment_example, 0.08), # 8% recurring
        ]

        for _ in range(size):
            # Select transaction type based on distribution
            rand = random.random()
            cumulative = 0
            for generator, probability in type_generators:
                cumulative += probability
                if rand <= cumulative:
                    dataset.append(generator())
                    break

        return dataset

    def save_dataset(self, output_dir='./reverse_payments_dataset'):
        """Generate and save train/val/test splits"""
        os.makedirs(output_dir, exist_ok=True)

        print(f"Generating {TRAIN_SIZE} training examples...")
        train_data = self.generate_dataset(TRAIN_SIZE)

        print(f"Generating {VAL_SIZE} validation examples...")
        val_data = self.generate_dataset(VAL_SIZE)

        print(f"Generating {TEST_SIZE} test examples...")
        test_data = self.generate_dataset(TEST_SIZE)

        # Save as JSON
        with open(f'{output_dir}/train.json', 'w') as f:
            json.dump(train_data, f, indent=2)

        with open(f'{output_dir}/validation.json', 'w') as f:
            json.dump(val_data, f, indent=2)

        with open(f'{output_dir}/test.json', 'w') as f:
            json.dump(test_data, f, indent=2)

        print(f"\nDataset saved to {output_dir}/")
        print(f"  - train.json: {len(train_data)} examples")
        print(f"  - validation.json: {len(val_data)} examples")
        print(f"  - test.json: {len(test_data)} examples")
        print(f"\nTotal: {len(train_data) + len(val_data) + len(test_data)} examples")

        # Print sample examples
        print("\n" + "="*80)
        print("SAMPLE EXAMPLES (REVERSE MODEL - English to Structured):")
        print("="*80)
        for i, example in enumerate(train_data[:3], 1):
            print(f"\nExample {i}:")
            print(f"INPUT (Natural Language): {example['input']}")
            print(f"OUTPUT (Structured): {example['output']}")

        return train_data, val_data, test_data


if __name__ == "__main__":
    print("Reverse Payments Dataset Generator")
    print("Converts natural language payment descriptions to structured metadata")
    print("="*80)

    generator = ReversePaymentsDatasetGenerator()
    generator.save_dataset()

    print("\n✓ Reverse dataset generation complete!")
    print("\nNext steps:")
    print("1. Review the generated data in ./reverse_payments_dataset/")
    print("2. Run the fine-tuning script: python finetune_phi3_reverse.py")
