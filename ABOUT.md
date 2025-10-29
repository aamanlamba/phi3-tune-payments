# About Phi-3 Payments Fine-tuning

## Description

Bidirectional fine-tuning of Microsoft's Phi-3-Mini model for payment transaction processing using LoRA (Low-Rank Adaptation). Includes both forward (structured → natural language) and reverse (natural language → structured) models optimized for NVIDIA RTX 3060 (12GB VRAM).

## Key Features

- **Dual Model Approach**: Forward and reverse fine-tuning for comprehensive payment processing
- **Efficient Training**: LoRA-based fine-tuning requires only ~10GB VRAM
- **Synthetic Dataset**: 500 examples covering 8 transaction types
- **Production Ready**: Includes testing, validation, and deployment examples
- **Fast Inference**: 2-3 seconds per transaction on RTX 3060

## Models

### Forward Model (master branch)
Converts structured payment metadata into customer-friendly natural language descriptions.

**Use cases:** Customer notifications, transaction receipts, email alerts

### Reverse Model (reverse-structured-extraction branch)
Extracts structured metadata from natural language payment descriptions.

**Use cases:** Payment APIs, chatbots, email parsing, voice interfaces

## Transaction Types Supported

- Standard Payments (credit card, ACH, wire transfer, etc.)
- Refunds (full and partial)
- Chargebacks and disputes
- Failed/declined payments
- Transaction fees
- International transfers with currency conversion
- Recurring payments and subscriptions
- Pre-authorizations

## Technical Stack

- **Base Model**: microsoft/Phi-3-mini-4k-instruct (3.8B parameters)
- **Fine-tuning**: PEFT with LoRA (rank 16, alpha 32)
- **Framework**: Hugging Face Transformers, PyTorch
- **Quantization**: 8-bit (BitsAndBytes) for training
- **Hardware**: NVIDIA GPU with 12GB+ VRAM

## Quick Stats

- **Training Time**: 30-60 minutes on RTX 3060
- **Model Size**: 7GB base + 8-15MB LoRA adapters
- **Dataset Size**: 500 examples (400 train, 50 val, 50 test)
- **Inference Speed**: 25-30 tokens/second
- **Accuracy**: ~95% on validation set

## License

MIT License - Based on Microsoft's Phi-3 model

## Repository Topics

`machine-learning` `nlp` `fine-tuning` `phi-3` `lora` `payments` `fintech` `transformers` `pytorch` `huggingface` `nvidia` `rtx-3060` `llm` `language-model` `structured-data-extraction` `data-to-text` `text-to-data`
