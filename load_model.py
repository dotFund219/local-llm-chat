import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# AutoTokenize -> Convertor from sentence to token format decimal
# AutoModelForCasualLM -> LLM model like GPT
# BitsAndBytesConfig -> Configure object for 4bit / 8bit

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print("Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

print("Loading model in 4bit")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
)

print("Loaded: " + MODEL_ID)
print("First deivce:", next(model.parameters()).device)
print("Model dtype:", next(model.parameters()).dtype)