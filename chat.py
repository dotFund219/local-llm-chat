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

def generate(user_text: str) -> str:
    # it is better to use chatting template on Mistral Instruct model
    messages = [{"role": "user", "content": user_text}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize = False,
        add_generate_propmt = True
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens = 256,
            do_sample = True,
            temperature = 0.7,
            top_p = 0.9,
            pad_token_id = tokenizer.eos_token_id,
        )

    # Decord generated part except prompt
    gen_ids = out[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

print("\nLocal chat ready. Type 'exit' to quit.")
while True:
    user = input("\nYou> ").strip()
    if user.lower() in ["exit", "quit"]:
        break
    ans = generate(user)
    print("\nAssistant>", ans)