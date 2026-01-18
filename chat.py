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

# 1) system prompt
history = [
    {"role": "system", "content": "You are the good AI assistant."}
]

def trim_history_by_turns(history, keep_last_turns=6):
    """
    1 system prompt + N user text history.
    """
    if len(history) <= 1:
        return history
    system = history[:1]
    rest = history[1:]

    trimmed = rest[-(keep_last_turns * 2):]
    return system + trimmed

@torch.inference_mode()
def chat_once(user_text,
              max_new_tokens=128,
              temperature=0.7,
              top_p=0.9):
    global history

    # 2) add the user text on history
    history.append({"role": "user", "content": user_text})

    # 3) set the history memory as 6
    history = trim_history_by_turns(history, keep_last_turns=6)

    # 4) Add the chat template if it has.
    if hasattr(tokenizer, "apply_chat_template"):
        inputs = tokenizer.apply_chat_template(
            history,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)
    else:
        # if there is no template then it makes as simple one.
        prompt = ""
        for m in history:
            prompt += f"{m['role'].upper()}: {m['content']}\n"
        prompt += "ASSISTANT: "
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 5) generation configuration
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        temperature=temperature,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )

    try:
        output = model.generate(inputs, **gen_kwargs)

        # decord
        new_tokens = output[0, inputs.shape[-1]:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        gen_kwargs["max_new_tokens"] = min(64, max_new_tokens)
        gen_kwargs["temperature"] = 0.3
        output = model.generate(inputs, **gen_kwargs)
        new_tokens = output[0, inputs.shape[-1]:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # 7) add assistant on the history
    history.append({"role": "assistant", "content": answer})

    return answer

print("\nLocal chat ready. Type 'exit' to quit.")
while True:
    user = input("\nYou> ").strip()
    if user.lower() in ["exit", "quit"]:
        break
    ans = chat_once(user)
    print("\nAssistant>", ans)