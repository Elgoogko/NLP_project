import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = "./qwen-sql-lora/best_adapter"

# Charger le tokenizer 
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)

# Charger le modèle de base
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    dtype=torch.bfloat16
)

# Attacher l'adaptateur LoRA par-dessus le modèle de base
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()


def generate_response(user_message, system_message="Tu es un assistant technique expert en MySQL, conçu pour aider les débutants à écrire des requêtes SQL."):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )

    input_length = inputs["input_ids"].shape[-1]
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    return response

# Test
print(generate_response("Donne moi les commandes entre 2024 et 2026"))
print(generate_response("Peux-tu m'écrire un poème sur les chats ?")) 
