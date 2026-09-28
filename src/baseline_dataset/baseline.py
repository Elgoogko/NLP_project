import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

print("Chargement du modèle et du tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
print("Modèle prêt à être utilisé !")


# 6 messages = les 3 derniers échanges
MAX_HISTORY_LENGTH = 6
chat_history = []


# Variante A : Prompt système simple et direct
system_prompt_v1 = (
    "Tu es un assistant MySQL. Ton rôle est de transformer des demandes en "
    "français en requêtes SQL. "
    "Si la demande est hors sujet, refuse de répondre."
)

# Variante B : Prompt système strict et détaillé
system_prompt_v2 = (
    "Tu es un assistant technique expert en MySQL, conçu pour aider les "
    "débutants à écrire des requêtes SQL.\n"
    "Règles strictes :\n"
    "1. Réponds uniquement en français.\n"
    "2. Si la demande concerne les bases de données ou MySQL, donne la requête "
    "SQL dans un bloc de code.\n"
    "3. Si la demande est hors sujet (ex: cuisine, poésie, code d'un autre "
    "langage), refuse poliment et réoriente l'utilisateur vers MySQL.\n"
    "4. Sois concis et précis."
)

# On choisit la Variante B comme meilleure configuration (few-shot/zéro-shot structuré)
selected_system_prompt = system_prompt_v2


def ask_mysql_assistant(
    user_input, model, tokenizer, system_prompt=selected_system_prompt
):
  global chat_history

  messages = [{"role": "system", "content": system_prompt}]

  messages.extend(chat_history[-MAX_HISTORY_LENGTH:])

  messages.append({"role": "user", "content": user_input})

  prompt_text = tokenizer.apply_chat_template(
      messages, tokenize=False, add_generation_prompt=True
  )

  inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)


  with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.2,
        do_sample=True,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )

  generated_tokens = [
      output_ids[len(input_ids) :]
      for input_ids, output_ids in zip(inputs.input_ids, outputs)
  ]
  response = tokenizer.batch_decode(
      generated_tokens, skip_special_tokens=True
  )[0]

  chat_history.append({"role": "user", "content": user_input})
  chat_history.append({"role": "assistant", "content": response})

  return response


# Je ne sais pas si on en a besoin
"""
def reset_chat():
  global chat_history
  chat_history = []
  print("L'historique de conversation a été réinitialisé.")
"""








# Test 1 : Demande valide (SELECT)
print(
    ask_mysql_assistant(
        "Je veux la liste de tous les utilisateurs inscrits.", model, tokenizer
    )
)

# Test 2 : Demande hors-sujet pour vérifier le refus
print(
    ask_mysql_assistant(
        "Peux-tu me donner une recette de crêpes ?", model, tokenizer
    )
)

# Test 3 : Réinitialisation si besoin
# reset_chat()