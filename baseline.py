"""
Script d'inférence pour un modèle de langage (LLM) spécialisé en MySQL.

Ce script charge le modèle Qwen-Coder pré-entraîné, gère un historique de
conversation (mémoire à court terme) et intègre un prompt système strict
pour forcer le modèle à agir uniquement comme un assistant de base de données.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Identifiant du modèle sur le Hub Hugging Face
model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

print("Chargement du modèle et du tokenizer...")
# Chargement du tokenizer pour convertir le texte en tokens
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Chargement du modèle avec optimisation de la mémoire (bfloat16) et gestion automatique du matériel (CPU/GPU)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
print("Modèle prêt à être utilisé !")


# Paramètre pour limiter la mémoire du modèle (6 messages = les 3 derniers échanges complets : user + assistant)
MAX_HISTORY_LENGTH = 6
# Liste globale pour conserver le contexte de la conversation en cours
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
    """
    Envoie une requête utilisateur au modèle et retourne sa réponse générée,
    tout en maintenant l'historique de la conversation.

    Args:
        user_input (str): La question ou consigne de l'utilisateur.
        model (AutoModelForCausalLM): Le modèle Hugging Face chargé en mémoire.
        tokenizer (AutoTokenizer): Le tokenizer associé au modèle.
        system_prompt (str, optional): Le comportement dicté au modèle. Par défaut `selected_system_prompt`.

    Returns:
        str: La réponse textuelle générée par l'assistant.
    """
    global chat_history

    # Initialisation de la trame de messages avec le comportement système
    messages = [{"role": "system", "content": system_prompt}]

    # Ajout du contexte précédent (limité pour éviter de saturer le modèle et dépasser le contexte)
    messages.extend(chat_history[-MAX_HISTORY_LENGTH:])

    # Ajout de la nouvelle question de l'utilisateur
    messages.append({"role": "user", "content": user_input})

    # Formatage des messages selon le template de chat spécifique au modèle (ex: ajout des balises <|im_start|>)
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Tokenisation et transfert des données sur le même matériel que le modèle (CPU/GPU)
    inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)

    # Génération de la réponse sans calculer les gradients (économise la mémoire VRAM)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,                  # Limite de longueur de la réponse
            temperature=0.2,                     # Température basse pour des réponses précises et déterministes (idéal pour le code)
            do_sample=True,                      # Active l'échantillonnage
            top_p=0.9,                           # Nucleus sampling pour éviter les mots improbables
            pad_token_id=tokenizer.eos_token_id, # Spécifie le token de fin pour éviter les avertissements (warnings)
        )

    # Extraction uniquement des nouveaux tokens générés (en ignorant ceux du prompt initial)
    generated_tokens = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, outputs)
    ]
    
    # Décodage des tokens en texte lisible pour l'humain
    response = tokenizer.batch_decode(
        generated_tokens, skip_special_tokens=True
    )[0]

    # Mise à jour de la mémoire globale de conversation avec ce nouvel échange
    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": response})

    return response





# --- Phase de Tests ---

# Test 1 : Demande valide (SELECT attendu)
print(
    ask_mysql_assistant(
        "Je veux la liste de tous les utilisateurs inscrits.", model, tokenizer
    )
)

# Test 2 : Demande hors-sujet pour vérifier le refus du prompt système
print(
    ask_mysql_assistant(
        "Peux-tu me donner une recette de crêpes ?", model, tokenizer
    )
)

# Test 3 : Réinitialisation si besoin
# reset_chat()