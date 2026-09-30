from peft import PeftModel

# Charger ou sauvegarder ton modèle Peft
model.save_pretrained("./my-lora-adapter")

# Publier directement sur le Hub Hugging Face
# Remplace 'votre-username/nom-du-repo' par tes identifiants
model.push_to_hub("votre-username/mon-super-lora", private=False)