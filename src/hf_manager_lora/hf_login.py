import os
from dotenv import load_dotenv
from huggingface_hub import login

def setup_hf_login():
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if token:
        login(token)
    else:
        print(
            "Aucun token de connexion à HuggingFace n'a été fournit / trouvé. Le téléchargement d'un nouveau modèle peut être plus lent ou bloqué en raison des quotas.")
