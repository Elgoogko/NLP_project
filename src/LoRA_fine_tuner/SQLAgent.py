import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

class SQLAgent:
    """
    Cette classe charge un modèle pré-entraîné avec son adaptateur LoRA,
    et gère la génération de réponses en tenant compte d'un historique de
    conversation de taille contrôlée.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        adapter_path: str = "qwen-sql-lora",
        system_message: str = "Tu es un assistant technique expert en MySQL, conçu pour aider les débutants à écrire des requêtes SQL.",
        max_history_turns: int = 4,
    ):
        """
        Charge le tokenizer et le modèle de base, puis attache l'adaptateur LoRA par-dessus.
        :param model_name: Nom complet du modèle de base sur HuggingFace
        :type model_name: str
        :param adapter_path: Chemin local vers l'adaptateur LoRA entraîné
        :type adapter_path: str
        :param system_message: Message système par défaut, définissant le rôle de l'agent
        :type system_message: str
        :param max_history_turns: Nombre de tours (user, assistant) conservés dans
            l'historique envoyé au modèle, pour ne pas dépasser le contexte
            d'entraînement (max_length=512)
        :type max_history_turns: int
        """

        self.adapter_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", adapter_path, "best_adapter"))
        print(self.adapter_path)
        assert os.path.exists(self.adapter_path), f"Path to adapter doesn't exists {self.adapter_path}"

        self.model_name = model_name
        self.system_message = system_message
        self.max_history_turns = max_history_turns

        self.use_cuda = torch.cuda.is_available()

        self.tokenizer = AutoTokenizer.from_pretrained(self.adapter_path)

        base_model = self._load_base_model()
        self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
        self.model.eval()

        # historique du mode CLI
        self.history: list[tuple[str, str]] = []

    def _load_base_model(self):
        """
        Charge le modèle de base en bfloat16
        :return: Le modèle de base chargé
        """
        try:
            return AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto" if self.use_cuda else None,
                dtype=torch.bfloat16,
            )
        except Exception as e:
            print(f"bfloat16 non supporté sur ce matériel ({e}), repli sur float32.")
            return AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto" if self.use_cuda else None,
                dtype=torch.float32,
            )

    def _build_messages(self, user_message: str, external_history: list = None) -> list[dict]:
        """
        Construit la liste structurée des messages (le contexte complet) à envoyer au modèle.
        Cette fonction est flexible : elle s'adapte automatiquement si l'historique provient 
        de notre terminal (CLI) ou d'une interface web externe complexe comme Gradio.

        :param user_message: Le nouveau message tapé par l'utilisateur.
        :type user_message: str
        :param external_history: Historique fourni par une interface graphique externe, le cas échéant.
        :type external_history: list, optionnel
        :return: Une liste de dictionnaires au format OpenAI [{"role": "...", "content": "..."}, ...]
        :rtype: list[dict]
        """
        messages = [{"role": "system", "content": self.system_message}]
        
        user_msg_str = str(user_message)
        
        # Traitement de l'historique venant d'une interface externe (Gradio)
        if external_history and len(external_history) > 0:            
            limit = self.max_history_turns * 2
            for msg in external_history[-limit:]:
                role = msg.get("role", "user") if isinstance(msg, dict) else getattr(msg, "role", "user")
                raw_content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
                
                # Extraction du texte (Gestion du format Multimodal Gradio)
                content_str = ""
                if isinstance(raw_content, list):
                    for item in raw_content:
                        if isinstance(item, dict) and "text" in item:
                            content_str += item["text"] + "\n"
                else:
                    content_str = str(raw_content)
                    
                messages.append({"role": str(role), "content": content_str.strip()})

        # 2. Traitement de l'historique du Terminal (CLI) si Gradio n'est pas utilisé
        elif not external_history and self.history:
            for past_user, past_assistant in self.history[-self.max_history_turns:]:
                messages.append({"role": "user", "content": str(past_user)})
                if past_assistant:
                    messages.append({"role": "assistant", "content": str(past_assistant)})
                    
        # On ajoute le message actuel de l'utilisateur
        messages.append({"role": "user", "content": user_msg_str})
        print(messages)
        
        return messages

    def generate_response(self, user_message: str, external_history: list = None) -> str:
        """
        Génère une réponse SQL à partir de la question de l'utilisateur et du contexte actif.
        Gère la traduction du texte en tokens, l'inférence du modèle sans calcul de gradients, 
        et le décodage du résultat final.

        :param user_message: La question posée par l'utilisateur.
        :type user_message: str
        :param external_history: L'historique d'une interface externe, si applicable.
        :type external_history: list, optionnel
        :return: Le code SQL brut ou l'explication générée par le modèle.
        :rtype: str
        """
        messages = self._build_messages(user_message, external_history)

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        input_length = inputs["input_ids"].shape[-1]
        response = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

        # On met à jour l'historique interne si en mode CLI
        if external_history is None:
            self.history.append((user_message, response))

        return response

    def reset_history(self) -> None:
        """
        Vide l'historique de conversation courant.
        :return: None
        """
        self.history = []
        print("Historique réinitialisé.")

    def run_cli(self) -> None:
        """
        Lance une boucle interactive en ligne de commande pour discuter avec
        l'agent. Tape "reset" pour vider l'historique, "exit" ou "quit" pour
        quitter.
        :return: None
        """
        print(f"--- {self.__class__.__name__} (CLI) ---")
        print(f"Rôle : {self.system_message}")
        print("Commandes : 'reset' pour vider l'historique, 'exit'/'quit' pour quitter.\n")

        while True:
            user_message = input("Vous : ").strip()

            if user_message.lower() in ("exit", "quit"):
                print("Fin de la session.")
                break

            if user_message.lower() == "reset":
                self.reset_history()
                continue

            if not user_message:
                continue

            response = self.generate_response(user_message)
            print(f"Agent : {response}\n")


# Exemple d'utilisation dans le terminal :
if __name__ == "__main__":
    agent = SQLAgent()
    agent.run_cli()
