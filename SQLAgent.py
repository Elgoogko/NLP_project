import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


class SQLAgent:
    """
    Cette classe charge un modèle pré-entraîné avec son adaptateur LoRA,
    et gère la génération de réponses en tenant compte d'un historique de
    conversation de taille contrôlée.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        adapter_path: str = "./qwen-sql-lora/best_adapter",
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
        self.model_name = model_name
        self.adapter_path = adapter_path
        self.system_message = system_message
        self.max_history_turns = max_history_turns

        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)

        base_model = self._load_base_model()
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.eval()

        self.history: list[tuple[str, str]] = []

    def _load_base_model(self):
        """
        Charge le modèle de base en bfloat16
        :return: Le modèle de base chargé
        """
        return AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            dtype=torch.bfloat16,
        )

    def _build_messages(self, user_message: str) -> list[dict]:
        """
        Construit la liste de messages (system + historique tronqué + message courant)
        au format attendu par apply_chat_template.
        :param user_message: Le nouveau message de l'utilisateur
        :type user_message: str
        :return: La liste de messages formatée
        :rtype: list[dict]
        """
        messages = [{"role": "system", "content": self.system_message}]
        for past_user, past_assistant in self.history[-self.max_history_turns:]:
            messages.append({"role": "user", "content": past_user})
            messages.append({"role": "assistant", "content": past_assistant})
        messages.append({"role": "user", "content": user_message})
        return messages

    def generate_response(self, user_message: str) -> str:
        """
        Génère une réponse à partir du message utilisateur et de l'historique
        courant, puis met à jour l'historique avec ce nouvel échange.
        :param user_message: Le message de l'utilisateur
        :type user_message: str
        :return: La réponse générée par le modèle
        :rtype: str
        """
        messages = self._build_messages(user_message)

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


# Exemple d'utilisation :
if __name__ == "__main__":
    agent = SQLAgent()
    agent.run_cli()
