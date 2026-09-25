import os
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

class ObserveModel:
    """
    Cette classe est utilisé pour observer le comportement d'un modèle de langage (LLM) donnée en amont.
    Elle peut être utilisée sur des modèles purs ou sur des agents IA (LLM + LoRA).
    """
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        """
        Initialise la classe avec le nom du modèle sur HuggingFace
        :param model_name: Nom complet du modèle sur HuggingFace
        :type model_name: str
        """
        load_dotenv()
        token = os.getenv("HF_TOKEN")
        if token:
            login(token)

        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def observe_tokenization(self, text: str) -> torch.Tensor:
        """
        Analyse la tokenisation du texte d'entrée.
        :param text: Un texte d'entrée
        :type text: str
        :return: Tokens décodés par le modèle
        """
        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"]

        print("--- Tokenisation ---")
        print(f"Texte d'origine : '{text}'")
        print(f"Input IDs : {input_ids.tolist()[0]}")
        print(
            "Tokens décodés un par un :"
            f" {[self.tokenizer.decode([i]) for i in input_ids[0]]}"
        )

        return input_ids

    def observe_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Extrait et affiche la forme des logits retournés par le modèle.
        :param input_ids: Tokens décodés avec leurs ID
        :type input_ids: torch. Tensor
        :return:
        """
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)

        logits = outputs.logits
        print("\n--- Logits ---")
        print(f"Forme (shape) des logits : {logits.shape}")

        # On retourne les logits du tout dernier token
        return logits[0, -1, :]

    def observe_softmax(
        self, next_token_logits: torch.Tensor, top_k: int = 5
    ) -> torch.Tensor:
        """
        Calcule et affiche la distribution de probabilité (Softmax) du prochain token.
        :param next_token_logits: Les logits calculés des tokens précédents
        :param top_k: Le top K correspond au K tokens les plus probables qu'on conserve. Sur ~128000 on ne garde que les K les plus probables pour la suite
        :type top_k: int
        :return:
        """
        probs = torch.softmax(next_token_logits, dim=-1)

        print("\n--- Distribution Softmax ---")
        top_prob, top_idx = torch.topk(probs, top_k)
        for i in range(top_k):
            token_str = self.tokenizer.decode([top_idx[i]])
            print(
                f"Top {i+1} : '{token_str}' (ID: {top_idx[i].item()}) -"
                f" Probabilité : {top_prob[i].item():.4f}"
            )

        return probs

    def observe_sampling_strategies(
        self,
        next_token_logits: torch.Tensor,
            temperatures: list[float|int]|None=None,
        top_k: int = 10,
    ) -> None:
        """
        Analyse l'effet d'argmax, de la température et du top-k sur la sélection.
        :param next_token_logits: Les logits calculés des tokens précédents
        :type next_token_logits: torch. Tensor
        :param temperatures: Une liste de températures, pour observer l'effet de la température sur le même tirage
        :type temperatures: list[float]
        :param top_k: Le top K correspond au K tokens les plus probables qu'on conserve. Sur ~128000 on ne garde que les K les plus probables pour la suite
        :type top_k: int
        :return:
        """
        if temperatures is None:
            temperatures = [0.2, 0.7, 2.0]
        print("\n--- Stratégies de sélection ---")

        # A. Argmax
        argmax_id = torch.argmax(next_token_logits).item()
        print(f"Argmax -> '{self.tokenizer.decode([argmax_id])}'")

        # B. Température
        for temp in temperatures:
            scaled_logits = next_token_logits / temp
            probs_temp = torch.softmax(scaled_logits, dim=-1)
            top_p, top_i = torch.topk(probs_temp, 3)
            print(
                f"Température {temp} -> Top 1 proba: {top_p[0].item():.4f}"
                f" ('{self.tokenizer.decode([top_i[0]])}')"
            )

        # C. Top-K Sampling
        indices_to_remove = (
            next_token_logits
            < torch.topk(next_token_logits, top_k)[0][..., -1, None]
        )
        filtered_logits = next_token_logits.masked_fill(
            indices_to_remove, -float("Inf")
        )
        probs_top_k = torch.softmax(filtered_logits, dim=-1)
        sampled_top_k_id = torch.multinomial(probs_top_k, num_samples=1).item()
        print(
            f"Échantillonnage Top-K (K={top_k}) -> Token tiré :"
            f" '{self.tokenizer.decode([sampled_top_k_id])}'"
        )

    def observe_top_p(
            self, next_token_logits: torch.Tensor, top_p: float = 0.90, max_tokens_to_show:int = 5
    ) -> None:
        """
        Analyse et applique l'échantillonnage Top-P (Nucleus Sampling).
        :param next_token_logits:
        :type next_token_logits: Torch. Tensor
        :param top_p:
        :type top_p: float
        :param max_tokens_to_show:
        :type max_tokens_to_show: int
        :return:
        """

        # 1. Trier les logits par ordre décroissant
        sorted_logits, sorted_indices = torch.sort(
            next_token_logits, descending=True
        )

        # 2. Calculer les probabilités cumulées
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # 3. Retirer les tokens dont le cumul dépasse P
        # (On décale d'un cran vers la droite pour conserver le premier token qui fait franchir le seuil P)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
            ..., :-1
        ].clone()
        sorted_indices_to_remove[..., 0] = 0

        # 4. Appliquer le masque (–inf) sur les logits filtrés
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        filtered_logits = next_token_logits.clone()
        filtered_logits[indices_to_remove] = -float("Inf")

        # 5. Calculer la nouvelle distribution et tirer un token
        probs_top_p = torch.softmax(filtered_logits, dim=-1)
        sampled_id = torch.multinomial(probs_top_p, num_samples=1).item()

        kept_mask = filtered_logits != -float("Inf")
        kept_indices = sorted_indices[kept_mask[sorted_indices]]

        print(f"\n--- Extrait des mots retenus (Top-P = {top_p}) ---")
        print(
            f"Nombre total de mots conservés : {len(kept_indices)} /"
            f" {len(next_token_logits)}"
        )

        # Afficher les N premiers mots retenus
        nb_to_show = min(max_tokens_to_show, len(kept_indices))
        print(f"Les {nb_to_show} premiers mots du noyau :")

        for i in range(nb_to_show):
            idx = kept_indices[i].item()
            word = self.tokenizer.decode([idx])
            prob = probs_top_p[idx].item()
            print(f"  {i + 1}. '{word}' (ID: {idx}) -> Proba réajustée : {prob:.4f}")

        # Affichage du nombre de tokens conservés
        nb_tokens_conserves = (filtered_logits != -float("Inf")).sum().item()

        print(f"\n--- Échantillonnage Top-P (P={top_p}) ---")
        print(
            f"Nombre de tokens conservés dans le 'noyau' : {nb_tokens_conserves} /"
            f" {len(next_token_logits)}"
        )
        print(f"Token tiré : '{self.tokenizer.decode([sampled_id])}'")

    def run_pipeline(self, text: str) -> None:
        """
        Exécute l'ensemble des observations à la suite.
        :param text: Texte d'entrée
        :return: None
        """
        input_ids = self.observe_tokenization(text)
        next_token_logits = self.observe_logits(input_ids)
        self.observe_softmax(next_token_logits)
        self.observe_sampling_strategies(next_token_logits)
        self.observe_top_p(next_token_logits)


# Exemple d'utilisation :
if __name__ == "__main__":
    observer = ObserveModel()
    observer.run_pipeline("La capitale de la France est")