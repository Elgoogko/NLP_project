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
        :return: None
        """
        load_dotenv()
        token = os.getenv("HF_TOKEN")
        if token:
            login(token)
        else:
            print("Aucun token de connexion à HuggingFace n'a été fournit / trouvé. Le téléchargement d'un nouveau modèle peut être plus lent ou bloqué en raison des quotas.")

        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def get_emb_token(self, text: str):
        """
        Renvoie l'embedding du premier token trouvé dans la chaine de caractères
        :param text: chaine de caractère, token ou mot
        :type text: str
        :return: embending du premier token (taille variable selon le LLM chargé)
        """
        # 1. Obtenir l'ID du token
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        token_id = token_ids[0]  # On prend le premier token

        # 2. Extraire le vecteur depuis la couche d'embedding
        embedding_layer = self.model.get_input_embeddings()
        token_id_tensor = torch.tensor([token_id])

        with torch.no_grad():
            token_embedding = embedding_layer(token_id_tensor)

        print("Forme du vecteur :", token_embedding.shape)
        print("Vecteur :", token_embedding)


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
        print("\n--- Stratégies de sélection ---")

        # A. Argmax
        self.observe_argmax(next_token_logits)

        # B. Température
        self.observe_temperature(next_token_logits, temperatures)

        # C. Top-K
        self.observe_top_k(next_token_logits, top_k)

        # D. Top-P
        self.observe_top_p(next_token_logits)

    def observe_argmax(self, next_token_logits : torch.Tensor) -> str:
        """
        Prend en entrée un logits et renvoi le token le plus probable (directement décodé)
        :param next_token_logits: Probabilités des tokens
        :return: token le plus probable
        """
        argmax_id = torch.argmax(next_token_logits).item()
        print(f"Argmax -> '{self.tokenizer.decode([argmax_id])}'")
        return self.tokenizer.decode([argmax_id])

    def observe_top_k(self, next_token_logits: torch.Tensor, top_k : int = 10) -> torch.Tensor:
        """
        Permet d'appliqer et d'observer l'effet du top-k (sélection des k plus probables)
        :param next_token_logits: Probabilités des prochains tokens
        :param top_k: nombre de tokens à conserver
        :return: nouvelles probabilités : tous les tokens en dehors du top k sont à -inf
        """
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
        return filtered_logits

    @staticmethod
    def _apply_temperature(next_token_logits: torch.Tensor, temperature:float) -> torch.Tensor:
        """
        Applique la temperature aux probabilités du prochain token. C'est à dire : divise chaque probabilité par la température
        :param next_token_logits: les probabilités du prochain token
        :type next_token_logits: torch.Tensor
        :param temperature: la température, c'est-à-dire la capacité à être créatif ou non
        :type temperature: float
        :return:
        """
        assert temperature > 0, "Le temperature ne peux pas être inférieur ou égale a 0"
        return next_token_logits/temperature

    def observe_temperature(self, next_token_logits: torch.Tensor, temperatures: list[float]|float= 1) -> torch.Tensor | list[torch.Tensor] :
        """
        Applique et montre l'effet de la tempéta
        :param next_token_logits:
        :param temperatures:
        :return:
        """
        if type(temperatures) is float:
            print(f"- Temperature of {temperatures}")
            return self._apply_temperature(next_token_logits, temperatures)
        else:
            scaled_logits_list : list[torch.Tensor] = [computed_temp_logits for computed_temp_logits in
                                  map(self._apply_temperature, next_token_logits, temperatures)]
            for i, scaled_logits in enumerate(scaled_logits_list):
                probs_temp = torch.softmax(scaled_logits, dim=-1)
                top_p, top_i = torch.topk(probs_temp, 3)
                print(
                    f"Température {temperatures[i]} -> Top 1 proba: {top_p[0].item():.4f}"
                    f" ('{self.tokenizer.decode([top_i[0]])}')"
                )
            return scaled_logits_list

    def observe_top_p(
            self, next_token_logits: torch.Tensor, top_p: float = 0.90, max_tokens_to_show:int = 5
    ) -> torch.Tensor:
        """
        Analyse et applique l'échantillonnage Top-P (Nucleus Sampling).
        :param next_token_logits:
        :type next_token_logits: Torch. Tensor
        :param top_p: probabilité totale à atteindre
        :type top_p: float
        :param max_tokens_to_show:
        :type max_tokens_to_show: int
        :return: probabilités des tokens modifiés
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
        return filtered_logits

    def observe_random_token(
            self, next_token_logits: torch.Tensor, num_samples: int = 1
    ) -> str | list[str]:
        """
        Tire un ou plusieurs tokens au hasard en respectant la distribution de probabilité (Softmax)
        Issue des logits transmis.
        :param next_token_logits: Logits (bruts ou filtrés par T/K/P)
        :type next_token_logits: torch.Tensor
        :param num_samples: Nombre de tokens à tirer
        :type num_samples: int
        :return: Le ou les tokens tirés sous forme de chaîne de caractères
        """
        # 1. Conversion des logits filtrés en probabilités
        probs = torch.softmax(next_token_logits, dim=-1)

        # 2. Tirage aléatoire pondéré selon les probabilités
        sampled_indices = torch.multinomial(probs, num_samples=num_samples)

        if num_samples == 1:
            token_id = sampled_indices.item()
            token_str = self.tokenizer.decode([token_id])
            prob_val = probs[token_id].item()
            print(
                f"Tirage aléatoire (Sampling) -> Token tiré : '{token_str}' (ID:"
                f" {token_id}) avec une probabilité de {prob_val:.4f}"
            )
            return token_str
        else:
            tokens_str = []
            print(f"Tirage aléatoire de {num_samples} tokens :")
            for i, idx_tensor in enumerate(sampled_indices):
                idx = idx_tensor.item()
                t_str = self.tokenizer.decode([idx])
                tokens_str.append(t_str)
                print(
                    f"  Tirage {i + 1} : '{t_str}' (ID: {idx}) - Proba :"
                    f" {probs[idx].item():.4f}"
                )
            return tokens_str

    def observe_pipeline_selection(self, next_token_logits: torch.Tensor, temperature: float=0.9, top_k : int=50, top_p: float=0.9, use_argmax:bool = True) -> str:
        """
        Montre le pipeline complet de sélection d'un LLM classique type Qwen2.5
        :param next_token_logits: Les probabilités du prochain token
        :type next_token_logits: torch.Tensor
        :param temperature: Va plus ou moins lisée les probabilités et donc affecter la créativité
        :type temperature: float
        :param top_k: Combien de tokens garder dans les plus probables
        :type top_k: int
        :param top_p: La probabilité nécessaire à atteindre
        :type top_p: float
        :param use_argmax: Utilisé l'argmax pour la sélection final OU une sélection aléatoire
        :type use_argmax: bool
        :return: Prochain token
        """

        print("-- Observation d'une pipeline complète --")

        print("# étape 1")
        next_token_logits_1 = self.observe_temperature(next_token_logits, temperature)

        print("# étape 2")
        next_token_logits_2 = self.observe_top_k(next_token_logits_1, top_k=top_k)

        print("# étape 3")
        next_token_logits_3 = self.observe_top_p(next_token_logits_2, top_p=top_p)

        print("# étape 4")
        next_token_logits_4 = self.observe_softmax(next_token_logits_3)

        result = self.observe_argmax(next_token_logits_4) if use_argmax else self.observe_random_token(next_token_logits_4)

        print(f"# Sélection finale : {self.tokenizer.decode([result])}")

        return self.tokenizer.decode([result])

    def run_pipeline(self, text: str,temperature: float=0.9, top_k : int=50, top_p: float=0.9, use_argmax:bool = True) -> None:
        """
        Exécute l'ensemble des observations à la suite.
        :param use_argmax:
        :param top_p:
        :param top_k:
        :param temperature:
        :param text: Texte d'entrée
        :return: None
        """
        input_ids = self.observe_tokenization(text)
        next_token_logits = self.observe_logits(input_ids)
        self.observe_pipeline_selection(next_token_logits, temperature, top_k, top_p, use_argmax)
