import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os
from dotenv import load_dotenv

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

load_dotenv()
token = os.getenv("HF_TOKEN")
if token:
    login(token)
else:
    print(
        "Aucun token de connexion à HuggingFace n'a été fournit / trouvé. Le téléchargement d'un nouveau modèle peut être plus lent ou bloqué en raison des quotas.")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --- Option QLoRA ---
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
#     bnb_4bit_use_double_quant=True,
# )

USE_CUDA = torch.cuda.is_available()
USE_BF16 = USE_CUDA and torch.cuda.is_bf16_supported()
model_dtype = torch.bfloat16 if USE_BF16 else torch.float32

if not USE_CUDA:
    print("Aucun GPU détecté : l'entraînement va tourner sur CPU et sera très lent "
          "pour un modèle de 1.5B (potentiellement plusieurs heures, voire plus, "
          "pour 6 époques). Pense à réduire num_train_epochs / la taille du jeu de "
          "données pour tester rapidement, ou utilise un GPU (Colab, etc.) pour "
          "l'entraînement complet.")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    # quantization_config=bnb_config,   # à retirer si LoRA simple en bf16
    device_map="auto" if USE_CUDA else None,
    dtype=model_dtype,
)

# Configuration LoRA

from peft import LoraConfig

peft_config = LoraConfig(
    r=16,                 # rang de l'adaptateur
    lora_alpha=32,        # facteur d'échelle (souvent 2x le rang)
    lora_dropout=0.05,    # % de neurones désactivés aléatoirement pr éviter overfitting
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", # modules attention
                     "gate_proj", "up_proj", "down_proj"],  # MLP
)

from datasets import load_dataset
# 1. Chemin absolu du dossier contenant le script actuel (src/LoRA_fine_tuner)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Construction du chemin vers le dossier cible
# ".." permet de remonter dans le dossier parent (src)
dataset_dir = os.path.join(current_dir, "..", "baseline_dataset", "datasets")

# os.path.normpath "nettoie" le chemin en résolvant le ".." pour un rendu plus propre (optionnel mais recommandé)
dataset_dir = os.path.normpath(dataset_dir)

# 3. Chargement avec les chemins absolus générés dynamiquement
dataset = load_dataset("json", data_files={
    "train": os.path.join(dataset_dir, "train_dataset.json"),
    "validation": os.path.join(dataset_dir, "val_dataset.json"),
    "test": os.path.join(dataset_dir, "test_dataset.json")
})
print(f"Train: {len(dataset['train'])} | Val: {len(dataset['validation'])} | Test: {len(dataset['test'])}")




def create_lora(adapter_name: str):
    """
    Créer l'adaptateur LoRA et le sauvegarde en traçan sa perte.
    :param adapter_name: Le nom de l'adaptateur
    :return:
    """
    # Configuration de l'entrainement
    from trl.trainer.sft_trainer import SFTTrainer
    from trl.trainer.sft_config import SFTConfig
    path_to_adapter = os.path.join(current_dir, "..", "..", adapter_name)

    training_args = SFTConfig(
        output_dir=os.path.join(path_to_adapter, "checkpoints"),
        num_train_epochs=6,  # lit 6 fois le jeu de données
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,  # batch effectif = 16
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        # warmup_ratio=0.03,
        warmup_steps=20,
        max_length=512,  # longueur max des séquences
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,  # garde le meilleur checkpoint
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=USE_BF16,
        use_cpu=not USE_CUDA,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    print("Début de l'entrainement...")
    train_result = trainer.train()

    import matplotlib.pyplot as plt

    history = trainer.state.log_history
    train_loss = [(h["step"], h["loss"]) for h in history if "loss" in h]
    eval_loss  = [(h["step"], h["eval_loss"]) for h in history if "eval_loss" in h]

    plt.figure()
    plt.plot(*zip(*train_loss), label="train loss")
    plt.plot(*zip(*eval_loss), label="val loss", marker="o")
    plt.xlabel("step"); plt.ylabel("loss"); plt.legend(); plt.grid(True, linestyle="--", alpha=0.4)
    plt.title("Courbes de perte - LoRA Qwen2.5-1.5B-Instruct")
    plt.savefig("loss_curves.png")
    plt.show()


    # Sauvegarde de l'adaptateur
    trainer.save_model(os.path.join(path_to_adapter, "best_adapter"))

    tokenizer.save_pretrained(os.path.join(path_to_adapter, "best_adapter"))

