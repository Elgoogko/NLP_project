from pathlib import Path

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = str(Path(__file__).resolve().parents[2] / "qwen-sql-lora" / "best_adapter")

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

system_message="Tu es un assistant technique expert en MySQL, conçu pour aider les débutants à écrire des requêtes SQL."

def generate_response(user_message, history):
    messages = [{"role": "system", "content": system_message},]

    # Conversion de list/dictionnaire à str
    for entry in history:
        if isinstance(entry, (list, tuple)):
            u_text, a_text = entry
            if u_text:
                messages.append({"role": "user", "content": str(u_text)})
            if a_text:
                messages.append({"role": "assistant", "content": str(a_text)})
        elif isinstance(entry, dict):
            messages.append({
                "role": entry.get("role", "user"),
                "content": str(entry.get("content", ""))
            })

    messages.append({"role": "user", "content": user_message})

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

with gr.Blocks() as demo:
    gr.Markdown("# Hub Multi-modèle")
    gr.Markdown("Basculez d'un onglet à l'autre pour tester différents modèles.")
    
    with gr.Tabs():
        # Onglet 1 : Modèle Qwen Finetuné
        with gr.Tab("Qwen 2.5"):
            with gr.Group(visible=False) as description_box:
                gr.Markdown(
                    """
                    Modèle Qwen 2.5 1.5B adapté avec un adaptateur LoRA pour l'assistance SQL / MySQL.
                    """
                )
                close_desc_btn = gr.Button("Fermer la description", size="sm")
            info_btn = gr.Button("Description du modèle", variant="secondary")
            gr.ChatInterface(
                fn = generate_response,
                save_history = True,
                stop_btn = True,
            )

        # Onglet 2 : futur modèle à implémenter
        with gr.Tab("Coming soon"):
            gr.Markdown( " **Coming soon** ")

    def toggle_description(visible: bool):
        return gr.update(visible=visible)

    info_btn.click(fn=lambda: toggle_description(True), outputs=[description_box])
    close_desc_btn.click(fn=lambda: toggle_description(False), outputs=[description_box])

def launch_ui():
    demo.launch()
