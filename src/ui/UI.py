from pathlib import Path

import gradio as gr
from src.LoRA_fine_tuner.SQLAgent import SQLAgent

#initialisation de l'agent
agent = SQLAgent()

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
                fn = agent.generate_response,
                save_history = True,
                stop_btn = True,
            )

        # Onglet 2 : futur modèle à implémenter
        with gr.Tab("Coming soon"):
            gr.Markdown( " **Coming soon** ")

    #Fonction permettant l'affichage de la descrition
    def toggle_description(visible: bool):
        return gr.update(visible=visible)

    info_btn.click(fn=lambda: toggle_description(True), outputs=[description_box])
    close_desc_btn.click(fn=lambda: toggle_description(False), outputs=[description_box])

def launch_ui():
    demo.launch()

if __name__ == "__main__":
    demo.launch()