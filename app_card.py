"""
Versión con diseño de tarjetas para la búsqueda semántica
"""

import gradio as gr
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import os
import json
from typing import Tuple
import time

# Configuración
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

# Variables globales
model = None
index = None
config = None

def load_resources():
    """Carga los recursos necesarios una sola vez"""
    global model, index, config
    
    if model is None or index is None:
        print("🔄 Cargando recursos...")
        
        try:
            with open('pinecone_config.json', 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            raise Exception("⚠️ No se encontró pinecone_config.json")
        
        model = SentenceTransformer(config['model_name'])
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(config['index_name'])
        
        print("✅ Recursos cargados")
    
    return model, index, config

def search_products_cards(query: str, num_results: int = 12) -> Tuple[str, str]:
    """Búsqueda que devuelve resultados en formato HTML cards"""
    try:
        model, index, config = load_resources()
        
        if not query.strip():
            return "", ""
        
        start_time = time.time()
        
        # Buscar
        query_vector = model.encode(query).tolist()
        results = index.query(
            vector=query_vector,
            top_k=num_results,
            include_metadata=True
        )
        
        if not results['matches']:
            return "<p style='text-align:center; color:#6b7280;'>No se encontraron productos</p>", ""
        
        # Crear HTML con cards
        cards_html = '<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px; padding: 16px;">'
        
        for match in results['matches']:
            m = match.get('metadata', {})
            score_color = "#10b981" if match['score'] > 0.8 else "#f59e0b" if match['score'] > 0.6 else "#6b7280"
            
            card = f'''
            <div style="background: white; border: 1px solid #e5e7eb; border-radius: 12px; padding: 16px; 
                        box-shadow: 0 1px 3px rgba(0,0,0,0.1); transition: all 0.2s; cursor: pointer;"
                 onmouseover="this.style.boxShadow='0 4px 6px rgba(0,0,0,0.1)'" 
                 onmouseout="this.style.boxShadow='0 1px 3px rgba(0,0,0,0.1)'">
                <h3 style="margin: 0 0 8px 0; font-size: 16px; font-weight: 600; color: #1f2937; 
                          overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                    {m.get('product_name', 'Sin nombre')[:40]}
                </h3>
                <div style="display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px;">
                    <span style="background: #eff6ff; color: #3b82f6; padding: 4px 8px; border-radius: 6px; 
                                font-size: 12px; font-weight: 500;">
                        {m.get('type', 'N/A')}
                    </span>
                    <span style="background: #fef3c7; color: #d97706; padding: 4px 8px; border-radius: 6px; 
                                font-size: 12px; font-weight: 600;">
                        ${m.get('product-price', 'N/A')}
                    </span>
                </div>
                <div style="border-top: 1px solid #f3f4f6; padding-top: 8px; margin-top: auto;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 12px; color: #6b7280;">
                            📍 {m.get('location', 'N/A')[:20]}
                        </span>
                        <span style="background: {score_color}; color: white; padding: 2px 8px; 
                                    border-radius: 12px; font-size: 11px; font-weight: 600;">
                            {match['score']:.0%}
                        </span>
                    </div>
                </div>
            </div>
            '''
            cards_html += card
        
        cards_html += '</div>'
        
        elapsed = time.time() - start_time
        status = f"✨ {len(results['matches'])} productos encontrados en {elapsed:.1f}s"
        
        return cards_html, status
        
    except Exception as e:
        return f"<p style='color: #ef4444;'>Error: {str(e)}</p>", ""

def create_cards_interface():
    """Interfaz con diseño de tarjetas"""
    
    css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: 0 auto !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }
    .search-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 40px 20px;
        border-radius: 0 0 24px 24px;
        margin: -20px -20px 20px -20px;
    }
    .gr-button-primary {
        background: white !important;
        color: #764ba2 !important;
        border: none !important;
        font-weight: 600 !important;
    }
    .gr-button-primary:hover {
        background: #f9fafb !important;
    }
    """
    
    with gr.Blocks(css=css, theme=gr.themes.Soft()) as app:
        # Header con gradiente
        with gr.Column(elem_classes="search-container"):
            gr.Markdown(
                """<h1 style='text-align: center; color: white; margin: 0 0 20px 0; 
                             font-size: 32px; font-weight: 700;'>
                    🔍 Búsqueda Inteligente
                </h1>""",
                elem_classes="header"
            )
            
            with gr.Row():
                query = gr.Textbox(
                    placeholder="¿Qué producto buscas hoy?",
                    show_label=False,
                    container=False,
                    elem_classes="search-box",
                    scale=5
                )
                btn = gr.Button("Buscar", size="lg", scale=1)
        
        # Status
        status = gr.Markdown(elem_classes="status-text")
        
        # Resultados en cards
        results = gr.HTML(label="", show_label=False)
        
        # Sugerencias
        with gr.Row():
            gr.Examples(
                [
                    ["croissant de pistacho"],
                    ["chocolate premium"],
                    ["postre artesanal"],
                    ["snack saludable"]
                ],
                inputs=query,
                examples_per_page=8,
                label="💡 Prueba buscar:"
            )
        
        # Handlers
        btn.click(
            search_products_cards,
            [query],
            [results, status]
        )
        query.submit(
            search_products_cards,
            [query],
            [results, status]
        )
    
    return app

if __name__ == "__main__":
    print("🚀 Iniciando versión con tarjetas...")
    try:
        load_resources()
        app = create_cards_interface()
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False
        )
    except Exception as e:
        print(f"❌ Error: {e}")