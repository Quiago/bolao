"""
Versión minimalista de la aplicación de búsqueda semántica
"""

import gradio as gr
import pandas as pd
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

def search_products(query: str, num_results: int = 10) -> Tuple[pd.DataFrame, str]:
    """Búsqueda semántica simplificada"""
    try:
        model, index, config = load_resources()
        
        if not query.strip():
            return pd.DataFrame(), ""
        
        start_time = time.time()
        
        # Buscar
        query_vector = model.encode(query).tolist()
        results = index.query(
            vector=query_vector,
            top_k=num_results,
            include_metadata=True
        )
        
        if not results['matches']:
            return pd.DataFrame(), "Sin resultados"
        
        # Crear DataFrame compacto
        data = []
        for i, match in enumerate(results['matches'], 1):
            m = match.get('metadata', {})
            data.append({
                '#': i,
                'Producto': m.get('product_name', 'N/A')[:50],  # Limitar longitud
                'Tipo': m.get('type', 'N/A'),
                '💰': f"${m.get('product-price', 'N/A')}",
                '📍': m.get('location', 'N/A')[:20],  # Limitar longitud
                '⭐': f"{match['score']:.2f}"
            })
        
        df = pd.DataFrame(data)
        elapsed = time.time() - start_time
        
        return df, f"✓ {len(data)} resultados en {elapsed:.1f}s"
        
    except Exception as e:
        return pd.DataFrame(), f"Error: {str(e)}"

def create_minimal_interface():
    """Interfaz ultra minimalista"""
    
    css = """
    #component-0 {
        max-width: 900px;
        margin: 0 auto;
    }
    .gradio-container {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    .gr-button-primary {
        background: #2563eb !important;
        border: none !important;
    }
    .gr-input {
        border-radius: 8px !important;
    }
    tbody tr:hover {
        background-color: #f9fafb !important;
    }
    .status {
        text-align: right;
        color: #6b7280;
        font-size: 0.875rem;
    }
    """
    
    with gr.Blocks(css=css, theme=gr.themes.Base()) as app:
        with gr.Column(elem_id="component-0"):
            # Header compacto
            gr.Markdown("## 🔍 Búsqueda Semántica")
            
            # Barra de búsqueda simple
            with gr.Row():
                query = gr.Textbox(
                    placeholder="Buscar productos...",
                    show_label=False,
                    container=False,
                    elem_classes="search-input",
                    scale=8
                )
                btn = gr.Button("Buscar", size="sm", scale=1)
                results_count = gr.Slider(5, 20, 10, step=5, label="", visible=False)
            
            # Status minimalista
            status = gr.Markdown(elem_classes="status", visible=True)
            
            # Resultados
            results = gr.DataFrame(
                headers=["#", "Producto", "Tipo", "💰", "📍", "⭐"],
                wrap=True,
                #height=350,
                interactive=False,
                show_label=False
            )
            
            # Búsquedas rápidas
            gr.Examples(
                ["croissant", "chocolate", "dulce", "snack"],
                inputs=query,
                examples_per_page=10,
                label="Búsquedas populares:"
            )
            
            # Handlers
            btn.click(
                search_products,
                [query, results_count],
                [results, status]
            )
            query.submit(
                search_products,
                [query, results_count],
                [results, status]
            )
    
    return app

if __name__ == "__main__":
    print("🚀 Iniciando versión minimalista...")
    try:
        load_resources()
        app = create_minimal_interface()
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False
        )
    except Exception as e:
        print(f"❌ Error: {e}")