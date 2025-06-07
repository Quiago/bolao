"""
Versión con diseño de tarjetas para la búsqueda semántica
Incluye todos los filtros de la versión original
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

def search_products_cards(
    query: str, 
    num_results: int = 12, 
    filter_type: str = "", 
    min_score: float = 0.0
) -> Tuple[str, str]:
    """Búsqueda que devuelve resultados en formato HTML cards con filtros"""
    try:
        model, index, config = load_resources()
        
        if not query.strip():
            return "", ""
        
        start_time = time.time()
        
        # Generar embedding de la consulta
        query_vector = model.encode(query).tolist()
        
        # Construir filtro si se especifica
        filter_dict = None
        if filter_type and filter_type != "Todos":
            filter_dict = {"type": {"$eq": filter_type}}
        
        # Realizar búsqueda
        results = index.query(
            vector=query_vector,
            top_k=num_results,
            include_metadata=True,
            filter=filter_dict
        )
        
        if not results['matches']:
            return "<p style='text-align:center; color:#6b7280; padding: 40px;'>No se encontraron productos</p>", ""
        
        # Filtrar por score mínimo
        filtered_matches = [m for m in results['matches'] if m['score'] >= min_score]
        
        if not filtered_matches:
            return f"<p style='text-align:center; color:#6b7280; padding: 40px;'>No se encontraron productos con score ≥ {min_score:.2f}</p>", ""
        
        # Crear HTML con cards
        cards_html = '<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px; padding: 16px;">'
        
        for match in filtered_matches:
            m = match.get('metadata', {})
            score_color = "#10b981" if match['score'] > 0.8 else "#f59e0b" if match['score'] > 0.6 else "#6b7280"
            
            card = f'''
            <div style="background: white; border: 1px solid #e5e7eb; border-radius: 12px; padding: 16px; 
                        box-shadow: 0 1px 3px rgba(0,0,0,0.1); transition: all 0.2s; cursor: pointer;"
                 onmouseover="this.style.boxShadow='0 4px 6px rgba(0,0,0,0.1)'" 
                 onmouseout="this.style.boxShadow='0 1px 3px rgba(0,0,0,0.1)'">
                <h3 style="margin: 0 0 8px 0; font-size: 16px; font-weight: 600; color: #1f2937; 
                          overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                    {m.get('product_name', 'N/A')[:40]}
                </h3>
                <div style="display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px;">
                    <span style="background: #eff6ff; color: #3b82f6; padding: 4px 8px; border-radius: 6px; 
                                font-size: 12px; font-weight: 500;">
                        {m.get('type', 'N/A')}
                    </span>
                    <span style="background: #fef3c7; color: #d97706; padding: 4px 8px; border-radius: 6px; 
                                font-size: 12px; font-weight: 600;">
                        ${m.get('product_price', 'N/A')}
                    </span>
                </div>
                <p style="font-size: 13px; color: #4b5563; margin: 8px 0; line-height: 1.4;">
                    🏪 {m.get('name', 'N/A')[:30]}
                </p>
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
        status = f"✨ {len(filtered_matches)} productos encontrados en {elapsed:.1f}s"
        
        return cards_html, status
        
    except Exception as e:
        return f"<p style='color: #ef4444; text-align: center; padding: 40px;'>Error: {str(e)}</p>", ""

def get_product_types():
    """Obtiene los tipos de productos únicos del CSV"""
    try:
        df = pd.read_csv('product.csv')
        types = ['Todos'] + sorted(df['type'].dropna().unique().tolist())
        return types
    except Exception:
        return ['Todos']

def create_cards_interface():
    """Interfaz con diseño de tarjetas y filtros completos"""
    
    # Cargar tipos de productos
    product_types = get_product_types()
    
    css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: 0 auto !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }
    .search-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 40px 20px 30px 20px;
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
    .filter-section {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        margin-top: 20px;
    }
    .gr-form {
        background: transparent !important;
    }
    .gr-input-label {
        color: white !important;
        font-size: 13px !important;
        font-weight: 500 !important;
    }
    .gr-dropdown {
        background: white !important;
    }
    .gr-slider .gr-slider-container {
        background: rgba(255, 255, 255, 0.2) !important;
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
            
            # Filtros en un acordeón elegante
            with gr.Accordion("⚙️ Filtros avanzados", open=False, elem_classes="filter-section"):
                with gr.Row():
                    num_results = gr.Slider(
                        minimum=6,
                        maximum=50,
                        value=12,
                        step=6,
                        label="📊 Número de resultados",
                        info="Cantidad de productos a mostrar"
                    )
                    filter_type = gr.Dropdown(
                        choices=product_types,
                        value="Todos",
                        label="🏷️ Filtrar por tipo",
                        info="Categoría de productos"
                    )
                    min_score = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.0,
                        step=0.05,
                        label="📈 Score mínimo",
                        info="Relevancia mínima (0-1)"
                    )
        
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
                    ["snack saludable"],
                    ["dulce tradicional"],
                    ["regalo gourmet"]
                ],
                inputs=query,
                examples_per_page=8,
                label="💡 Prueba buscar:"
            )
        
        # Handlers
        btn.click(
            search_products_cards,
            [query, num_results, filter_type, min_score],
            [results, status]
        )
        query.submit(
            search_products_cards,
            [query, num_results, filter_type, min_score],
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