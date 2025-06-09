"""
Versión con diseño de tarjetas para la búsqueda semántica
Incluye filtros por tipo y location, muestra dirección y redes sociales
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
    filter_location: str = "",
    min_score: float = 0.0
) -> Tuple[str, str]:
    """Búsqueda que devuelve resultados en formato HTML cards con filtros múltiples"""
    try:
        model, index, config = load_resources()
        
        if not query.strip():
            return "", ""
        
        start_time = time.time()
        
        # Generar embedding de la consulta
        query_vector = model.encode(query).tolist()
        
        # Construir filtros múltiples - Pinecone requiere $and en el nivel superior
        filters = []
        
        if filter_type and filter_type != "Todos":
            filters.append({"type": {"$eq": filter_type}})
            
        if filter_location and filter_location != "Todas":
            filters.append({"location": {"$eq": filter_location}})
        
        # Usar $and siempre que haya filtros
        if filters:
            if len(filters) == 1:
                filter_dict = filters[0]
            else:
                filter_dict = {"$and": filters}
        else:
            filter_dict = None
        
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
        cards_html = '<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 16px; padding: 16px;">'
        
        for match in filtered_matches:
            m = match.get('metadata', {})
            score_color = "#10b981" if match['score'] > 0.8 else "#f59e0b" if match['score'] > 0.6 else "#6b7280"
            
            # Construir sección de redes sociales
            social_media_html = ""
            social_networks = {
                'facebook': '📘',
                'instagram': '📷', 
                'twitter': '🐦',
                'phone': '📱',
                'website': '🌐',
                'email': '📧'
            }
            
            social_links = []
            for network, icon in social_networks.items():
                if network in m and m[network] and str(m[network]).strip():
                    link = m[network]
                    # Añadir protocolo si no lo tiene
                    if network == 'website' and not link.startswith(('http://', 'https://')):
                        link = f"https://{link}"
                    if network == 'email' and not link.startswith('mailto:'):
                        link = f"mailto:{link}"
                    if network == "instagram" and not link.startswith('https://'):
                        link = f"https://www.instagram.com/{link.lstrip('@')}"
                    if network == "facebook" and not link.startswith('https://'):
                        link = f"https://www.facebook.com/{link.lstrip('@')}"
                    
                    if network == "phone":
                        social_links.append(f'''
                            <a href="#" style="text-decoration: none; 
                               background: #f3f4f6; padding: 4px 8px; border-radius: 6px; 
                               font-size: 12px; color: #374151; transition: all 0.2s;"
                               onmouseover="this.style.background='#e5e7eb'" 
                               onmouseout="this.style.background='#f3f4f6'">
                                {icon} {int(link)}
                            </a>
                        ''')
                    else:
                        social_links.append(f'''
                            <a href="{link}" target="_blank" style="text-decoration: none; 
                               background: #f3f4f6; padding: 4px 8px; border-radius: 6px; 
                               font-size: 12px; color: #374151; transition: all 0.2s;"
                               onmouseover="this.style.background='#e5e7eb'" 
                               onmouseout="this.style.background='#f3f4f6'">
                                {icon} {network.title()}
                            </a>
                        ''')
            
            if social_links:
                social_media_html = f'''
                    <div style="margin: 8px 0;">
                        <div style="display: flex; flex-wrap: wrap; gap: 4px;">
                            {" ".join(social_links)}
                        </div>
                    </div>
                '''
            
            # Construir dirección si existe
            address_html = ""
            if 'address' in m and m['address'] and str(m['address']).strip():
                address_html = f'''
                    <p style="font-size: 12px; color: #6b7280; margin: 6px 0; line-height: 1.3;">
                        📍 {m['address'][:50]}{'...' if len(str(m['address'])) > 50 else ''}
                    </p>
                '''
            
            card = f'''
            <div style="background: white; border: 1px solid #e5e7eb; border-radius: 12px; padding: 16px; 
                        box-shadow: 0 1px 3px rgba(0,0,0,0.1); transition: all 0.2s; cursor: pointer;
                        display: flex; flex-direction: column; height: 100%;"
                 onmouseover="this.style.boxShadow='0 4px 6px rgba(0,0,0,0.1)'; this.style.transform='translateY(-2px)'" 
                 onmouseout="this.style.boxShadow='0 1px 3px rgba(0,0,0,0.1)'; this.style.transform='translateY(0)'">
                
                <h3 style="margin: 0 0 8px 0; font-size: 16px; font-weight: 600; color: #1f2937; 
                          overflow: hidden; text-overflow: ellipsis; white-space: nowrap;"
                    title="{m.get('product_name', 'N/A')}">
                    {m.get('product_name', 'N/A')[:40]}{'...' if len(str(m.get('product_name', ''))) > 40 else ''}
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
                
                <p style="font-size: 13px; color: #4b5563; margin: 8px 0; line-height: 1.4; font-weight: 500;">
                    🏪 {m.get('name', 'N/A')[:35]}{'...' if len(str(m.get('name', ''))) > 35 else ''}
                </p>
                
                {address_html}
                
                <p style="font-size: 12px; color: #6b7280; margin: 4px 0;">
                    📍 {m.get('location', 'N/A')}
                </p>
                
                {social_media_html}
                
                <div style="border-top: 1px solid #f3f4f6; padding-top: 12px; margin-top: auto;">
                    <div style="display: flex; justify-content: center; align-items: center;">
                        <span style="background: {score_color}; color: white; padding: 4px 12px; 
                                    border-radius: 20px; font-size: 12px; font-weight: 600;">
                            Relevancia: {match['score']:.0%}
                        </span>
                    </div>
                </div>
            </div>
            '''
            cards_html += card
        
        cards_html += '</div>'
        
        elapsed = time.time() - start_time
        filters_applied = []
        if filter_type and filter_type != "Todos":
            filters_applied.append(f"Tipo: {filter_type}")
        if filter_location and filter_location != "Todas":
            filters_applied.append(f"Ubicación: {filter_location}")
        if min_score > 0:
            filters_applied.append(f"Score ≥ {min_score:.2f}")
        
        filters_text = f" | Filtros: {', '.join(filters_applied)}" if filters_applied else ""
        status = f"✨ {len(filtered_matches)} productos encontrados en {elapsed:.1f}s{filters_text}"
        
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

def get_locations():
    """Obtiene las ubicaciones únicas del CSV"""
    try:
        df = pd.read_csv('product.csv')
        locations = ['Todas'] + sorted(df['location'].dropna().unique().tolist())
        return locations
    except Exception:
        return ['Todas']

def create_cards_interface():
    """Interfaz con diseño de tarjetas y filtros completos"""
    
    # Cargar tipos de productos y ubicaciones
    product_types = get_product_types()
    locations = get_locations()
    
    css = """
    .gradio-container {
        max-width: 1400px !important;
        margin: 0 auto !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }
    .search-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 40px 20px 40px 20px;
        border-radius: 0 0 24px 24px;
        margin: -20px -20px 20px -20px;
    }
    .gr-button-primary {
        background: white !important;
        color: #764ba2 !important;
        border: none !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }
    .gr-button-primary:hover {
        background: #f9fafb !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15) !important;
    }
    .filter-section {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 24px 24px 60px 24px;
        margin-top: 20px;
        margin-bottom: 40px;
        backdrop-filter: blur(10px);
        min-height: 200px;
    }
    .gr-form {
        background: transparent !important;
    }
    .gr-input-label {
        color: white !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        margin-bottom: 8px !important;
    }
    
    /* Asegurar que los dropdowns no se superpongan */
    .dropdown-row {
        margin-bottom: 40px !important;
        padding-bottom: 20px !important;
    }
    
    .slider-row {
        margin-top: 30px !important;
    }
    
    .gr-slider .gr-slider-container {
        background: rgba(255, 255, 255, 0.2) !important;
    }
    .status-text {
        text-align: center;
        color: #6b7280;
        font-weight: 500;
        margin: 16px 0;
    }
    
    /* Forzar layout normal para dropdowns */
    .gr-dropdown {
        position: static !important;
    }
    
    .gr-dropdown > div {
        position: static !important;
    }
    """
    
    with gr.Blocks(css=css, theme=gr.themes.Soft()) as app:
        # Header con gradiente
        with gr.Column(elem_classes="search-container"):
            gr.Markdown(
                """<h1 style='text-align: center; color: white; margin: 0 0 8px 0; 
                             font-size: 32px; font-weight: 700; text-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                    🔍 BOLAO APP
                </h1>
                <p style='text-align: center; color: rgba(255,255,255,0.9); margin: 0 0 20px 0; 
                         font-size: 16px; font-weight: 400;'>
                    Encuentra productos al instante con nuestra búsqueda inteligente
                </p>""",
                elem_classes="header"
            )
            
            with gr.Row():
                query = gr.Textbox(
                    placeholder="¿Qué producto buscas hoy? Ej: croissant de pistacho, chocolate premium...",
                    show_label=False,
                    container=False,
                    elem_classes="search-box",
                    scale=5
                )
                btn = gr.Button("🔍 Buscar", size="lg", scale=1)
            
            # Filtros en un acordeón elegante con más espacio
            with gr.Accordion("⚙️ Filtros avanzados", open=False, elem_classes="filter-section"):
                # Primera fila: Dropdowns con mucho espacio
                with gr.Row(elem_classes="dropdown-row"):
                    with gr.Column():
                        filter_type = gr.Dropdown(
                            choices=product_types,
                            value="Todos",
                            label="🏷️ Tipo de producto",
                            info="Filtrar por categoría"
                        )
                    with gr.Column():
                        filter_location = gr.Dropdown(
                            choices=locations,
                            value="Todas",
                            label="📍 Ubicación",
                            info="Filtrar por ciudad/zona"
                        )
                
                # Espaciador visual
                gr.HTML("<div style='height: 20px;'></div>")
                
                # Botón para limpiar filtros
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.HTML("")  # Espaciador
                    with gr.Column(scale=1):
                        clear_btn = gr.Button("🗑️ Limpiar filtros", variant="secondary", size="sm")
                    with gr.Column(scale=2):
                        gr.HTML("")  # Espaciador
                
                # Espaciador visual
                gr.HTML("<div style='height: 10px;'></div>")
                
                # Segunda fila: Sliders con separación
                with gr.Row(elem_classes="slider-row"):
                    with gr.Column():
                        num_results = gr.Slider(
                            minimum=6,
                            maximum=50,
                            value=12,
                            step=6,
                            label="📊 Cantidad de resultados",
                            info="Número de productos a mostrar"
                        )
                    with gr.Column():
                        min_score = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.0,
                            step=0.05,
                            label="📈 Relevancia mínima",
                            info="Score de similitud (0-100%)"
                        )
        
        # Espacio adicional después de filtros
        gr.HTML("<div style='height: 30px;'></div>")
        
        # Status
        status = gr.Markdown(elem_classes="status-text")
        
        # Resultados en cards
        results = gr.HTML(label="", show_label=False)
        
        # Sugerencias
        with gr.Row():
            gr.Examples(
                [
                    ["croissant de pistacho"],
                    ["chocolate premium artesanal"],
                    ["postre sin gluten"],
                    ["snack saludable orgánico"],
                    ["dulce tradicional casero"],
                    ["regalo gourmet especial"],
                    ["café de especialidad"],
                    ["torta de cumpleaños"]
                ],
                inputs=query,
                examples_per_page=8,
                label="💡 Prueba buscar:"
            )
        
        # Información adicional
        with gr.Accordion("ℹ️ Guía de uso", open=False):
            gr.Markdown("""
            ### 🚀 Cómo usar la búsqueda inteligente:
            
            **🔍 Búsqueda:**
            - Escribe de forma natural: "croissant de pistacho", "chocolate premium"
            - La IA entiende sinónimos y términos relacionados
            - No necesitas palabras exactas
            
            **⚙️ Filtros avanzados:**
            - **Tipo**: Filtra por categoría específica (dulces, snacks, etc.)
            - **Ubicación**: Encuentra productos por zona geográfica
            - **Cantidad**: Ajusta cuántos resultados ver (6-50)
            - **Relevancia**: Controla la precisión de búsqueda (0-100%)
            
            **📊 Entendiendo los resultados:**
            - **Verde (>80%)**: Muy relevante para tu búsqueda
            - **Amarillo (60-80%)**: Relevante
            - **Gris (<60%)**: Menos relevante
            
            **🔗 Información en tarjetas:**
            - Precio y tipo de producto
            - Nombre del establecimiento y dirección
            - Redes sociales (cuando están disponibles)
            - Teléfono y contacto
            
            **💡 Tip:** Si no encuentras lo que buscas, prueba con términos más generales o ajusta los filtros
            """)
        
        # Handlers
        def search_handler(query, num_results, filter_type, filter_location, min_score):
            print(f"🔍 Búsqueda: '{query}' | Tipo: '{filter_type}' | Ubicación: '{filter_location}' | Cantidad: {num_results} | Score: {min_score}")
            return search_products_cards(query, num_results, filter_type, filter_location, min_score)
        
        def clear_filters():
            return (
                12,  # num_results
                "Todos",  # filter_type
                "Todas",  # filter_location
                0.0,  # min_score
                "",  # results
                "🔄 Filtros limpiados. ¡Haz una nueva búsqueda!"  # status
            )
        
        # Eventos de búsqueda
        btn.click(
            search_handler,
            inputs=[query, num_results, filter_type, filter_location, min_score],
            outputs=[results, status]
        )
        query.submit(
            search_handler,
            inputs=[query, num_results, filter_type, filter_location, min_score],
            outputs=[results, status]
        )
        
        # Evento para limpiar filtros
        clear_btn.click(
            clear_filters,
            outputs=[num_results, filter_type, filter_location, min_score, results, status]
        )
    
    return app

if __name__ == "__main__":
    print("🚀 Iniciando búsqueda inteligente con filtros avanzados...")
    try:
        load_resources()
        app = create_cards_interface()
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True
        )
    except Exception as e:
        print(f"❌ Error al iniciar la aplicación: {e}")
        import traceback
        traceback.print_exc()