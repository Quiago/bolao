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
from feedback import send_feedback_email

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

def open_google_forms():
    """
    Abre Google Forms en nueva pestaña
    Reemplaza 'TU_GOOGLE_FORM_URL' con tu URL real de Google Forms
    """
    # Para crear tu Google Form:
    # 1. Ve a forms.google.com
    # 2. Crea un nuevo formulario
    # 3. Añade preguntas como:
    #    - Calificación general (escala 1-5)
    #    - ¿Qué te gustó más?
    #    - ¿Qué podríamos mejorar?
    #    - ¿Recomendarías BOLAO?
    #    - Comentarios adicionales
    # 4. En "Enviar", copia el enlace
    # 5. Reemplaza la URL abajo
    
    google_form_url = "https://forms.gle/TU_GOOGLE_FORM_ID"  # Reemplazar con tu URL
    
    return f"""
    <script>
        window.open('{google_form_url}', '_blank');
    </script>
    <p style='color: #10b981; text-align: center;'>
        📝 Formulario de feedback abierto en nueva pestaña
    </p>
    """

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
        padding: 20px 20px;
        border-radius: 0 0 24px 24px;
        margin: -20px -20px 15px -20px;
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
    .gr-button-secondary {
        background: rgba(255,255,255,0.9) !important;
        color: #6b7280 !important;
        border: 1px solid rgba(255,255,255,0.3) !important;
        font-weight: 500 !important;
    }
    .gr-button-secondary:hover {
        background: white !important;
        color: #374151 !important;
    }
    .filter-section {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 15px;
        margin-top: 10px;
        margin-bottom: 15px;
        backdrop-filter: blur(10px);
        min-height: 140px;
    }
    .gr-form {
        background: transparent !important;
    }
    /* Alineación específica para dropdowns */
    .dropdown-container {
        display: flex !important;
        align-items: flex-start !important;
        gap: 16px !important;
    }
    
    .dropdown-item {
        flex: 1 !important;
        min-height: 80px !important;
    }
    
    /* Forzar alineación de labels */
    .gr-input-label {
        color: white !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        margin-bottom: 6px !important;
        height: 20px !important;
        display: block !important;
    }
    
    /* Asegurar mismo tamaño para dropdowns */
    .gr-dropdown {
        min-height: 60px !important;
    }
    
    .gr-dropdown > div {
        min-height: 60px !important;
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
    
    /* Responsive design */
    @media (max-width: 768px) {
        .gradio-container {
            max-width: 100% !important;
            padding: 0 10px !important;
        }
        .search-container {
            padding: 20px 15px !important;
            margin: -10px -10px 15px -10px !important;
        }
        .filter-section {
            padding: 15px !important;
            margin-top: 10px !important;
        }
        .gr-input-label {
            font-size: 13px !important;
        }
    }
    
    /* Estilos para diseño compacto */
    .compact-feedback {
        background: #f8fafc !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        padding: 15px !important;
        margin: 10px 0 !important;
    }
    
    .compact-examples {
        margin: 10px 0 !important;
    }
    
    /* Layout responsivo mejorado */
    @media (max-width: 768px) {
        .gradio-container {
            max-width: 100% !important;
            padding: 0 10px !important;
        }
        .search-container {
            padding: 15px 10px !important;
            margin: -10px -10px 10px -10px !important;
        }
        .filter-section {
            padding: 10px !important;
            margin-top: 5px !important;
            min-height: 120px !important;
        }
        .gr-input-label {
            font-size: 12px !important;
        }
    }
    """
    
    with gr.Blocks(css=css, theme=gr.themes.Soft()) as app:
        # Header con gradiente (más compacto)
        with gr.Column(elem_classes="search-container"):
            gr.Markdown(
                """<h1 style='text-align: center; color: white; margin: 0 0 5px 0; 
                             font-size: 28px; font-weight: 700; text-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                    🔍 BOLAO
                </h1>
                <p style='text-align: center; color: rgba(255,255,255,0.9); margin: 0 0 15px 0; 
                         font-size: 14px; font-weight: 400;'>
                    Encuentra productos al instante con filtros personalizados
                </p>""",
                elem_classes="header"
            )
            
            with gr.Row():
                query = gr.Textbox(
                    placeholder="¿Qué producto buscas hoy? Ej: croissant de pistacho, chocolate premium...",
                    show_label=False,
                    container=False,
                    elem_classes="search-box",
                    scale=4
                )
                btn = gr.Button("🔍 Buscar", size="lg", scale=1)
                clear_btn = gr.Button("🗑️ Limpiar", variant="secondary", size="lg", scale=1)
            
            # Tips rápidos siempre visibles (más compactos)
            gr.Markdown("""
            <div style='background: rgba(255,255,255,0.15); border-radius: 8px; padding: 8px; margin: 10px 0 5px 0;'>
                <div style='color: white; font-size: 12px; line-height: 1.3; text-align: center;'>
                    💡 <strong>Tips:</strong> 
                    Escribe naturalmente • Usa filtros para refinar • Verde = muy relevante
                </div>
            </div>
            """)
            
            # Filtros compactos en acordeón
            with gr.Accordion("⚙️ Filtros avanzados", open=False, elem_classes="filter-section"):
                # Dropdowns - cada uno en su propia fila para alineación perfecta
                with gr.Row():
                    filter_type = gr.Dropdown(
                        choices=product_types,
                        value="Todos",
                        label="🏷️ Tipo de producto",
                        info="Filtrar por categoría",
                        container=True
                    )
                
                with gr.Row():
                    filter_location = gr.Dropdown(
                        choices=locations,
                        value="Todas",
                        label="📍 Ubicación", 
                        info="Filtrar por ciudad/zona",
                        container=True
                    )
                
                # Sliders en una fila (más compactos)
                with gr.Row():
                    with gr.Column(scale=1):
                        num_results = gr.Slider(
                            minimum=6,
                            maximum=50,
                            value=12,
                            step=6,
                            label="📊 Cantidad",
                            info="Núm. productos"
                        )
                    with gr.Column(scale=1):
                        min_score = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.0,
                            step=0.05,
                            label="📈 Relevancia",
                            info="Score mínimo"
                        )
        
        # Layout en dos columnas para que todo quepa en la primera pantalla
        with gr.Row():
            # Columna izquierda: Ejemplos
            with gr.Column(scale=1):
                gr.Examples(
                    [
                        ["croissant de pistacho"],
                        ["chocolate premium"],
                        ["postre sin gluten"],
                        ["snack saludable"],
                        ["dulce tradicional"],
                        ["café especial"]
                    ],
                    inputs=query,
                    examples_per_page=6,
                    label="💡 Prueba buscar:"
                )
            
            # Información adicional compacta
            with gr.Column(scale=1):
                gr.Markdown("""
                ℹ️ Guía de uso
                **🔍 Búsqueda inteligente:**
                Escribe naturalmente y baje en la pantalla para ver los resultados.

                **⚙️ Filtros:** Combina tipo, ubicación, cantidad y relevancia para refinar resultados.

                **📊 Relevancia:** Verde (>80%) = muy relevante, Amarillo (60-80%) = relevante, Gris (<60%) = menos relevante.

                **🔗 Tarjetas:** Muestran precio, establecimiento, dirección y redes sociales cuando están disponibles.
                """)
            
            # Columna derecha: Feedback compacto
            with gr.Column(scale=1):
                gr.Markdown("""
                <div style='background: #f8fafc; border-radius: 8px; padding: 15px; border: 1px solid #e2e8f0;'>
                    <h4 style='color: #374151; margin: 0 0 10px 0; font-size: 16px; font-weight: 600;'>
                        💬 ¿Cómo vamos?
                    </h4>
                    <p style='color: #6b7280; margin: 0 0 10px 0; font-size: 13px;'>
                        Tu feedback nos ayuda a mejorar
                    </p>
                </div>
                """)
                
                rating = gr.Radio(
                    choices=["⭐ Malo", "⭐⭐ Regular", "⭐⭐⭐ Bueno", "⭐⭐⭐⭐ Muy bueno", "⭐⭐⭐⭐⭐ Excelente"],
                    label="🌟 Califica:",
                    value=None
                )
                
                quick_feedback = gr.Textbox(
                    placeholder="Comentario breve...",
                    label="💭 Comentario:",
                    lines=1,
                    max_lines=2
                )
                
                with gr.Row():
                    submit_feedback_btn = gr.Button("📤 Enviar", variant="primary", size="sm")
                    feedback_btn = gr.Button("📝 Más info", variant="secondary", size="sm")
        
        feedback_status = gr.Markdown("", elem_classes="status-text")
        
        # Status
        status = gr.Markdown(elem_classes="status-text")
        
        # Resultados en cards
        results = gr.HTML(label="", show_label=False)      
        # Footer compacto
        gr.Markdown("""
        <div style='text-align: center; color: #9ca3af; font-size: 10px; margin-top: 20px; padding: 10px; 
                    border-top: 1px solid #e5e7eb;'>
            🤖 BOLAO - Búsqueda inteligente powered by AI
        </div>
        """)
        
        # Handlers
        def search_handler(query, num_results, filter_type, filter_location, min_score):
            print(f"🔍 Búsqueda: '{query}' | Tipo: '{filter_type}' | Ubicación: '{filter_location}' | Cantidad: {num_results} | Score: {min_score}")
            return search_products_cards(query, num_results, filter_type, filter_location, min_score)
        
        def clear_filters():
            return (
                "Todos",  # filter_type
                "Todas",  # filter_location
                12,  # num_results
                0.0,  # min_score
                "",  # results
                "🔄 Filtros limpiados. ¡Haz una nueva búsqueda!"  # status
            )
        
        def handle_feedback(rating, comment):
            result = send_feedback_email(rating or "", comment or "")
            return result, "", None  # status, limpiar comment, limpiar rating
        
        def handle_google_forms():
            return open_google_forms()
        
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
            outputs=[filter_type, filter_location, num_results, min_score, results, status]
        )
        
        # Eventos de feedback
        submit_feedback_btn.click(
            handle_feedback,
            inputs=[rating, quick_feedback],
            outputs=[feedback_status, quick_feedback, rating]
        )
        
        feedback_btn.click(
            handle_google_forms,
            outputs=[feedback_status]
        )
    
    return app

if __name__ == "__main__":
    print("🚀 Iniciando búsqueda inteligente con filtros avanzados...")
    try:
        load_resources()
        app = create_cards_interface()
        app.launch(
            #server_name="0.0.0.0",
            #server_port=7860,
            share=True,
            show_error=True
        )
    except Exception as e:
        print(f"❌ Error al iniciar la aplicación: {e}")
        import traceback
        traceback.print_exc()