"""
Aplicación Gradio para búsqueda semántica de productos
Esta app se conecta al índice de Pinecone existente y permite hacer búsquedas
"""

import gradio as gr
import pandas as pd
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import os
import json
from typing import List, Dict, Tuple
import time

# Configuración
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

# Variables globales para evitar recargar
model = None
index = None
config = None

def load_resources():
    """Carga los recursos necesarios (modelo e índice) una sola vez"""
    global model, index, config
    
    if model is None or index is None:
        print("🔄 Cargando recursos...")
        
        # Cargar configuración
        try:
            with open('pinecone_config.json', 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            raise Exception("⚠️ No se encontró pinecone_config.json. Ejecuta primero el script de indexación.")
        
        # Cargar modelo
        print("  - Cargando modelo de embeddings...")
        model = SentenceTransformer(config['model_name'])
        
        # Conectar a Pinecone
        print("  - Conectando a Pinecone...")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(config['index_name'])
        
        print("✅ Recursos cargados exitosamente")
    
    return model, index, config

def search_products(
    query: str, 
    num_results: int = 10, 
    filter_type: str = "", 
    min_score: float = 0.0
) -> Tuple[pd.DataFrame, str]:
    """
    Busca productos similares usando búsqueda semántica
    """
    try:
        # Cargar recursos si no están cargados
        model, index, config = load_resources()
        
        if not query.strip():
            return pd.DataFrame(), "⚠️ Por favor ingresa un término de búsqueda"
        
        # Medir tiempo de búsqueda
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
        
        # Procesar resultados
        if not results['matches']:
            return pd.DataFrame(), "No se encontraron productos similares"
        
        # Filtrar por score mínimo
        filtered_matches = [m for m in results['matches'] if m['score'] >= min_score]
        
        if not filtered_matches:
            return pd.DataFrame(), f"No se encontraron productos con score >= {min_score}"
        
        # Convertir a DataFrame
        data = []
        for i, match in enumerate(filtered_matches, 1):
            metadata = match.get('metadata', {})
            data.append({
                '#': i,
                'Producto': metadata.get('product_name', 'N/A'),
                'Tipo': metadata.get('type', 'N/A'),
                'Precio': metadata.get('product-price', 'N/A'),
                'Lugar': metadata.get('name', 'N/A'),
                'Ubicación': metadata.get('location', 'N/A'),
            })
        
        df = pd.DataFrame(data)
        
        # Mensaje de estado
        search_time = time.time() - start_time
        status = f"✅ Encontrados {len(filtered_matches)} productos en {search_time:.2f} segundos"
        
        return df, status
        
    except Exception as e:
        return pd.DataFrame(), f"❌ Error: {str(e)}"

def get_product_types():
    """Obtiene los tipos de productos únicos del CSV"""
    try:
        df = pd.read_csv('product.csv')
        types = ['Todos'] + sorted(df['type'].dropna().unique().tolist())
        return types
    except:
        return ['Todos']

def create_interface():
    """Crea la interfaz de Gradio"""
    
    # Cargar tipos de productos
    product_types = get_product_types()
    
    with gr.Blocks(title="🔍 Búsqueda Semántica de Productos") as app:
        gr.Markdown("""
        # 🔍 Búsqueda Semántica de Productos
        
        Esta aplicación utiliza **búsqueda semántica** para encontrar productos similares basándose en el significado
        de tu consulta, no solo en coincidencias exactas de palabras.
        
        ### ¿Cómo funciona?
        1. Escribe lo que buscas (ej: "postre con chocolate", "algo crujiente", "snack saludable")
        2. Ajusta los parámetros si lo deseas
        3. Haz clic en **Buscar** 
        """)
        
        with gr.Row():
            with gr.Column(scale=3):
                query_input = gr.Textbox(
                    label="🔎 ¿Qué estás buscando?",
                    placeholder="Ej: croissant de pistacho, dulce con frutos secos, algo para el desayuno...",
                    lines=2
                )
            
            with gr.Column(scale=1):
                search_btn = gr.Button("🚀 Buscar", variant="primary", scale=2)
        
        with gr.Row():
            with gr.Column():
                num_results = gr.Slider(
                    minimum=1,
                    maximum=50,
                    value=10,
                    step=1,
                    label="📊 Número de resultados"
                )
            
            with gr.Column():
                filter_type = gr.Dropdown(
                    choices=product_types,
                    value="Todos",
                    label="🏷️ Filtrar por tipo"
                )
            
            with gr.Column():
                min_score = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.0,
                    step=0.05,
                    label="📈 Score mínimo"
                )
        
        # Output
        status_output = gr.Textbox(label="Estado", interactive=False)
        results_output = gr.DataFrame(
            label="🛍️ Productos encontrados",
            headers=["Ranking", "Producto", "Tipo", "Precio", "Disponible", "Score", "ID"],
            datatype=["number", "str", "str", "str", "str", "str", "str"],
            col_count=(7, "fixed"),
            wrap=True
        )
        
        # Ejemplos
        gr.Examples(
            examples=[
                ["croissant de pistacho"],
                ["chocolate amargo"],
                ["postre sin azúcar"],
                ["snack crujiente"],
                ["dulce tradicional"],
                ["algo para el desayuno"],
                ["regalo gourmet"]
            ],
            inputs=query_input,
            label="💡 Ejemplos de búsqueda"
        )
        
        # Event handlers
        search_btn.click(
            fn=search_products,
            inputs=[query_input, num_results, filter_type, min_score],
            outputs=[results_output, status_output]
        )
        
        query_input.submit(
            fn=search_products,
            inputs=[query_input, num_results, filter_type, min_score],
            outputs=[results_output, status_output]
        )
        
        # Footer
        gr.Markdown("""
        ---
        ### 📌 Notas:
        - La búsqueda semántica encuentra productos basándose en el **significado**, no en palabras exactas
        - Un **score** más alto indica mayor similitud (1.0 = idéntico, 0.0 = sin relación)
        - Puedes usar descripciones naturales como "algo dulce pero no muy pesado"
        """)
    
    return app

if __name__ == "__main__":
    # Pre-cargar recursos antes de lanzar la app
    print("🚀 Iniciando aplicación...")
    try:
        load_resources()
        app = create_interface()
        app.launch(
            server_name="0.0.0.0",  # Para acceso desde otras máquinas en la red
            server_port=7860,
            share=False,  # Cambiar a True para compartir públicamente
            inbrowser=True
        )
    except Exception as e:
        print(f"❌ Error al iniciar la aplicación: {e}")
        print("Asegúrate de haber ejecutado primero el script de indexación.")