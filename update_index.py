"""
Script para actualizar el índice de Pinecone con nuevos productos
Útil para agregar productos sin recrear todo el índice
"""

import pandas as pd
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import os
import json
from datetime import datetime

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
NEW_PRODUCTS_FILE = 'new_products.csv'  # Archivo con nuevos productos

def clean_dataframe(df):
    """
    Limpia valores NaN del DataFrame de manera simple y efectiva.
    """
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(0)
        elif pd.api.types.is_string_dtype(df[col]) or df[col].dtype == 'object':
            df[col] = df[col].fillna('')
    return df

def update_index():
    """Actualiza el índice con nuevos productos"""
    
    print(f"🔄 Actualizando índice con productos de {NEW_PRODUCTS_FILE}")
    
    # Cargar configuración
    with open('pinecone_config.json', 'r') as f:
        config = json.load(f)
    
    # Cargar nuevos productos
    new_df = pd.read_csv(NEW_PRODUCTS_FILE)
    print(f"📊 {len(new_df)} nuevos productos para agregar")
    
    # Limpiar datos
    print("🧹 Limpiando datos...")
    new_df = clean_dataframe(new_df)
    
    # Cargar modelo
    print("🤖 Cargando modelo de embeddings...")
    model = SentenceTransformer(config['model_name'])
    
    # Conectar a Pinecone
    print("📌 Conectando a Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(config['index_name'])
    
    # Obtener el último ID usado
    stats = index.describe_index_stats()
    last_id = stats.get('total_vector_count', 0)
    
    # Generar embeddings para nuevos productos
    print("🔄 Generando embeddings...")
    embeddings = model.encode(new_df['product_name'].tolist(), batch_size=64, show_progress_bar=True)
    new_df['embeddings'] = embeddings.tolist()
    new_df['ids'] = [str(last_id + i) for i in range(len(new_df))]
    
    # Insertar nuevos productos
    print("📤 Insertando nuevos productos...")
    batch_size = 64
    successful = 0
    
    for i in range(0, len(new_df), batch_size):
        i_end = min(i + batch_size, len(new_df))
        batch = new_df[i:i_end]
        
        ids = batch['ids'].tolist()
        emb = batch['embeddings'].tolist()
        
        # Preparar metadata
        columns_to_drop = ['ids', 'embeddings', 'product_name']
        metadata_batch = batch.drop(columns_to_drop, axis=1).to_dict('records')
        
        # Agregar product_name a metadata
        for j, m in enumerate(metadata_batch):
            m['product_name'] = batch.iloc[j]['product_name']
        
        to_upsert = list(zip(ids, emb, metadata_batch))
        
        try:
            index.upsert(vectors=to_upsert)
            successful += len(to_upsert)
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n✅ Actualización completada!")
    print(f"   - Productos agregados: {successful}")
    print(f"   - Total de productos en el índice: {stats['total_vector_count'] + successful}")
    
    # Actualizar configuración
    config['total_products'] = stats['total_vector_count'] + successful
    config['last_updated'] = datetime.now().isoformat()
    
    with open('pinecone_config.json', 'w') as f:
        json.dump(config, f, indent=2)

if __name__ == "__main__":
    update_index()