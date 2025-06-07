"""
Script para indexar productos en Pinecone
Ejecutar este script una sola vez para crear el índice y cargar los datos
"""

import pandas as pd
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import os
from tqdm.auto import tqdm
import json
from datetime import datetime

# Configuración
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
INDEX_NAME = 'products-embeddings'
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
CSV_FILE = 'product.csv'

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

def create_index():
    """Crea o actualiza el índice de Pinecone con los productos"""
    
    print(f"🚀 Iniciando indexación de productos desde {CSV_FILE}")
    
    # Cargar los datos
    print("📊 Cargando datos...")
    df = pd.read_csv(CSV_FILE)
    print(f"   - {len(df)} productos cargados")
    
    # Limpiar datos
    print("\n🧹 Limpiando datos...")
    df = clean_dataframe(df)
    
    # Mostrar información sobre NaN (después de limpiar)
    print("\n🔍 Verificando calidad de datos después de limpieza:")
    nan_count_total = 0
    for col in df.columns:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            print(f"   - {col}: {nan_count} valores NaN restantes")
            nan_count_total += nan_count
    if nan_count_total == 0:
        print("   - ✅ No hay valores NaN en el dataset")
    
    # Inicializar el modelo de embeddings
    print("\n🤖 Cargando modelo de embeddings...")
    model = SentenceTransformer(MODEL_NAME)
    
    # Generar embeddings
    print("\n🔄 Generando embeddings...")
    embeddings = model.encode(df['product_name'].tolist(), batch_size=64, show_progress_bar=True)
    df['embeddings'] = embeddings.tolist()
    df['ids'] = df.index.astype('str')
    
    # Inicializar Pinecone
    print("\n📌 Conectando a Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Verificar dimensiones
    dimensions = len(df['embeddings'][0])
    print(f"   - Dimensiones de embeddings: {dimensions}")
    
    # Verificar si el índice existe
    existing_indexes = pc.list_indexes().names()
    
    if INDEX_NAME not in existing_indexes:
        print(f"\n✨ Creando índice '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=dimensions,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print("   - Índice creado exitosamente")
    else:
        print(f"\n✅ El índice '{INDEX_NAME}' ya existe")
    
    # Obtener referencia al índice
    index = pc.Index(INDEX_NAME)
    
    # Insertar datos
    print("\n📤 Insertando datos en Pinecone...")
    batch_size = 64
    successful_inserts = 0
    failed_inserts = 0
    
    for i in tqdm(range(0, len(df), batch_size)):
        i_end = min(i + batch_size, len(df))
        batch = df[i:i_end]
        
        ids = batch['ids'].tolist()
        emb = batch['embeddings'].tolist()
        
        # Preparar metadata
        columns_to_drop = ['ids', 'embeddings', 'product_name']
        if 'text' in batch.columns:
            columns_to_drop.append('text')
        if 'path' in batch.columns:
            columns_to_drop.append('path')
        
        metadata_batch = batch.drop(columns_to_drop, axis=1).to_dict('records')
        
        # Agregar product_name a la metadata para mostrar en resultados
        for j, m in enumerate(metadata_batch):
            m['product_name'] = batch.iloc[j]['product_name']
        
        to_upsert = list(zip(ids, emb, metadata_batch))
        
        try:
            index.upsert(vectors=to_upsert)
            successful_inserts += len(to_upsert)
        except Exception as e:
            print(f"\n❌ Error en batch {i}-{i_end}: {e}")
            failed_inserts += len(to_upsert)
    
    # Verificar estadísticas
    print("\n📊 Estadísticas del índice:")
    stats = index.describe_index_stats()
    print(f"   - Total de vectores: {stats['total_vector_count']}")
    print(f"   - Inserciones exitosas: {successful_inserts}")
    print(f"   - Inserciones fallidas: {failed_inserts}")
    
    # Guardar configuración
    config = {
        'index_name': INDEX_NAME,
        'model_name': MODEL_NAME,
        'dimensions': dimensions,
        'total_products': len(df),
        'indexed_at': datetime.now().isoformat(),
        'columns': list(df.columns)
    }
    
    with open('pinecone_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n✅ Indexación completada!")
    print("   - Configuración guardada en pinecone_config.json")
    
    # Hacer una consulta de prueba
    print("\n🧪 Realizando consulta de prueba...")
    test_query = "chocolate"
    query_vector = model.encode(test_query).tolist()
    results = index.query(vector=query_vector, top_k=3, include_metadata=True)
    
    print(f"   Resultados para '{test_query}':")
    for match in results['matches']:
        print(f"   - {match['metadata'].get('product_name', 'N/A')} (score: {match['score']:.3f})")

if __name__ == "__main__":
    create_index()