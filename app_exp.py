import pandas as pd
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import os
from tqdm.auto import tqdm

# Cargar los datos
df = pd.read_csv('product.csv')

# Limpiar datos antes de procesar
# Fill NaN values with appropriate defaults based on column type
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(0)
    elif pd.api.types.is_string_dtype(df[col]) or df[col].dtype == 'object':
        df[col] = df[col].fillna('')
# Opcionalmente, puedes llenar NaN con valores específicos según la columna
# Por ejemplo:
# df['pickup'] = df['pickup'].fillna(False)  # Si pickup es booleano
# df['phone'] = df['phone'].fillna('')  # Si phone es string
# O puedes eliminar filas con NaN en columnas críticas:
# df = df.dropna(subset=['product_name'])

# Inicializar el modelo de embeddings
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Generar embeddings
embeddings = model.encode(df['product_name'], batch_size=64, show_progress_bar=True)
df['embeddings'] = embeddings.tolist()
df['ids'] = df.index.astype('str')

# Inicializar Pinecone con la nueva API
pinecone_api = os.getenv('PINECONE_API_KEY', "test")
pc = Pinecone(api_key=pinecone_api)

# Definir las dimensiones de los embeddings
dimensions_embeddings = len(df['embeddings'][0])
index_name = 'products-embeddings'

# Verificar si el índice existe y crearlo si no existe
existing_indexes = pc.list_indexes().names()

if index_name not in existing_indexes:
    # Crear índice en la versión gratuita (solo disponible en us-east-1)
    pc.create_index(
        name=index_name,
        dimension=dimensions_embeddings,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"  # La versión gratuita solo permite esta región
        )
    )
    print(f"Índice '{index_name}' creado exitosamente")
else:
    print(f"El índice '{index_name}' ya existe")

# Obtener referencia al índice
index = pc.Index(index_name)

# Preparar los datos para la inserción
batch_size = 64


for i in tqdm(range(0, len(df), batch_size)):
    # Encontrar el final del batch
    i_end = min(i + batch_size, len(df))
    # Extraer el batch
    batch = df[i:i_end]
    
    # Obtener ids y embeddings
    ids = batch['ids'].tolist()
    emb = batch['embeddings'].tolist()
    
    # Preparar metadata (eliminar columnas que no queremos incluir)
    columns_to_drop = ['ids', 'embeddings', 'product_name']  # product_name ya está en embeddings
    if 'text' in batch.columns:
        columns_to_drop.append('text')
    if 'path' in batch.columns:
        columns_to_drop.append('path')
    
    # Crear metadata y limpiar valores NaN
    metadata_batch = batch.drop(columns_to_drop, axis=1).to_dict('records')
    
    # Crear lista de tuplas para upsert
    to_upsert = list(zip(ids, emb, metadata_batch))
    
    # Insertar/actualizar los registros en Pinecone
    try:
        index.upsert(vectors=to_upsert)
    except Exception as e:
        print(f"Error en batch {i}-{i_end}: {e}")
        # Opcionalmente, puedes intentar insertar registro por registro
        for record in to_upsert:
            try:
                index.upsert(vectors=[record])
            except Exception as e2:
                print(f"Error insertando registro {record[0]}: {e2}")

# Verificar las estadísticas del índice
print("\nEstadísticas del índice:")
stats = index.describe_index_stats()
print(stats)

# Realizar una consulta de ejemplo
query = 'croissant de pistaccio'
query_vector = model.encode(query).tolist()

# Realizar la búsqueda con filtro
responses = index.query(
    vector=query_vector,
    top_k=10,
    include_metadata=True,
    filter={
        "type": {"$in": ['dulcerias']}
    }
)

print("\nResultados de la consulta:")
for match in responses['matches']:
    print(f"ID: {match['id']}, Score: {match['score']:.4f}")
    if 'metadata' in match:
        print(f"Metadata: {match['metadata']}")
    print("---")