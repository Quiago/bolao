# 🔍 Sistema de Búsqueda Semántica de Productos

Este proyecto implementa un sistema de búsqueda semántica para productos usando Pinecone y Gradio.

## 🚀 Características

- **Búsqueda semántica**: Encuentra productos por significado, no solo por palabras exactas
- **Interfaz web amigable**: Aplicación Gradio fácil de usar
- **Filtros avanzados**: Filtra por tipo de producto y score mínimo
- **Rápido y escalable**: Usa Pinecone para búsquedas vectoriales eficientes
- **Actualizable**: Agrega nuevos productos sin recrear todo el índice

## 📋 Requisitos

- Python 3.9+
- Cuenta gratuita en [Pinecone](https://www.pinecone.io/)
- API Key de Pinecone

## 🛠️ Instalación

1. **Clona o descarga este proyecto**

2. **Instala las dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configura tu API Key de Pinecone**:
   ```bash
   export PINECONE_API_KEY="tu-api-key-aqui"
   ```

## 📊 Preparación de datos

Tu archivo `product.csv` debe tener al menos las siguientes columnas:
- `product_name`: Nombre del producto (requerido)
- `type`: Tipo/categoría del producto
- `price`: Precio
- `available`: Disponibilidad (boolean)
- Otras columnas opcionales según tus necesidades

## 🏃‍♂️ Uso

### 1. Indexar productos (primera vez)

Ejecuta este script una sola vez para crear el índice y cargar todos los productos:

```bash
python pinecone_indexer.py
```

Esto:
- Crea un índice en Pinecone
- Genera embeddings para todos los productos
- Los sube a Pinecone
- Guarda la configuración en `pinecone_config.json`

### 2. Lanzar la aplicación web

```bash
python gradio_search_app.py
```

La aplicación se abrirá automáticamente en tu navegador. Si no, ve a: http://localhost:7860

### 3. Actualizar con nuevos productos (opcional)

Si tienes nuevos productos para agregar:

1. Crea un archivo `new_products.csv` con el mismo formato
2. Ejecuta:
   ```bash
   python update_index.py
   ```

## 🎯 Ejemplos de búsqueda

La búsqueda semántica entiende el significado, así que puedes buscar:

- ✅ "croissant de pistacho"
- ✅ "algo dulce para el desayuno"
- ✅ "postre sin azúcar"
- ✅ "snack crujiente y saludable"
- ✅ "regalo gourmet elegante"

## 🔧 Personalización

### Cambiar el modelo de embeddings

En `pinecone_indexer.py`, cambia:
```python
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
```

Por cualquier modelo de [Sentence Transformers](https://www.sbert.net/docs/pretrained_models.html).

### Ajustar la interfaz

Modifica `gradio_search_app.py` para:
- Cambiar colores/temas
- Agregar más filtros
- Modificar la visualización de resultados

## 📈 Rendimiento

- **Indexación inicial**: ~1-2 minutos por cada 10,000 productos
- **Búsquedas**: <0.5 segundos típicamente
- **Límites plan gratuito**: 
  - 5 índices máximo
  - 2GB de almacenamiento
  - Solo región us-east-1

## 🐛 Solución de problemas

### Error: "Index not found"
- Asegúrate de haber ejecutado `pinecone_indexer.py` primero

### Error: "API Key not valid"
- Verifica tu PINECONE_API_KEY
- Asegúrate de que esté configurada como variable de entorno

### Valores NaN en datos
- El script automáticamente limpia valores NaN
- Revisa los logs durante la indexación para ver qué columnas tienen problemas

## 📚 Recursos adicionales

- [Documentación de Pinecone](https://docs.pinecone.io/)
- [Documentación de Grad