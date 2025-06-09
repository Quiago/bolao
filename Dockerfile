# Usa una imagen base de Python oficial y ligera
FROM python:3.12-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia primero el archivo de requisitos para aprovechar el caché de Docker
COPY requirements.txt .

# Instala las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto de los archivos de tu proyecto al contenedor
COPY . .

# Expone el puerto que Gunicorn usará (esto es para documentación, no es obligatorio)
EXPOSE 7860

# El comando que se ejecutará cuando el contenedor inicie
# Gunicorn se enlazará automáticamente al puerto que App Runner le asigne
CMD ["python", "app_card.py"]