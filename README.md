# Descripción del Proyecto
Este repositorio contiene el código fuente de un sistema de recomendación de productos desarrollado como parte de la Práctica Integradora de Ciencia de Datos en la asignatura de Inteligencia en Ciencia de Datos. El sistema utiliza técnicas de procesamiento de lenguaje natural (NLP) y métodos de recomendación basados en contenido para sugerir productos similares, optimizando la experiencia de compra en una tienda en línea.

# Objetivo
Desarrollar un modelo de recomendación de productos para una empresa de comercio electrónico, con el fin de mejorar la experiencia del usuario mediante sugerencias personalizadas basadas en productos similares. La solución considera la popularidad del producto y la similitud de su descripción.

# Características Principales: 
- Análisis Exploratorio de Datos (EDA): Se exploran tendencias y patrones clave del dataset, incluyendo análisis de frecuencia y estadísticas descriptivas.
- Preprocesamiento de Datos: Se aplican técnicas de NLP mediante el uso de TF-IDF para procesar las descripciones de los productos.
- Modelo de Recomendación: Modelo basado en la similitud de contenido utilizando cosine similarity y el score de popularidad.
- Interfaz de Usuario: Interfaz intuitiva desarrollada con Streamlit para interactuar con el modelo, seleccionar productos y recibir recomendaciones.
- Despliegue en la Nube: Configuración para implementar el modelo y la interfaz de usuario en un servicio en la nube.
  
# Contenidos
app.py: Código principal de la aplicación en Streamlit.
zara.csv: Dataset de productos.
requirements.txt: Dependencias del proyecto.
README.md: Descripción general del proyecto (este archivo).

# Instrucciones
Clona este repositorio.
Instala las dependencias ejecutando pip install -r requirements.txt.
Inicia la aplicación con streamlit run app.py.

