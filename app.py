import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

# Configuración de la página
st.set_page_config(page_title="Sistema de Recomendación de Productos ZARA", page_icon="🛍️", layout="centered")

# Información 
st.markdown("""
    <div style='text-align: center; color: #333333; font-size: 20px;'> 
        <strong>INTELIGENCIA EN CIENCIA DE DATOS</strong><br>
        <em>Martha Espinal - Matrícula: 24-1430</em>
    </div>
    <br>
""", unsafe_allow_html=True)

# Título de la app
st.markdown("<h1 style='text-align: center; color: #FF5A5F;'>Sistema de Recomendación de Productos ZARA 🛒</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #808080;'>Descubre los productos que podrían interesarte</p>", unsafe_allow_html=True)

# Descripción introductoria
st.markdown("""
    <div style='text-align: center; color: #333333; font-size: 18px;'>
        Este sistema utiliza un modelo de recomendación que combina la similitud de contenido 
        con la popularidad del producto para ofrecerte las mejores opciones.
    </div>
    <br>
""", unsafe_allow_html=True)

# Cargar el conjunto de datos
@st.cache_data
def load_data():
    data = pd.read_csv("zara.csv", delimiter=';')
    return data

data = load_data()

# Normalizar 'Sales Volume' para obtener un puntaje de popularidad
scaler = MinMaxScaler()
data['popularity_score'] = scaler.fit_transform(data[['Sales Volume']])

# Selección del producto
st.subheader("🔎 Selecciona un producto para ver las recomendaciones")
product_list = data['name'].dropna().unique()
selected_product = st.selectbox("Producto", product_list)

# Control deslizante para ajustar la popularidad
st.markdown("<h5 style='color: #333333;'>💡 Ajusta la influencia de la popularidad en las recomendaciones:</h5>", unsafe_allow_html=True)
popularity_weight = st.slider("", 0.0, 1.0, 0.5)

# Función de recomendación
def recommend_products(selected_product, data, top_n=5, popularity_weight=0.5):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(data['description'].fillna(''))
    similarity_matrix = cosine_similarity(tfidf_matrix)

    product_index = data[data['name'] == selected_product].index[0]
    similarity_scores = list(enumerate(similarity_matrix[product_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    recommended_indices = [i[0] for i in similarity_scores[1:top_n*2+1]]
    
    # Calcular popularidad y similitud en un solo puntaje
    recommended_products = data.iloc[recommended_indices].copy()
    recommended_products['similarity_score'] = [similarity_scores[i][1] for i in range(1, top_n*2+1)]
    recommended_products['combined_score'] = (recommended_products['similarity_score'] * (1 - popularity_weight) +
                                              recommended_products['popularity_score'] * popularity_weight)

    final_recommendations = recommended_products.sort_values(by='combined_score', ascending=False).head(top_n)
    return final_recommendations[['name', 'Sales Volume', 'price', 'description', 'popularity_score', 'similarity_score', 'combined_score']]

# Definir relevancia
popularity_threshold = 0.7
similarity_threshold = 0.5

def is_relevant(row):
    return row['popularity_score'] >= popularity_threshold and row['similarity_score'] >= similarity_threshold

def evaluate_recommendations(selected_product, data, top_n=5, popularity_weight=0.5):
    recommendations = recommend_products(selected_product, data, top_n=top_n, popularity_weight=popularity_weight)
    recommendations['is_relevant'] = recommendations.apply(is_relevant, axis=1)
    precision = recommendations['is_relevant'].sum() / top_n
    return precision, recommendations

# Mostrar recomendaciones y precisión
if selected_product:
    st.subheader(f"📋 Recomendaciones para: **{selected_product}**")
    recommendations = recommend_products(selected_product, data, top_n=5, popularity_weight=popularity_weight)
    
    # Estilo de las recomendaciones
    st.write(recommendations.style.set_properties(**{
        'background-color': '#F7F9FA', 
        'color': '#333333', 
        'border-color': '#FF5A5F'
    }))

    # Mostrar precisión
    precision, detailed_recommendations = evaluate_recommendations(selected_product, data, top_n=5, popularity_weight=popularity_weight)
    st.markdown(f"<h5 style='color: #333333;'>🎯 Precisión en el Top-5: {precision:.2f}</h5>", unsafe_allow_html=True)
    
    st.subheader("📜 Detalles de las recomendaciones:")
    st.write(detailed_recommendations)

# Muestra de la matriz TF-IDF
st.subheader("📊 Muestra de la matriz TF-IDF (primeras 5 filas, 10 columnas)")
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(data['description'].fillna(''))
tfidf_sample_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
st.write(tfidf_sample_df.iloc[:5, :10])

# Productos similares según la descripción
st.subheader("🧩 Productos similares según la descripción")
product_index = data[data['name'] == selected_product].index[0]
similarity_scores = list(enumerate(cosine_similarity(tfidf_matrix)[product_index]))
similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

# Mostrar los 5 productos más similares
for i, (idx, score) in enumerate(similarity_scores[1:6]):
    st.markdown(f"<h6>{i+1}. Producto índice {idx} - Similaridad: {score:.4f}</h6>", unsafe_allow_html=True)
    st.markdown(f"**Nombre**: {data.iloc[idx]['name']}")
    st.markdown(f"**Descripción**: {data.iloc[idx]['description']}")
    st.write(" ")


