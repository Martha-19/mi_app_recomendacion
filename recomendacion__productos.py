# Importar las librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Paso 1: Cargar y limpiar los datos
file_path_zara = 'mi_app/zara.csv'
zara_data = pd.read_csv(file_path_zara, delimiter=';')

# Revisar valores nulos y tipos de datos
zara_data_nulls = zara_data.isnull().sum()
zara_data_dtypes = zara_data.dtypes
print("Valores nulos por columna al cargar los datos:\n", zara_data_nulls)
print("\nTipos de datos de cada columna:\n", zara_data_dtypes)
print("\nPrimeras filas del dataset:\n", zara_data.head())

# Paso 2: Análisis Exploratorio de Datos (EDA) - Graficar distribuciones
print("\n--- Análisis Exploratorio de Datos (EDA) ---")
plt.figure(figsize=(10, 6))
plt.hist(zara_data['Sales Volume'], bins=30, edgecolor='black')
plt.title('Distribución de Volumen de Ventas')
plt.xlabel('Volumen de Ventas')
plt.ylabel('Frecuencia')
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(zara_data['price'], bins=30, edgecolor='black')
plt.title('Distribución de Precios (USD)')
plt.xlabel('Precio (USD)')
plt.ylabel('Frecuencia')
plt.show()

plt.figure(figsize=(10, 6))
position_counts = zara_data['Product Position'].value_counts()
plt.bar(position_counts.index, position_counts.values, color='skyblue', edgecolor='black')
plt.title('Distribución de Posición de Productos en Tienda')
plt.xlabel('Posición del Producto')
plt.ylabel('Frecuencia')
plt.show()

plt.figure(figsize=(10, 6))
promotion_counts = zara_data['Promotion'].value_counts()
plt.bar(promotion_counts.index, promotion_counts.values, color='orange', edgecolor='black')
plt.title('Distribución de Promociones')
plt.xlabel('Promoción')
plt.ylabel('Frecuencia')
plt.show()

plt.figure(figsize=(10, 6))
category_counts = zara_data['Product Category'].value_counts()
plt.bar(category_counts.index, category_counts.values, color='green', edgecolor='black')
plt.title('Distribución de Categorías de Productos')
plt.xlabel('Categoría del Producto')
plt.ylabel('Frecuencia')
plt.xticks(rotation=45)
plt.show()

# Normalizar 'Sales Volume' para crear una métrica de popularidad
scaler = MinMaxScaler()
zara_data['popularity_score'] = scaler.fit_transform(zara_data[['Sales Volume']])
print("\nValores de 'popularity_score' después de la normalización:\n", zara_data[['Sales Volume', 'popularity_score']].head())

# Mostrar los productos más populares
top_popular_products = zara_data.sort_values(by='popularity_score', ascending=False)[['name', 'Sales Volume', 'popularity_score']].head(10)
print("\nRanking de los productos más populares:")
print(top_popular_products)

# Procesamiento de las descripciones de productos con TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(zara_data['description'].fillna(''))
tfidf_sample_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
print("\nMuestra de la matriz TF-IDF (primeras 5 filas, 10 columnas):")
print(tfidf_sample_df.iloc[:5, :10])

# Matriz de similitud de coseno
similarity_matrix = cosine_similarity(tfidf_matrix)

# Función de recomendación que incorpora la popularidad
def recommend_products_with_popularity(product_index, similarity_matrix, df, top_n=5, popularity_weight=0.5):
    similarity_scores = list(enumerate(similarity_matrix[product_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    recommended_indices = [i[0] for i in similarity_scores[1:top_n*2+1]]
    recommended_products = df.iloc[recommended_indices].copy()
    recommended_products['similarity_score'] = [similarity_scores[i][1] for i in range(1, top_n*2+1)]
    recommended_products['combined_score'] = (recommended_products['similarity_score'] * (1 - popularity_weight) +
                                              recommended_products['popularity_score'] * popularity_weight)
    final_recommendations = recommended_products.sort_values(by='combined_score', ascending=False).head(top_n)
    return final_recommendations[['name', 'Sales Volume', 'popularity_score', 'similarity_score', 'combined_score']]

recommendations = recommend_products_with_popularity(0, similarity_matrix, zara_data)
print("Recomendaciones para el primer producto:")
print(recommendations)

# Definir los umbrales de relevancia
popularity_threshold = 0.7
similarity_threshold = 0.5

def is_relevant(row):
    return row['popularity_score'] >= popularity_threshold and row['similarity_score'] >= similarity_threshold

def evaluate_recommendations(product_index, similarity_matrix, df, top_n=5):
    recommendations = recommend_products_with_popularity(product_index, similarity_matrix, df, top_n=top_n)
    recommendations['is_relevant'] = recommendations.apply(is_relevant, axis=1)
    precision = recommendations['is_relevant'].sum() / top_n
    return precision, recommendations[['name', 'Sales Volume', 'popularity_score', 'similarity_score', 'combined_score', 'is_relevant']]

# Evaluación de ejemplo para el primer producto en el conjunto de datos
product_index = 0
precision, detailed_recommendations = evaluate_recommendations(product_index, similarity_matrix, zara_data)

# Mostrar precisión y detalles de las recomendaciones
print(f"Precisión en el Top-{5} para el producto índice {product_index}: {precision:.2f}")
print("Detalles de las recomendaciones:")
print(detailed_recommendations)
