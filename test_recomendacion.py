import pytest
import pandas as pd
import numpy as np
from recomendacion__productos import recommend_products_with_popularity, evaluate_recommendations
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Datos de prueba simulados
@pytest.fixture
def sample_data():
    data = pd.DataFrame({
        'name': ['Producto A', 'Producto B', 'Producto C'],
        'description': ['Zapatos de cuero', 'Camiseta de algodón', 'Pantalón de mezclilla'],
        'Sales Volume': [100, 50, 200]
    })
    scaler = MinMaxScaler()
    data['popularity_score'] = scaler.fit_transform(data[['Sales Volume']])
    return data

@pytest.fixture
def similarity_matrix(sample_data):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sample_data['description'].fillna(''))
    return cosine_similarity(tfidf_matrix)

def test_recommend_products(sample_data, similarity_matrix):
    recommendations = recommend_products_with_popularity(0, similarity_matrix, sample_data, top_n=2)
    assert len(recommendations) == 2, "Debe devolver exactamente 2 recomendaciones" 
    assert 'name' in recommendations.columns, "El dataframe debe incluir la columna 'name'"

def test_evaluate_recommendations(sample_data, similarity_matrix):
    precision, recommendations = evaluate_recommendations(0, similarity_matrix, sample_data, top_n=2)
    assert 0 <= precision <= 1, "La precisión debe estar entre 0 y 1"
    assert 'is_relevant' in recommendations.columns, "Debe incluir la columna 'is_relevant'"





    
