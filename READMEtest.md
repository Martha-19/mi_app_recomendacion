# Recrear el archivo README.md ya que el entorno se reinició

readme_content = """# 📌 Diseño e Implementación de Pruebas Unitarias

Este documento describe las pruebas unitarias implementadas para validar el funcionamiento del sistema de recomendación de productos.

## 🛠 Tecnologías Utilizadas
- Python 3.x
- Pytest
- Pandas
- Scikit-Learn

## 📌 Casos de Prueba

### ✅ **Caso de Prueba 1: Carga de Datos**
| ID | Descripción |
|----|------------|
| TC_001 | Validar que el dataset `zara.csv` se cargue correctamente sin valores críticos nulos. |

#### 🔹 **Pasos de Prueba**
1. Ejecutar `load_data()`.
2. Verificar que el DataFrame contiene las columnas esperadas.
3. Asegurar que no haya valores nulos en las columnas `name` y `description`.

#### 🔹 **Resultado Esperado**
- El dataset se carga correctamente sin errores.

---

### ✅ **Caso de Prueba 2: Generación de Recomendaciones**
| ID | Descripción |
|----|------------|
| TC_002 | Probar que la función `recommend_products_with_popularity()` genera recomendaciones adecuadas. |

#### 🔹 **Pasos de Prueba**
1. Llamar a `recommend_products_with_popularity()` con un producto de prueba.
2. Verificar que devuelve al menos `top_n` productos recomendados.
3. Confirmar que los valores `similarity_score` y `popularity_score` están dentro de los rangos esperados.

#### 🔹 **Resultado Esperado**
- La función devuelve productos relevantes según la similitud y popularidad.

---

## 🔍 **Ejecución de Pruebas**
Para ejecutar las pruebas unitarias, usa el siguiente comando en la terminal:

```bash
python -m pytest test_recomendacion.py
