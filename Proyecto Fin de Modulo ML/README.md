# 🚀 Proyecto Final – Módulo 3: Machine Learning  

Este repositorio contiene el **proyecto final del módulo de Machine Learning** del bootcamp de Data Science.  
El objetivo principal es aplicar las técnicas aprendidas en el curso para analizar y modelar un conjunto de datos reales del ámbito financiero.  

---

## 📊 Análisis Exploratorio de Datos (EDA)  

Se trabajó con un **dataset de quejas presentadas por consumidores ante la Oficina de Protección Financiera al Consumidor (CFPB)**.  
El análisis incluyó:  
- Limpieza y preparación de datos (tratamiento de valores nulos, outliers y variables categóricas).  
- Análisis descriptivo de variables categóricas y numéricas.  
- Visualización de patrones en las quejas y en las respuestas de las compañías.  

---

## 🤖 Modelos Predictivos  

Se implementaron distintos **modelos de clasificación supervisada** con técnicas de balanceo de clases, búsqueda de hiperparámetros y validación cruzada.  

### 🔹 Modelo Predictivo 1  
Predicción de si el **consumidor disputará** la respuesta recibida por parte de la compañía.  

### 🔹 Modelo Predictivo 2  
Predicción de si la **respuesta de la compañía llegará a tiempo**.  

### 🔹 Modelo Predictivo 3  
Predicción del **tiempo de respuesta** de la compañía (clasificación en intervalos).  

---

## 🛠️ Tecnologías Utilizadas  

- **Python 3.13.2**  
- **Pandas, Numpy, Matplotlib, Seaborn** → Análisis y visualización de datos.  
- **Scikit-learn, Imbalanced-learn** → Preprocesamiento, balanceo y modelado.  
- **XGBoost** → Modelos avanzados de clasificación.  
- **Joblib** → Guardado y carga de modelos entrenados.  

---

## 📦 Instalación  

Clona el repositorio y crea un entorno con las dependencias necesarias:  

```bash
git clone https://github.com/tu-usuario/tu-repo.git
cd tu-repo

# Crear entorno con pyenv o venv
pyenv install 3.13.2
pyenv local 3.13.2

# Instalar dependencias
pip install -r requirements.txt