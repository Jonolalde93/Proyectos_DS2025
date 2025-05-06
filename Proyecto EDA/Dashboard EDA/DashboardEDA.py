import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy.stats import chi2_contingency, shapiro, wilcoxon, mannwhitneyu

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "stroke.csv")

dataset_stroke = pd.read_csv(csv_path)

dataset_stroke['bmi'] = dataset_stroke['bmi'].fillna(dataset_stroke['bmi'].median())

st.title("Análisis de Factores asociados al Ictus")
st.markdown("Dashboard que nos permite explorar visualmente la relación de factores tanto clínicos como sociodemográficos y la aparición de accidentes cerebrovasculares.")

vars_cuantitativas = ['age','avg_glucose_level','bmi']
vars_cualitativas = ['gender','hypertension','heart_disease','ever_married','work_type','Residence_type','smoking_status']

st.header("1. Análisis de Variables cuantitativas")
vars_cuantitativas = ["age", "avg_glucose_level", "bmi"]
for col in vars_cuantitativas:
    st.subheader(f"Distribución de {col}")
    fig, ax = plt.subplots()
    sns.histplot(data=dataset_stroke, x=col, kde=True, color="green", bins=30, ax=ax)
    st.pyplot(fig)

    st.subheader(f"Distribución de {col} por Ictus")
    fig2, ax2 = plt.subplots()
    sns.kdeplot(data=dataset_stroke, x=col, hue="stroke", fill=True, color='#00CCFF', ax=ax2)
    st.pyplot(fig2)

    st.subheader(f"Boxplot de {col} por Ictus")
    fig3, ax3 = plt.subplots()
    sns.boxplot(data=dataset_stroke, x="stroke", y=col, color='#800020', ax=ax3)
    st.pyplot(fig3)

st.header("2. Analisis de las variables categóricas")
for col in vars_cualitativas:
    st.subheader(f"{col}")
    fig, ax = plt.subplots()
    order = dataset_stroke[col].value_counts().index
    sns.countplot(data=dataset_stroke, x=col, order=order, color="purple", ax=ax)
    ax.set_title(f"Distribución de {col}")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Mostrar frecuencias y porcentajes
    freq = dataset_stroke[col].value_counts(dropna=False)
    percent = round(dataset_stroke[col].value_counts(normalize=True, dropna=False) * 100, 2)
    st.dataframe(pd.DataFrame({"Frecuencia": freq, "%": percent}))
    
    proporcion_stroke = pd.crosstab(dataset_stroke[col], dataset_stroke['stroke'], normalize='index')
    if 1 in proporcion_stroke.columns:
        fig2, ax2 = plt.subplots()
        proporcion_stroke[1].plot(kind='bar', color='orange', ax=ax2)
        ax2.set_title(f'Proporción de Ictus por {col}')
        ax2.set_ylabel('Proporción de Ictus')
        ax2.set_ylim(0, 0.2)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
        for p in ax2.patches:
            height = p.get_height()
            ax2.annotate(f'{height:.2%}', (p.get_x() + p.get_width() / 2., height),
                         ha='center', va='bottom', fontsize=8)

        st.pyplot(fig2)
    else:
        st.info(f"No hay casos de ictus registrados para las categorías en {col}.")

st.header("3. Matriz de Correlación")
columnas_numericas = dataset_stroke.select_dtypes(include=['float64', 'int64']).columns
columnas_numericas = columnas_numericas.drop('id')
corr_matrix = dataset_stroke[columnas_numericas].corr()
fig_corr, ax_corr = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax_corr, fmt=".2f", annot_kws={"size": 10}, cbar=True)
ax_corr.set_title("Matriz de Correlación entre Variables")
st.pyplot(fig_corr)

st.header("4. Pruebas de Independencia Chi-Cuadrado para las variables cualitativas")
for col in vars_cualitativas:
    st.subheader(f"Chi-cuadrado: stroke vs {col}")
    contingency = pd.crosstab(dataset_stroke["stroke"], dataset_stroke[col])
    chi2, p, dof, expected = chi2_contingency(contingency)
    st.write(f"p-valor: {p:.4g}")

st.header("5. Pruebas de Normalidad y Comparación para las variables cuantitativas")
for col in vars_cuantitativas:
    st.subheader(f"{col}")
    group0 = dataset_stroke[dataset_stroke.stroke == 0][col].dropna()
    group1 = dataset_stroke[dataset_stroke.stroke == 1][col].dropna()

    st.write("**Prueba de normalidad (Shapiro-Wilk)**")
    p0 = shapiro(group0)[1]
    p1 = shapiro(group1)[1]
    st.write(f"Grupo sin ictus: p-valor = {p0:.2e}")
    st.write(f"Grupo con ictus: p-valor = {p1:.2e}")

    st.write("**Prueba de Wilcoxon/Mann-Whitney**")
    p_wilcoxon = mannwhitneyu(group0, group1, alternative="two-sided")[1]
    st.write(f"p-valor: {p_wilcoxon:.2e}")