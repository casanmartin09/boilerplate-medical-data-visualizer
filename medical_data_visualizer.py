import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Cargar el conjunto de datos médicos
df = pd.read_csv('medical_examination.csv')

# 2. Agregar columna 'overweight' basada en IMC > 25
bmi = df['weight'] / ((df['height'] / 100) ** 2)
df['overweight'] = (bmi > 25).astype(int)

# 3. Normalizar variables clínicas (0: óptimo, 1: sobre lo normal)
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

# 4. Función de graficación categórica
def draw_cat_plot():
    # 5. Transformar a formato tidy/long con pd.melt
    df_cat = pd.melt(
        df,
        id_vars=['cardio'],
        value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']
    )

    # 6. Agrupar y obtener frecuencias por cardio, variable y valor
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    # 7. Graficar distribución con sns.catplot
    fig_cat = sns.catplot(
        data=df_cat,
        x='variable',
        y='total',
        hue='value',
        col='cardio',
        kind='bar'
    )

    # 8. Extraer objeto de figura
    fig = fig_cat.fig

    # 9. Guardar y retornar
    fig.savefig('catplot.png')
    return fig

# 10. Función de mapa de calor
def draw_heat_map():
    # 11. Filtrado de registros anómalos e inconsistencias fisiológicas
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12. Matriz de correlación de Pearson
    corr = df_heat.corr()

    # 13. Máscara booleana para el triángulo superior
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14. Configurar lienzo gráfico
    fig, ax = plt.subplots(figsize=(12, 10))

    # 15. Renderizado del mapa de calor
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt='.1f',
        center=0,
        vmin=-0.16,
        vmax=0.32,
        cbar_kws={'shrink': 0.5},
        square=True,
        linewidths=0.5,
        ax=ax
    )

    # 16. Guardar y retornar
    fig.savefig('heatmap.png')
    return fig
