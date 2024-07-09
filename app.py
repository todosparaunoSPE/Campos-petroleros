# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 12:47:59 2024

@author: jperezr
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Datos de ejemplo de campos petroleros y punto de distribución
np.random.seed(0)
num_wells = 50
data = {
    'Pozo': [f'Pozo {i+1}' for i in range(num_wells)],
    'Latitud': np.random.uniform(18, 22, num_wells),
    'Longitud': np.random.uniform(-100, -95, num_wells),
    'Profundidad': np.random.uniform(1000, 5000, num_wells),
    'Producción': np.random.uniform(100, 10000, num_wells),
    'Presión': np.random.uniform(1000, 5000, num_wells)
}
df = pd.DataFrame(data)

# Generar historial de producción para cada pozo
historical_data = []
for i in range(num_wells):
    pozo = f'Pozo {i+1}'
    for j in range(10):
        historical_data.append({
            'Pozo': pozo,
            'Mes': j,
            'Producción': data['Producción'][i] * (1 - 0.1 * np.random.rand())
        })
historical_df = pd.DataFrame(historical_data)

# Coordenadas del punto de distribución
distribution_point = {'Latitud': 20.5, 'Longitud': -98.0}

# Función para calcular la ruta más corta utilizando NetworkX (Dijkstra)
def calculate_shortest_path(start, end):
    G = nx.Graph()
    
    # Agregar nodos para campos petroleros y punto de distribución
    for index, row in df.iterrows():
        G.add_node(row['Pozo'], pos=(row['Latitud'], row['Longitud']))
    G.add_node('Punto de Distribución', pos=(distribution_point['Latitud'], distribution_point['Longitud']))
    
    # Agregar aristas (edges) entre campos petroleros y punto de distribución
    for index, row in df.iterrows():
        G.add_edge(row['Pozo'], 'Punto de Distribución', weight=np.sqrt((row['Latitud'] - distribution_point['Latitud'])**2 + (row['Longitud'] - distribution_point['Longitud'])**2))
    
    # Calcular la ruta más corta utilizando el algoritmo de Dijkstra
    shortest_path = nx.shortest_path(G, source=start, target=end, weight='weight')
    shortest_distance = nx.shortest_path_length(G, source=start, target=end, weight='weight')
    
    return shortest_path, shortest_distance

# Función para simular datos sísmicos
def simulate_seismic_data(num_wells, amplitude_factor, depth_factor):
    depths = np.random.uniform(1000, 5000, num_wells)
    amplitude = amplitude_factor * np.random.randn(num_wells)
    depth_factor = np.linspace(0.5, 1.5, num_wells)
    seismic_data = amplitude * depth_factor
    return depths, seismic_data

# Configuración de la aplicación con Streamlit
st.title('Aplicación de Campos Petroleros y Datos Sísmicos')
st.markdown("""
<style>
    .main .block-container {
        max-width: 80%;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar con sección de ayuda
st.sidebar.header('Configuración')
st.sidebar.subheader('Ayuda')
st.sidebar.write("""
Esta aplicación permite realizar las siguientes funciones:

1. **Visualización de Campos Petroleros**:
   - Muestra un mapa interactivo con la ubicación de los pozos petroleros y el punto de distribución.

2. **Simulación de Datos Sísmicos**:
   - Genera datos sísmicos simulados basados en el número de pozos, el factor de amplitud y el factor de profundidad ajustados mediante sliders en la barra lateral.
   - Visualiza los datos sísmicos simulados en un gráfico y en un DataFrame.

3. **Historial de Producción**:
   - Permite seleccionar uno o varios pozos y muestra el historial de producción en un gráfico y en un DataFrame.

4. **Cálculo de Ruta Más Corta**:
   - Calcula y visualiza la ruta más corta desde un pozo inicial hasta un pozo de destino o el punto de distribución utilizando el algoritmo de Dijkstra.

5. **Comparación de Modelos de Regresión**:
   - Compara el rendimiento de varios modelos de regresión (Linear Regression, Decision Tree, Random Forest, Ridge Regression) para predecir la producción futura de los pozos petroleros.
""")

# Sliders para ajustar los parámetros de simulación de datos sísmicos
st.sidebar.subheader('Configuración de Datos Sísmicos')
num_wells = st.sidebar.slider('Número de Pozos', 10, 100, 50)
amplitude_factor = st.sidebar.slider('Factor de Amplitud', 0.1, 2.0, 1.0)
depth_factor = st.sidebar.slider('Factor de Profundidad', 0.5, 1.5, 1.0)

# Simular datos sísmicos
depths, seismic_data = simulate_seismic_data(num_wells, amplitude_factor, depth_factor)
df_seismic = pd.DataFrame({
    'Pozo': [f'Pozo {i+1}' for i in range(num_wells)],
    'Profundidad': depths,
    'Datos Sísmicos Simulados': seismic_data
})

# Mostrar gráfico de datos sísmicos simulados
st.header('Datos Sísmicos Simulados')
st.subheader('Gráfico de Datos Sísmicos Simulados vs Profundidad')
fig_seismic = px.scatter(df_seismic, x='Datos Sísmicos Simulados', y='Profundidad', color='Pozo', hover_name='Pozo')
st.plotly_chart(fig_seismic, use_container_width=True)

# Mostrar DataFrame de datos sísmicos simulados
st.subheader('DataFrame de Datos Sísmicos Simulados')
st.dataframe(df_seismic)

# Mostrar mapa interactivo con campos petroleros y punto de distribución
st.header('Mapa Interactivo de Campos Petroleros y Punto de Distribución')
fig = go.Figure()

# Agregar puntos de campos petroleros al mapa
fig.add_trace(go.Scattergeo(
    lon = df['Longitud'],
    lat = df['Latitud'],
    text = df['Pozo'],
    mode = 'markers',
    marker_color = df['Profundidad'],
    marker = dict(
        size = 10,
        opacity = 0.8,
        reversescale = True,
        autocolorscale = False,
        symbol = 'square',
        line = dict(
            width=1,
            color='rgba(102, 102, 102)'
        ),
        colorscale = 'Viridis',
        cmin = df['Profundidad'].min(),
        colorbar_title = 'Profundidad'
    )))

# Agregar punto de distribución al mapa
fig.add_trace(go.Scattergeo(
    lon = [distribution_point['Longitud']],
    lat = [distribution_point['Latitud']],
    text = ['Punto de Distribución'],
    mode = 'markers',
    marker = dict(
        size = 15,
        color = 'red',
        symbol = 'star'
    )))

# Configurar diseño del mapa
fig.update_geos(
    showcountries=True,
    countrycolor="Black",
    showland=True,
    showocean=True,
    oceancolor="LightBlue",
    landcolor="Tan"
)
fig.update_layout(
    title = 'Mapa Interactivo de Campos Petroleros y Punto de Distribución',
    geo = dict(
        projection_type='natural earth',
        showland = True,
        landcolor = "LightGreen",
        showocean = True,
        oceancolor = "LightBlue",
        showcountries=True,
        showlakes=True,
        lakecolor="LightBlue"
    )
)

# Mostrar el mapa interactivo
st.plotly_chart(fig, use_container_width=True)

# Selección de uno o varios pozos para calcular y mostrar el historial de producción
st.sidebar.subheader('Historial de Producción')
selected_wells = st.sidebar.multiselect('Selecciona el Pozo o los Pozos', df['Pozo'])

# Filtrar el DataFrame de historial por los pozos seleccionados
filtered_historical_df = historical_df[historical_df['Pozo'].isin(selected_wells)]

# Gráfico de historial de producción por pozo
st.subheader('Historial de Producción por Pozo')
fig_historical = px.line(filtered_historical_df, x='Mes', y='Producción', color='Pozo', title='Historial de Producción por Pozo')
st.plotly_chart(fig_historical, use_container_width=True)

# Mostrar DataFrame de historial de producción filtrado
st.subheader('DataFrame de Historial de Producción Filtrado')
st.dataframe(filtered_historical_df)

# Selección de pozo inicial y final para calcular la ruta más corta
st.sidebar.subheader('Cálculo de Ruta Más Corta')
start_well = st.sidebar.selectbox('Selecciona el Pozo de Inicio', df['Pozo'])
end_well = st.sidebar.selectbox('Selecciona el Pozo de Destino o Punto de Distribución', df['Pozo'].tolist() + ['Punto de Distribución'])

# Calcular y mostrar la ruta más corta
if st.sidebar.button('Calcular Ruta Más Corta'):
    shortest_path, shortest_distance = calculate_shortest_path(start_well, end_well)
    st.subheader('Ruta Más Corta Calculada')
    st.write(f'Ruta Más Corta desde {start_well} hasta {end_well}:')
    st.write(' -> '.join(shortest_path))
    st.write(f'Distancia Total: {shortest_distance:.2f} unidades')

# Comparación de modelos de regresión para predecir la producción futura
st.header('Comparación de Modelos de Regresión')
X = df[['Latitud', 'Longitud', 'Profundidad', 'Presión']]
y = df['Producción']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar y evaluar modelos de regresión
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'Ridge Regression': Ridge()
}
model_results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    model_results[name] = score

# Mostrar los resultados de la comparación de modelos
st.subheader('Resultados de la Comparación de Modelos de Regresión')
results_df = pd.DataFrame(list(model_results.items()), columns=['Modelo', 'Score'])
st.dataframe(results_df)

# Aviso de derechos de autor
st.sidebar.markdown("""
    ---
    © 2024. Todos los derechos reservados.
    Creado por jahoperi.
""")
