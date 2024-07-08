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
from geopy.distance import geodesic

# Datos de ejemplo de campos petroleros y punto de distribución
np.random.seed(0)
num_wells = 50
data = {
    'Pozo': [f'Pozo {i+1}' for i in range(num_wells)],
    'Latitud': np.random.uniform(18, 22, num_wells),
    'Longitud': np.random.uniform(-100, -95, num_wells),
    'Profundidad': np.random.uniform(1000, 5000, num_wells),
    'Producción': np.random.uniform(100, 10000, num_wells)
}
df = pd.DataFrame(data)

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

# Sidebar con sección de ayuda
st.sidebar.subheader('Ayuda')
st.sidebar.write('Esta aplicación permite visualizar campos petroleros, simular datos sísmicos, calcular rutas de transporte y mostrar métricas de rendimiento.')

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
st.subheader('Gráfico de Datos Sísmicos Simulados vs Profundidad')
fig_seismic = px.scatter(df_seismic, x='Datos Sísmicos Simulados', y='Profundidad', color='Pozo', hover_name='Pozo')
st.plotly_chart(fig_seismic, use_container_width=True)

# Mostrar DataFrame de datos sísmicos simulados
st.subheader('DataFrame de Datos Sísmicos Simulados')
st.write(df_seismic)

# Mostrar mapa interactivo con campos petroleros y punto de distribución
st.subheader('Mapa Interactivo de Campos Petroleros y Punto de Distribución')
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

# Selección de inicio y fin para calcular la ruta más corta
start_wells = st.multiselect('Selecciona los Pozos de Inicio para Calcular Ruta', df['Pozo'])
end_well = 'Punto de Distribución'  # Punto de distribución fijo para ejemplo

# Calcular ruta más corta y mostrar resultados
shortest_paths = []
total_distances = []

for start_well in start_wells:
    shortest_path, shortest_distance = calculate_shortest_path(start_well, end_well)
    shortest_paths.append(shortest_path)
    total_distances.append(shortest_distance)

# Crear DataFrame para la ruta más corta y detalles
path_details = []

# Agregar detalles de los pozos de inicio seleccionados
for i, start_well in enumerate(start_wells):
    if start_well != 'Punto de Distribución':
        details_start = df[df['Pozo'] == start_well].iloc[0]
        distance_to_distribution = geodesic((details_start['Latitud'], details_start['Longitud']), (distribution_point['Latitud'], distribution_point['Longitud'])).kilometers
        path_details.append({
            'Nombre': start_well,
            'Latitud': details_start['Latitud'],
            'Longitud': details_start['Longitud'],
            'Distancia al Punto de Distribución (km)': distance_to_distribution,
            'Costo de Transporte ($USD)': distance_to_distribution * 10  # Ejemplo de costo de transporte
        })

# Agregar punto de distribución
path_details.append({
    'Nombre': 'Punto de Distribución',
    'Latitud': distribution_point['Latitud'],
    'Longitud': distribution_point['Longitud'],
    'Distancia al Punto de Distribución (km)': 0.0,
    'Costo de Transporte ($USD)': 0.0
})

# Convertir a DataFrame
df_path = pd.DataFrame(path_details)

# Mostrar DataFrame con la ruta más corta y detalles
st.subheader('Ruta Más Corta de Transporte y Detalles del Pozo y Punto de Distribución')
st.write(df_path)

# Métricas de rendimiento de la ruta más corta
st.subheader('Métricas de Rendimiento de la Ruta Más Corta')

if shortest_paths:
    st.write(f'Distancia Total de la Ruta Más Corta: {sum(total_distances):.2f} km')
    st.write(f'Número de Pozos en la Ruta Más Corta: {len(shortest_paths[0]) - 1}')
else:
    st.write('Selecciona al menos un pozo de inicio para calcular la ruta más corta.')

# Nota final
st.sidebar.markdown('---')
st.sidebar.subheader('Nota')
st.sidebar.write("""
Esta aplicación es un ejemplo educativo para visualizar campos petroleros, simular datos sísmicos, calcular rutas de transporte y mostrar métricas de rendimiento utilizando Streamlit y Plotly.
""")
