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
def simulate_seismic_data(num_wells, amplitude_factor, depth_factor):
    depths = np.random.uniform(1000, 5000, num_wells)
    amplitude = amplitude_factor * np.random.randn(num_wells)
    depth_factor = np.linspace(0.5, 1.5, num_wells)
    seismic_data = amplitude * depth_factor
    return depths, seismic_data

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

# Mostrar la ruta más corta en el mapa interactivo
if shortest_paths:
    fig_shortest_path = go.Figure(fig)
    for path in shortest_paths:
        path_coords = []
        for node in path:
            if node in df['Pozo'].values:
                coords = (df.loc[df['Pozo'] == node, 'Longitud'].values[0], df.loc[df['Pozo'] == node, 'Latitud'].values[0])
                path_coords.append(coords)
            else:
                st.warning(f'El pozo {node} no se encuentra en el DataFrame.')
                # Puedes decidir qué hacer en caso de que el nodo no exista en df
        
        path_coords.append((distribution_point['Longitud'], distribution_point['Latitud']))
        
        fig_shortest_path.add_trace(go.Scattergeo(
            lon = [coord[0] for coord in path_coords],
            lat = [coord[1] for coord in path_coords],
            mode = 'lines',
            line = dict(width = 2, color = 'blue'),
            name = f'Ruta Más Corta: {" -> ".join(path)}'
        ))

    fig_shortest_path.update_layout(title = 'Ruta Más Corta de Transporte en el Mapa Interactivo', showlegend=True)
    st.plotly_chart(fig_shortest_path, use_container_width=True)

# Métricas de rendimiento
if len(start_wells) > 0:
    total_distance = df_path['Distancia al Punto de Distribución (km)'].sum()
    st.subheader('Métricas de Rendimiento de la Ruta Más Corta')
    st.write(f'Distancia Total de la Ruta Más Corta: {total_distance:.2f} km')
    st.write(f'Número de Pozos en la Ruta Más Corta: {len(shortest_paths[0]) - 1}')

# Créditos y referencia
st.sidebar.markdown('---')
st.sidebar.subheader('Créditos y Referencia')
st.sidebar.write("""
- Desarrollado por: Javier Horacio Pérez Ricárdez
- Contacto: +52 55 7425 5593
""")

# Información adicional
st.sidebar.markdown('---')
st.sidebar.subheader('Información Adicional')
st.sidebar.write("""
Esta aplicación es un prototipo para visualizar campos petroleros y simular datos sísmicos. 
Se utiliza NetworkX para calcular rutas de transporte y Plotly para gráficos interactivos.
""")
