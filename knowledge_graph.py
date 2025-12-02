"""
NCLEX Item Bank Knowledge Graph Visualization
Visualizes embeddings from PostgreSQL database using dimensionality reduction and interactive graphs
"""

import streamlit as st
import psycopg2
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import umap
import networkx as nx
from scipy.spatial.distance import cosine
import json
import os

# ==================== Configuration ====================
DB_CONFIG = {
    "dbname": "pgvector",
    "user": "postgres",
    "password": "pgvector",
    "host": "localhost",
    "port": 5432
}

EMBEDDING_DIM = 384

# ==================== Helper Functions ====================

@st.cache_data
def load_embeddings_from_db():
    """Load all embeddings and metadata from PostgreSQL database"""
    # Try loading from Database
    results = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        query = """
        SELECT item_id, domain, topic, stem, "choice_A", "choice_B", 
               "choice_C", "choice_D", key, rationale, rasch_b, pvalue, 
               point_biserial, embedding
        FROM itembank
        WHERE embedding IS NOT NULL
        ORDER BY item_id
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        cursor.close()
        conn.close()
        st.toast("Loaded data from Database", icon="🗄️")
        
    except Exception as e:
        # Fallback to CSV
        csv_path = "item_bank_hosted.csv"
        if os.path.exists(csv_path):
            st.warning(f"⚠️ Database connection failed. Falling back to {csv_path}")
            try:
                df = pd.read_csv(csv_path)
                # Ensure columns match the SQL query order
                cols = ["item_id", "domain", "topic", "stem", "choice_A", "choice_B", 
                        "choice_C", "choice_D", "key", "rationale", "rasch_b", "pvalue", 
                        "point_biserial", "embedding"]
                # Fill NaNs with None to match psycopg2 behavior
                df = df.where(pd.notnull(df), None)
                results = df[cols].values.tolist()
            except Exception as csv_e:
                st.error(f"❌ Failed to load CSV fallback: {csv_e}")
                return [], [], np.array([])
        else:
            st.error(f"❌ Database failed and no CSV found: {str(e)}")
            return [], [], np.array([])

    if not results:
        return [], [], np.array([])

    # Parse results
    ids = []
    items = []
    embeddings = []
    
    for row in results:
        (item_id, domain, topic, stem, choice_a, choice_b, choice_c, choice_d, 
            key, rationale, rasch_b, pvalue, point_biserial, embedding_str) = row
        
        ids.append(item_id)
        
        # Build item data dictionary
        item_data = {
            'item_id': item_id,
            'item_stem': stem,
            'content_category': domain if domain else topic,
            'topic': topic,
            'domain': domain,
            'difficulty_level': 'Hard' if rasch_b and rasch_b > 0.5 else 'Medium' if rasch_b and rasch_b > -0.5 else 'Easy',
            'choices': {
                'A': choice_a,
                'B': choice_b,
                'C': choice_c,
                'D': choice_d
            },
            'key': key,
            'rationale': rationale,
            'rasch_b': rasch_b,
            'pvalue': pvalue,
            'point_biserial': point_biserial
        }
        items.append(item_data)
        
        # Parse embedding
        if isinstance(embedding_str, str):
            embedding = np.array(json.loads(embedding_str))
        else:
            # Handle case where it might already be a list/array (unlikely from CSV but possible)
            embedding = np.array(embedding_str)
            
        embeddings.append(embedding)
    
    embeddings_array = np.array(embeddings)
    
    st.success(f"✅ Loaded {len(ids)} items with {EMBEDDING_DIM}-dimensional embeddings")
    
    return ids, items, embeddings_array

@st.cache_data
def reduce_dimensions(embeddings, method='tsne', n_components=3, perplexity=30, max_iter=1000, n_neighbors=15, min_dist=0.1):
    """Reduce embedding dimensions for visualization"""
    if method == 'tsne':
        reducer = TSNE(n_components=n_components, perplexity=perplexity, 
                      max_iter=max_iter, random_state=42)
    elif method == 'pca':
        reducer = PCA(n_components=n_components, random_state=42)
    elif method == 'umap':
        reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors,
                           min_dist=min_dist, random_state=42)
    else:
        reducer = TSNE(n_components=n_components, perplexity=perplexity, 
                      max_iter=max_iter, random_state=42)
    
    reduced = reducer.fit_transform(embeddings)
    return reduced

@st.cache_data
def cluster_embeddings(embeddings, n_clusters=8):
    """Perform K-means clustering on embeddings"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    return cluster_labels, kmeans.cluster_centers_

def extract_item_info(item_data):
    """Extract relevant information from item data for display"""
    stem = item_data.get('item_stem', 'N/A')
    category = item_data.get('content_category', 'Unknown')
    difficulty = item_data.get('difficulty_level', 'Unknown')
    
    # Truncate stem for display
    stem_short = stem[:100] + "..." if len(stem) > 100 else stem
    
    return {
        'stem': stem_short,
        'category': category,
        'difficulty': difficulty
    }

def build_similarity_graph(embeddings, ids, items, threshold=0.8, max_edges=1000):
    """Build a network graph based on cosine similarity"""
    G = nx.Graph()
    
    # Add nodes
    for i, item_id in enumerate(ids):
        info = extract_item_info(items[i])
        G.add_node(item_id, 
                  category=info['category'],
                  difficulty=info['difficulty'],
                  stem=info['stem'])
    
    # Add edges based on similarity
    n = len(embeddings)
    edge_count = 0
    
    # Calculate similarities and add edges
    similarities = []
    for i in range(n):
        for j in range(i + 1, n):
            similarity = 1 - cosine(embeddings[i], embeddings[j])
            if similarity >= threshold:
                similarities.append((i, j, similarity))
    
    # Sort by similarity and take top edges
    similarities.sort(key=lambda x: x[2], reverse=True)
    similarities = similarities[:max_edges]
    
    for i, j, sim in similarities:
        G.add_edge(ids[i], ids[j], weight=sim)
        edge_count += 1
    
    return G

# ==================== Visualization Functions ====================

def plot_3d_scatter(reduced_embeddings, ids, items, cluster_labels=None, color_by='cluster'):
    """Create interactive 3D scatter plot"""
    
    df_plot = pd.DataFrame({
        'X': reduced_embeddings[:, 0],
        'Y': reduced_embeddings[:, 1],
        'Z': reduced_embeddings[:, 2],
        'ID': ids
    })
    
    # Add item information
    categories = []
    difficulties = []
    stems = []
    
    for item in items:
        info = extract_item_info(item)
        categories.append(info['category'])
        difficulties.append(info['difficulty'])
        stems.append(info['stem'])
    
    df_plot['Category'] = categories
    df_plot['Difficulty'] = difficulties
    df_plot['Stem'] = stems
    
    if cluster_labels is not None:
        df_plot['Cluster'] = [f"Cluster {c}" for c in cluster_labels]
    
    # Choose color dimension
    if color_by == 'cluster' and cluster_labels is not None:
        color_col = 'Cluster'
    elif color_by == 'category':
        color_col = 'Category'
    elif color_by == 'difficulty':
        color_col = 'Difficulty'
    else:
        color_col = 'Cluster' if cluster_labels is not None else 'Category'
    
    fig = px.scatter_3d(
        df_plot,
        x='X', y='Y', z='Z',
        color=color_col,
        hover_data=['ID', 'Category', 'Difficulty', 'Stem'],
        title=f'3D Visualization of NCLEX Item Embeddings (Colored by {color_col})',
        labels={'X': 'Dimension 1', 'Y': 'Dimension 2', 'Z': 'Dimension 3'}
    )
    
    fig.update_traces(marker=dict(size=5, opacity=0.8))
    fig.update_layout(height=700)
    
    return fig

def plot_2d_scatter(reduced_embeddings, ids, items, cluster_labels=None, color_by='cluster'):
    """Create interactive 2D scatter plot"""
    
    df_plot = pd.DataFrame({
        'X': reduced_embeddings[:, 0],
        'Y': reduced_embeddings[:, 1],
        'ID': ids
    })
    
    # Add item information
    categories = []
    difficulties = []
    stems = []
    
    for item in items:
        info = extract_item_info(item)
        categories.append(info['category'])
        difficulties.append(info['difficulty'])
        stems.append(info['stem'])
    
    df_plot['Category'] = categories
    df_plot['Difficulty'] = difficulties
    df_plot['Stem'] = stems
    
    if cluster_labels is not None:
        df_plot['Cluster'] = [f"Cluster {c}" for c in cluster_labels]
    
    # Choose color dimension
    if color_by == 'cluster' and cluster_labels is not None:
        color_col = 'Cluster'
    elif color_by == 'category':
        color_col = 'Category'
    elif color_by == 'difficulty':
        color_col = 'Difficulty'
    else:
        color_col = 'Cluster' if cluster_labels is not None else 'Category'
    
    fig = px.scatter(
        df_plot,
        x='X', y='Y',
        color=color_col,
        hover_data=['ID', 'Category', 'Difficulty', 'Stem'],
        title=f'2D Visualization of NCLEX Item Embeddings (Colored by {color_col})',
        labels={'X': 'Dimension 1', 'Y': 'Dimension 2'}
    )
    
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    fig.update_layout(height=600)
    
    return fig

def plot_network_graph(G, layout='spring'):
    """Create interactive network graph visualization"""
    
    # Calculate layout
    if layout == 'spring':
        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    elif layout == 'kamada':
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.circular_layout(G)
    
    # Extract node positions
    edge_x = []
    edge_y = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Node info
        node_data = G.nodes[node]
        category = node_data.get('category', 'Unknown')
        stem = node_data.get('stem', 'N/A')
        
        node_text.append(f"ID: {node}<br>Category: {category}<br>Stem: {stem}")
        node_color.append(len(list(G.neighbors(node))))  # Color by degree
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            color=node_color,
            size=10,
            colorbar=dict(
                thickness=15,
                title='Connections',
                xanchor='left'
            ),
            line=dict(width=2, color='white')
        )
    )
    
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title=dict(
                           text=f'Knowledge Graph: NCLEX Items Similarity Network ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)',
                           font=dict(size=16)
                       ),
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20, l=5, r=5, t=40),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       height=700
                   ))
    
    return fig

def plot_cluster_distribution(cluster_labels, items):
    """Plot distribution of items across clusters"""
    
    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
    
    # Get category distribution per cluster
    df_clusters = pd.DataFrame({
        'Cluster': [f"Cluster {c}" for c in cluster_labels],
        'Category': [extract_item_info(item)['category'] for item in items]
    })
    
    # Count plot
    fig1 = px.bar(
        x=cluster_counts.index,
        y=cluster_counts.values,
        labels={'x': 'Cluster', 'y': 'Number of Items'},
        title='Distribution of Items Across Clusters'
    )
    fig1.update_traces(marker_color='steelblue')
    
    # Category distribution per cluster
    cluster_category = df_clusters.groupby(['Cluster', 'Category']).size().reset_index(name='Count')
    fig2 = px.bar(
        cluster_category,
        x='Cluster',
        y='Count',
        color='Category',
        title='Category Distribution by Cluster',
        barmode='stack'
    )
    
    return fig1, fig2

# ==================== Main Streamlit App ====================

def main():
    st.set_page_config(page_title="NCLEX Knowledge Graph", layout="wide", page_icon="🧠")
    
    st.title("🧠 NCLEX Item Bank Knowledge Graph")
    st.markdown("**Visualize embeddings and relationships in the NCLEX item bank**")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    viz_type = st.sidebar.selectbox(
        "Visualization Type",
        ["3D Scatter Plot", "2D Scatter Plot", "Network Graph", "Cluster Analysis"]
    )
    
    # Load data
    with st.spinner("Loading embeddings from database..."):
        ids, items, embeddings = load_embeddings_from_db()
    
    if len(embeddings) == 0:
        st.error("No data loaded. Please check your database connection.")
        return
    
    # Dimensionality reduction settings
    if viz_type in ["3D Scatter Plot", "2D Scatter Plot"]:
        st.sidebar.subheader("Dimensionality Reduction")
        reduction_method = st.sidebar.radio("Method", ["t-SNE", "UMAP", "PCA"])
        
        perplexity = 30
        max_iter = 1000
        n_neighbors = 15
        min_dist = 0.1
        
        if reduction_method == "t-SNE":
            perplexity = st.sidebar.slider("Perplexity", 5, 50, 30)
            max_iter = st.sidebar.slider("Iterations", 250, 2000, 1000, step=250)
        elif reduction_method == "UMAP":
            n_neighbors = st.sidebar.slider("Neighbors", 2, 100, 15)
            min_dist = st.sidebar.slider("Min Distance", 0.0, 1.0, 0.1, 0.01)
        
        # Clustering settings
        perform_clustering = st.sidebar.checkbox("Apply Clustering", value=True)
        if perform_clustering:
            n_clusters = st.sidebar.slider("Number of Clusters", 3, 15, 8)
        
        # Color by
        color_options = ['cluster', 'category', 'difficulty'] if perform_clustering else ['category', 'difficulty']
        color_by = st.sidebar.selectbox("Color by", color_options)
        
        # Generate visualization
        n_components = 3 if viz_type == "3D Scatter Plot" else 2
        
        with st.spinner(f"Reducing dimensions with {reduction_method}..."):
            reduced = reduce_dimensions(
                embeddings, 
                method=reduction_method.lower().replace('-', ''),
                n_components=n_components,
                perplexity=perplexity,
                max_iter=max_iter,
                n_neighbors=n_neighbors,
                min_dist=min_dist
            )
        
        cluster_labels = None
        if perform_clustering:
            with st.spinner("Clustering embeddings..."):
                cluster_labels, _ = cluster_embeddings(embeddings, n_clusters)
        
        # Plot
        if viz_type == "3D Scatter Plot":
            fig = plot_3d_scatter(reduced, ids, items, cluster_labels, color_by)
        else:
            fig = plot_2d_scatter(reduced, ids, items, cluster_labels, color_by)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Items", len(ids))
        with col2:
            st.metric("Embedding Dimension", EMBEDDING_DIM)
        with col3:
            if perform_clustering:
                st.metric("Clusters", n_clusters)
    
    elif viz_type == "Network Graph":
        st.sidebar.subheader("Network Settings")
        similarity_threshold = st.sidebar.slider("Similarity Threshold", 0.5, 0.95, 0.8, 0.05)
        max_edges = st.sidebar.slider("Max Edges", 100, 5000, 1000, 100)
        layout_method = st.sidebar.selectbox("Layout", ["spring", "kamada", "circular"])
        
        with st.spinner("Building similarity graph..."):
            G = build_similarity_graph(embeddings, ids, items, similarity_threshold, max_edges)
        
        if G.number_of_edges() == 0:
            st.warning("⚠️ No edges found with current threshold. Try lowering the similarity threshold.")
            return
        
        fig = plot_network_graph(G, layout_method)
        st.plotly_chart(fig, use_container_width=True)
        
        # Network statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Nodes", G.number_of_nodes())
        with col2:
            st.metric("Edges", G.number_of_edges())
        with col3:
            avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
            st.metric("Avg Connections", f"{avg_degree:.2f}")
        with col4:
            density = nx.density(G)
            st.metric("Network Density", f"{density:.4f}")
    
    elif viz_type == "Cluster Analysis":
        st.sidebar.subheader("Clustering Settings")
        n_clusters = st.sidebar.slider("Number of Clusters", 3, 15, 8)
        
        with st.spinner("Performing clustering analysis..."):
            cluster_labels, centers = cluster_embeddings(embeddings, n_clusters)
            
            # Reduce to 2D for visualization
            reduced_2d = reduce_dimensions(embeddings, method='tsne', n_components=2)
        
        # Cluster scatter plot
        fig_scatter = plot_2d_scatter(reduced_2d, ids, items, cluster_labels, 'cluster')
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Distribution plots
        st.subheader("Cluster Distribution Analysis")
        fig1, fig2 = plot_cluster_distribution(cluster_labels, items)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            st.plotly_chart(fig2, use_container_width=True)
        
        # Cluster details
        st.subheader("Cluster Details")
        for cluster_id in range(n_clusters):
            with st.expander(f"Cluster {cluster_id} ({np.sum(cluster_labels == cluster_id)} items)"):
                cluster_items_idx = np.where(cluster_labels == cluster_id)[0]
                sample_items = [items[i] for i in cluster_items_idx[:5]]
                
                for i, item in enumerate(sample_items):
                    info = extract_item_info(item)
                    st.markdown(f"**Item {cluster_items_idx[i] + 1}** - {info['category']} ({info['difficulty']})")
                    st.caption(info['stem'])
                    st.divider()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info(f"📊 Dataset: {len(ids)} NCLEX items\n\n🔢 Embedding dim: {EMBEDDING_DIM}")

if __name__ == "__main__":
    main()
