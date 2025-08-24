import streamlit as st
import networkx as nx
import plotly.express as px
import pandas as pd
import os
from pyvis.network import Network
from collections import defaultdict

def create_enhanced_visualization(graph, layout_type="spring"):
    """Create interactive network visualization"""
    if len(graph.nodes()) == 0:
        return "<div>No connections found with current filters</div>"
    
    if layout_type == "spring":
        pos = nx.spring_layout(graph, k=1, iterations=50)
    elif layout_type == "kamada_kawai":
        pos = nx.kamada_kawai_layout(graph)
    elif layout_type == "circular":
        pos = nx.circular_layout(graph)
    elif layout_type == "shell":
        pos = nx.shell_layout(graph)
    else:
        pos = nx.random_layout(graph)
    
    net = Network(height="800px", width="100%", bgcolor="#1e1e1e", 
                  font_color="white", directed=True, notebook=True)
    
    net.set_options("""
    var options = {
      "physics": {
        "enabled": true,
        "stabilization": {"iterations": 100},
        "barnesHut": {
          "gravitationalConstant": -8000,
          "centralGravity": 0.3,
          "springLength": 95,
          "springConstant": 0.04,
          "damping": 0.09
        }
      },
      "interaction": {
        "dragNodes": true,
        "dragView": true,
        "zoomView": true
      }
    }
    """)
    
    for node in graph.nodes(data=True):
        node_id, data = node
        
        if data.get("node_type") == "directory":
            color = "#FFD700"
            size = 25
            shape = "box"
        else:
            file_ext = data.get("file_type", "")
            color_map = {
                ".py": "#3572A5", ".js": "#F7DF1E", ".ts": "#3178C6",
                ".html": "#E34C26", ".css": "#563D7C", ".json": "#40A832",
                ".md": "#083FA1", ".txt": "#CCCCCC", ".xml": "#FF9900",
                ".yml": "#808080", ".yaml": "#808080", ".jsx": "#61DAFB",
                ".tsx": "#61DAFB", ".vue": "#4FC08D", ".php": "#777BB4",
                ".java": "#ED8B00", ".cpp": "#00599C", ".c": "#A8B9CC"
            }
            color = color_map.get(file_ext, "#888888")
            size = min(15 + graph.degree(node_id) * 2, 35)
            shape = "dot"
        
        label = os.path.basename(node_id) if not node_id.startswith("📁") else node_id
        title = f"{node_id}<br>Connections: {graph.degree(node_id)}"
        if data.get("size"):
            title += f"<br>Size: {data['size']} bytes"
        
        net.add_node(node_id, label=label, color=color, size=size, 
                    title=title, shape=shape)
    
    for edge in graph.edges(data=True):
        source, target, data = edge
        edge_type = data.get("edge_type", "connected")
        
        if "function" in edge_type:
            color = "#FF4444"
            width = 3
            label = edge_type.replace("calls_function_", "calls: ")
        elif edge_type == "imports":
            color = "#4444FF"
            width = 2
            label = "imports"
        elif "link" in edge_type:
            color = "#44FF44"
            width = 1
            label = edge_type.replace("_", " ")
        elif edge_type == "contains":
            color = "#CCCCCC"
            width = 1
            label = "contains"
        else:
            color = "#888888"
            width = 1
            label = edge_type
        
        net.add_edge(source, target, color=color, width=width, 
                    title=label, label=label if len(label) < 15 else "",
                    arrows="to")
    
    html_file = "enhanced_network.html"
    net.save_graph(html_file)
    
    with open(html_file, "r", encoding="utf-8") as f:
        html_content = f.read()
    
    return html_content

def create_statistics_dashboard(graph, file_dependencies):
    """Create statistics dashboard for the repository"""
    st.subheader("📊 Repository Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Files", len([n for n in graph.nodes() if not n.startswith("📁")]))
    with col2:
        st.metric("Total Connections", len(graph.edges()))
    with col3:
        st.metric("Connected Files", len([n for n in graph.nodes() if graph.degree(n) > 0]))
    with col4:
        avg_degree = sum(dict(graph.degree()).values()) / len(graph.nodes()) if graph.nodes() else 0
        st.metric("Avg Connections/File", f"{avg_degree:.1f}")
    
    file_types = defaultdict(int)
    for node in graph.nodes(data=True):
        if not node[0].startswith("📁"):
            ext = os.path.splitext(node[0])[1] or "no extension"
            file_types[ext] += 1
    
    if file_types:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📂 File Type Distribution")
            df_types = pd.DataFrame(list(file_types.items()), columns=["Extension", "Count"])
            fig_pie = px.pie(df_types, values="Count", names="Extension", 
                           title="File Types in Repository")
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.subheader("🔗 Most Connected Files")
            degrees = [(node, degree) for node, degree in graph.degree() 
                      if not node.startswith("📁")]
            degrees.sort(key=lambda x: x[1], reverse=True)
            
            top_files = degrees[:10]
            if top_files:
                df_connected = pd.DataFrame(top_files, columns=["File", "Connections"])
                df_connected["File"] = df_connected["File"].apply(os.path.basename)
                fig_bar = px.bar(df_connected, x="Connections", y="File", 
                               orientation="h", title="Top Connected Files")
                st.plotly_chart(fig_bar, use_container_width=True)