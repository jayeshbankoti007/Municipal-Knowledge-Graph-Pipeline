import pickle
import networkx as nx
import plotly.graph_objects as go
from collections import Counter
import numpy as np


def visualize_knowledge_graph():    
    """Visualize the knowledge graph using Plotly"""
    
    # Load graph
    print("Loading graph...")
    with open("output/knowledge_graph.pkl", "rb") as f:
        graph = pickle.load(f)

    print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # Diagnostics - count nodes by type
    node_types = Counter(d.get('type') for n, d in graph.nodes(data=True))
    print("\nNode types:")
    for node_type, count in node_types.items():
        print(f"  {node_type}: {count}")

    # Diagnostics - count edges
    edge_types = Counter(d.get('relation') for u, v, d in graph.edges(data=True))
    print("\nEdge types:")
    for rel, count in edge_types.items():
        print(f"  {rel}: {count}")

    # Create layout
    print("\nCreating layout...")
    pos = nx.spring_layout(graph, seed=42, k=3, iterations=100)

    # Edge color mapping
    edge_colors_map = {
        'VOTED_ON': '#2E8B57',      # Green
        'MEMBER_OF': '#FF8C00',     # Orange
        'AUTHORIZES': '#DC143C',    # Red
        'MENTIONED_IN': '#4169E1',  # Blue
        'RELATES_TO': '#9370DB'     # Purple
    }

    # Create edge traces with proper connections
    edge_x = []
    edge_y = []
    edge_colors = []
    edge_hover = []

    for u, v, d in graph.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        
        relation = d.get('relation', 'Unknown')
        color = edge_colors_map.get(relation, '#999999')
        
        # Add edge line
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_colors.extend([color, color, None])
        
        # Create hover text for edge
        u_name = u.split(':')[1] if ':' in u else u
        v_name = v.split(':')[1] if ':' in v else v
        hover_text = f"<b>{relation}</b><br>{u_name} → {v_name}"
        
        if 'vote' in d:
            hover_text += f"<br>Vote: <b>{d['vote']}</b>"
        if 'role' in d:
            hover_text += f"<br>Role: {d['role']}"
        edge_hover.extend([hover_text, hover_text, None])

    # Single edge trace
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode='lines',
        line=dict(width=1.5, color='rgba(125,125,125,0.3)'),
        hoverinfo='text',
        hovertext=edge_hover,
        showlegend=False
    )

    # Node configuration - Bills are MOST PROMINENT
    node_config = {
        'Bill': {
            'color': '#FFD700',      # Gold - most prominent!
            'size': 35,              # Largest
            'line_width': 3,
            'line_color': '#FF8C00',
            'opacity': 1.0,
            'icon': '📜'
        },
        'Person': {
            'color': '#87CEEB',      # Sky blue
            'size': 18,
            'line_width': 2,
            'line_color': '#4682B4',
            'opacity': 0.85,
            'icon': '👤'
        },
        'Organization': {
            'color': '#FFE4B5',      # Moccasin
            'size': 22,
            'line_width': 2,
            'line_color': '#DEB887',
            'opacity': 0.85,
            'icon': '🏛️'
        },
        'Project': {
            'color': '#FFB6C1',      # Light pink
            'size': 20,
            'line_width': 2,
            'line_color': '#FF69B4',
            'opacity': 0.85,
            'icon': '🏗️'
        }
    }

    node_traces = []

    for node_type, config in node_config.items():
        # Filter nodes by type
        nodes = [n for n, d in graph.nodes(data=True) if d.get('type') == node_type]
        
        if not nodes:
            print(f"⚠️  WARNING: No {node_type} nodes found!")
            continue
        
        print(f"  Found {len(nodes)} {node_type} nodes")
        
        x_coords = []
        y_coords = []
        hover_texts = []
        node_labels = []
        
        for node in nodes:
            x_coords.append(pos[node][0])
            y_coords.append(pos[node][1])
            
            data = graph.nodes[node]
            node_id = node.split(':')[1] if ':' in node else node
            
            # Short label for display
            if node_type == 'Bill':
                node_labels.append(f"<b>{node_id}</b>")  # Bold for bills!
            else:
                # Truncate long names
                label = node_id[:20] + '...' if len(node_id) > 20 else node_id
                node_labels.append(label)
            
            # Build detailed hover text
            if node_type == 'Person':
                hover_text = f"<b>{config['icon']} PERSON</b><br>"
                hover_text += f"<b>{data.get('name', node_id)}</b><br>"
                hover_text += f"━━━━━━━━━━━━━━━<br>"
                hover_text += f"<b>Role:</b> {data.get('role', 'N/A')}<br>"
                hover_text += f"<b>Organization:</b> {data.get('organization', 'N/A')}<br>"
                
                # Count votes
                out_edges = list(graph.out_edges(node, data=True))
                votes = [(v, d.get('vote', 'N/A')) for u, v, d in out_edges 
                        if d.get('relation') == 'VOTED_ON']
                if votes:
                    hover_text += f"<br><b>📊 Votes Cast: {len(votes)}</b><br>"
                    for i, (bill, vote) in enumerate(votes[:5]):
                        bill_id = bill.split(':')[1] if ':' in bill else bill
                        hover_text += f"  {i+1}. {bill_id}: <b>{vote}</b><br>"
                    if len(votes) > 5:
                        hover_text += f"  ... +{len(votes)-5} more<br>"
            
            elif node_type == 'Bill':
                hover_text = f"<b>{config['icon']} BILL</b><br>"
                hover_text += f"<b style='font-size:14px'>{data.get('bill_id', node_id)}</b><br>"
                hover_text += f"━━━━━━━━━━━━━━━<br>"
                title = data.get('title', 'N/A')
                hover_text += f"<b>Title:</b><br>{title[:150]}{'...' if len(title) > 150 else ''}<br><br>"
                hover_text += f"<b>Type:</b> {data.get('type', 'N/A')}<br>"
                
                # PREDICTION - Most important!
                prediction = data.get('prediction', 'N/A')
                confidence = data.get('prediction_confidence', 'N/A')
                
                # Color code predictions
                pred_emoji = {
                    'APPROVED': '✅',
                    'REJECTED': '❌',
                    'UNCERTAIN': '❓'
                }
                
                hover_text += f"<br><b>🎯 PREDICTION: {pred_emoji.get(prediction, '❓')} {prediction}</b><br>"
                hover_text += f"<b>Confidence:</b> {confidence}<br>"
                reasoning = data.get('reasoning', 'N/A')
                hover_text += f"<b>Reasoning:</b><br>{reasoning[:200]}{'...' if len(reasoning) > 200 else ''}<br>"
                
                # Count votes
                in_edges = list(graph.in_edges(node, data=True))
                votes = [d.get('vote') for u, v, d in in_edges if d.get('relation') == 'VOTED_ON']
                if votes:
                    vote_counts = Counter(votes)
                    hover_text += f"<br><b>📊 Vote Breakdown ({len(votes)} total):</b><br>"
                    for vote_type, count in sorted(vote_counts.items(), key=lambda x: -x[1]):
                        hover_text += f"  • {vote_type}: {count}<br>"
            
            elif node_type == 'Organization':
                hover_text = f"<b>{config['icon']} ORGANIZATION</b><br>"
                hover_text += f"<b>{data.get('name', node_id)}</b><br>"
                hover_text += f"━━━━━━━━━━━━━━━<br>"
                aliases = data.get('aliases', '')
                if aliases:
                    hover_text += f"<b>Aliases:</b> {aliases}<br>"
                
                # Count members
                in_edges = list(graph.in_edges(node, data=True))
                members = [(u, d.get('role', 'member')) for u, v, d in in_edges 
                        if d.get('relation') == 'MEMBER_OF']
                if members:
                    hover_text += f"<br><b>👥 Members: {len(members)}</b><br>"
                    for i, (member, role) in enumerate(members[:8]):
                        member_name = member.split(':')[1] if ':' in member else member
                        hover_text += f"  {i+1}. {member_name} ({role})<br>"
                    if len(members) > 8:
                        hover_text += f"  ... +{len(members)-8} more<br>"
            
            elif node_type == 'Project':
                hover_text = f"<b>{config['icon']} PROJECT</b><br>"
                hover_text += f"<b>{data.get('name', node_id)}</b><br>"
                hover_text += f"━━━━━━━━━━━━━━━<br>"
                aliases = data.get('aliases', '')
                if aliases:
                    hover_text += f"<b>Aliases:</b> {aliases}<br>"
                
                # Count authorizing bills
                in_edges = list(graph.in_edges(node, data=True))
                bills = [u for u, v, d in in_edges if d.get('relation') == 'AUTHORIZES']
                if bills:
                    hover_text += f"<br><b>📋 Authorized by {len(bills)} bill(s):</b><br>"
                    for i, bill in enumerate(bills[:5]):
                        bill_id = bill.split(':')[1] if ':' in bill else bill
                        hover_text += f"  {i+1}. {bill_id}<br>"
                    if len(bills) > 5:
                        hover_text += f"  ... +{len(bills)-5} more<br>"
            
            hover_texts.append(hover_text)
        
        # Create node trace
        node_traces.append(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='markers+text',
            marker=dict(
                size=config['size'],
                color=config['color'],
                line=dict(width=config['line_width'], color=config['line_color']),
                opacity=config['opacity']
            ),
            text=node_labels,
            textposition='top center',
            textfont=dict(
                size=10 if node_type == 'Bill' else 7,  # Larger text for bills
                color='black',
                family='Arial Black' if node_type == 'Bill' else 'Arial'
            ),
            hovertext=hover_texts,
            hoverinfo='text',
            name=f"{config['icon']} {node_type}",
            showlegend=True,
            legendgroup=node_type
        ))

    # Create figure
    print("\nCreating interactive visualization...")
    fig = go.Figure(data=[edge_trace] + node_traces)

    # Update layout
    fig.update_layout(
        title={
            'text': '📜 Atlanta City Council Knowledge Graph<br><sub>BILLS (Gold) are the primary focus • Hover for predictions & details</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 26, 'color': '#1a1a1a'}
        },
        showlegend=True,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=100),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='#f8f9fa',
        width=1800,
        height=1400,
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor='rgba(255, 255, 255, 0.95)',
            bordercolor='#333',
            borderwidth=2,
            font=dict(size=14, family='Arial Black'),
            title=dict(text='<b>Node Types</b>', font=dict(size=16))
        )
    )

    # Save as interactive HTML
    output_file = "output/knowledge_graph_interactive.html"
    fig.write_html(output_file)
    print(f"\n✅ Interactive graph saved to: {output_file}")
    print(f"   Open in browser to explore!")

    print(f"\n💡 VISUALIZATION FEATURES:")
    print(f"   📜 BILLS (GOLD) - Largest & most prominent")
    print(f"   👤 People (Blue) - Voters & speakers")
    print(f"   🏛️ Organizations (Beige) - Departments & agencies")
    print(f"   🏗️ Projects (Pink) - Infrastructure & initiatives")

    print(f"\n📊 INTERACTIONS:")
    print(f"   • Hover over BILLS to see predictions & reasoning")
    print(f"   • Hover over nodes for detailed information")
    print(f"   • Click and drag to pan")
    print(f"   • Scroll to zoom")
    print(f"   • Click legend items to hide/show categories")