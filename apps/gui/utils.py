import plotly.graph_objects as go
import numpy as np

def get_rotation_demo(angle_deg):
    theta = np.radians(angle_deg)
    # Rotation Matrix for Y-axis
    R = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    
    # 21 Hand Landmarks
    hand_landmarks = np.array([
        [0, 0, 0],          # 0: Wrist
        [-0.15, 0.1, 0], [-0.25, 0.25, 0], [-0.35, 0.4, 0], [-0.45, 0.5, 0],
        [-0.1, 0.4, 0],  [-0.12, 0.55, 0], [-0.14, 0.7, 0], [-0.16, 0.8, 0],
        [0, 0.45, 0],    [0, 0.6, 0],      [0, 0.75, 0],     [0, 0.9, 0],
        [0.1, 0.4, 0],   [0.12, 0.55, 0],  [0.14, 0.7, 0],  [0.16, 0.8, 0],
        [0.2, 0.35, 0],  [0.25, 0.45, 0],  [0.3, 0.55, 0],  [0.35, 0.65, 0]
    ])
    
    rotated_hand = np.dot(hand_landmarks, R.T)
    
    connections = [
        (0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8),
        (9,10), (10,11), (11,12), (13,14), (14,15), (15,16),
        (0,17), (17,18), (18,19), (19,20), (5,9), (9,13), (13,17)
    ]
    
    fig = go.Figure()

    # Connections (Neon Green)
    for start, end in connections:
        line = np.array([rotated_hand[start], rotated_hand[end]])
        fig.add_trace(go.Scatter3d(
            x=line[:, 0], y=line[:, 1], z=line[:, 2],
            mode='lines',
            line=dict(color='#00FF00', width=6),
            hoverinfo='none', showlegend=False
        ))

    # Landmarks (Red)
    fig.add_trace(go.Scatter3d(
        x=rotated_hand[:, 0], y=rotated_hand[:, 1], z=rotated_hand[:, 2],
        mode='markers',
        marker=dict(size=5, color='red', opacity=1.0),
        showlegend=False
    ))

# Define a custom "Soft Blue" theme
    bg_color = "rgb(10, 25, 50)"      # Deep navy background
    grid_color = "rgb(50, 100, 200)"  # Bright blue grid lines
    plane_color = "rgb(15, 35, 70)"   # Slightly lighter blue for the planes
    
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=0, r=0, b=0, t=0),
        height=500,
        paper_bgcolor="rgba(0,0,0,0)", # Transparent around the plot
        plot_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            xaxis=dict(
                title='X', 
                backgroundcolor=plane_color, 
                gridcolor=grid_color, 
                showbackground=True, 
                zerolinecolor="white", 
                range=[-1, 1]
            ),
            yaxis=dict(
                title='Y', 
                backgroundcolor=plane_color, 
                gridcolor=grid_color, 
                showbackground=True, 
                zerolinecolor="white", 
                range=[-1, 1]
            ),
            zaxis=dict(
                title='Z', 
                backgroundcolor=plane_color, 
                gridcolor=grid_color, 
                showbackground=True, 
                zerolinecolor="white", 
                range=[-1, 1]
            ),
            aspectmode='cube'
        ),
        scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
    )
    return fig




def get_scaling_demo(scale_factor):
    # 21 Hand Landmarks (MediaPipe Structure)
    hand_landmarks = np.array([
        [0, 0, 0],          # 0: Wrist
        [-0.15, 0.1, 0], [-0.25, 0.25, 0], [-0.35, 0.4, 0], [-0.45, 0.5, 0],
        [-0.1, 0.4, 0],  [-0.12, 0.55, 0], [-0.14, 0.7, 0], [-0.16, 0.8, 0],
        [0, 0.45, 0],    [0, 0.6, 0],      [0, 0.75, 0],     [0, 0.9, 0],
        [0.1, 0.4, 0],   [0.12, 0.55, 0],  [0.14, 0.7, 0],  [0.16, 0.8, 0],
        [0.2, 0.35, 0],  [0.25, 0.45, 0],  [0.3, 0.55, 0],  [0.35, 0.65, 0]
    ])
    
    # Apply Scaling Transformation
    scaled_hand = hand_landmarks * scale_factor
    
    connections = [
        (0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8),
        (9,10), (10,11), (11,12), (13,14), (14,15), (15,16),
        (0,17), (17,18), (18,19), (19,20), (5,9), (9,13), (13,17)
    ]
    
    fig = go.Figure()

    # Draw Bones (Neon Green)
    for start, end in connections:
        line = np.array([scaled_hand[start], scaled_hand[end]])
        fig.add_trace(go.Scatter3d(
            x=line[:, 0], y=line[:, 1], z=line[:, 2],
            mode='lines',
            line=dict(color='#00FF00', width=6),
            showlegend=False
        ))

    # Draw Joints (Red)
    fig.add_trace(go.Scatter3d(
        x=scaled_hand[:, 0], y=scaled_hand[:, 1], z=scaled_hand[:, 2],
        mode='markers',
        marker=dict(size=5, color='red'),
        showlegend=False
    ))

    # Cyberpunk Blue Theme with Grid
    bg_color = "rgb(10, 25, 50)"
    grid_color = "rgb(50, 100, 200)"
    plane_color = "rgb(15, 35, 70)"

    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=0, r=0, b=0, t=0),
        height=500,
        scene=dict(
            xaxis=dict(backgroundcolor=plane_color, gridcolor=grid_color, showbackground=True, range=[-1, 1]),
            yaxis=dict(backgroundcolor=plane_color, gridcolor=grid_color, showbackground=True, range=[-1, 1]),
            zaxis=dict(backgroundcolor=plane_color, gridcolor=grid_color, showbackground=True, range=[-1, 1]),
            aspectmode='cube'
        )
    )
    return fig


def get_time_stretch_plot(rate):
    # Simulate a single landmark (e.g., Hand Y-position) moving over 60 frames
    t = np.linspace(0, 1, 60)
    # Original sine-wave motion
    original_y = np.sin(t * np.pi) 
    
    # Stretched/Compressed motion
    # We change the 'frequency' to simulate speed
    stretched_y = np.sin(t * np.pi * (1/rate))

    fig = go.Figure()
    
    # Original Motion Reference
    fig.add_trace(go.Scatter(x=np.arange(60), y=original_y, name="Original Speed",
                             line=dict(color='gray', dash='dash')))
    
    # Augmented Motion
    fig.add_trace(go.Scatter(x=np.arange(60), y=stretched_y, name="Augmented Speed",
                             line=dict(color='#00FF00', width=4)))

    fig.update_layout(
        title=f"Temporal Shift (Rate: {rate}x)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgb(10, 25, 50)",
        xaxis=dict(title="Frame Index (Time)", gridcolor="rgb(50, 100, 200)"),
        yaxis=dict(title="Joint Position", gridcolor="rgb(50, 100, 200)"),
        height=300,
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig


import plotly.graph_objects as go
import numpy as np

def get_translation_demo(dx, dy):
    # 21 Hand Landmarks (standard structure)
    hand_landmarks = np.array([
        [0, 0, 0],          # 0: Wrist
        [-0.15, 0.1, 0], [-0.25, 0.25, 0], [-0.35, 0.4, 0], [-0.45, 0.5, 0],
        [-0.1, 0.4, 0],  [-0.12, 0.55, 0], [-0.14, 0.7, 0], [-0.16, 0.8, 0],
        [0, 0.45, 0],    [0, 0.6, 0],      [0, 0.75, 0],     [0, 0.9, 0],
        [0.1, 0.4, 0],   [0.12, 0.55, 0],  [0.14, 0.7, 0],  [0.16, 0.8, 0],
        [0.2, 0.35, 0],  [0.25, 0.45, 0],  [0.3, 0.55, 0],  [0.35, 0.65, 0]
    ])
    
    # Apply Translation + Clipping to [0, 1] range as per requirements
    translated_hand = hand_landmarks + np.array([dx, dy, 0])
    translated_hand = np.clip(translated_hand, -1, 1) # Clipping for visual box
    
    connections = [
        (0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8),
        (9,10), (10,11), (11,12), (13,14), (14,15), (15,16),
        (0,17), (17,18), (18,19), (19,20), (5,9), (9,13), (13,17)
    ]
    
    fig = go.Figure()

    for start, end in connections:
        line = np.array([translated_hand[start], translated_hand[end]])
        fig.add_trace(go.Scatter3d(
            x=line[:, 0], y=line[:, 1], z=line[:, 2],
            mode='lines', line=dict(color='#00FF00', width=6), showlegend=False
        ))

    fig.add_trace(go.Scatter3d(
        x=translated_hand[:, 0], y=translated_hand[:, 1], z=translated_hand[:, 2],
        mode='markers', marker=dict(size=5, color='red'), showlegend=False
    ))

    # Theme Consistency
    plane_color = "rgb(15, 35, 70)"
    grid_color = "rgb(50, 100, 200)"

    fig.update_layout(
        template="plotly_dark", margin=dict(l=0, r=0, b=0, t=0), height=450,
        scene=dict(
            xaxis=dict(backgroundcolor=plane_color, gridcolor=grid_color, showbackground=True, range=[-1, 1]),
            yaxis=dict(backgroundcolor=plane_color, gridcolor=grid_color, showbackground=True, range=[-1, 1]),
            zaxis=dict(backgroundcolor=plane_color, gridcolor=grid_color, showbackground=True, range=[-1, 1]),
            aspectmode='cube'
        )
    )
    return fig


def get_ik_visual_demo(dx_change):
    # Fixed Shoulder and original arm lengths
    shoulder = np.array([0, 0])
    len_arm = 0.4
    len_forearm = 0.35
    
    # Original wrist target and shifted wrist target
    original_wrist = np.array([0.5, 0.2])
    target_wrist = original_wrist + np.array([dx_change, 0])
    
    # Simplified IK Solver for the Demo (finds the elbow position)
    dist = np.linalg.norm(target_wrist - shoulder)
    # Ensure distance is within reach for the demo
    dist = np.clip(dist, abs(len_arm - len_forearm) + 1e-5, len_arm + len_forearm - 1e-5)
    
    a = (dist**2 + len_arm**2 - len_forearm**2) / (2 * dist)
    h = np.sqrt(max(0, len_arm**2 - a**2))
    
    # Calculate Elbow Position
    vec_wrist = (target_wrist - shoulder) / dist
    proj_point = shoulder + a * vec_wrist
    perp_vec = np.array([-vec_wrist[1], vec_wrist[0]])
    elbow = proj_point + h * perp_vec # Choose one solution
    
    fig = go.Figure()

    # Draw Arm Segments (Shoulder -> Elbow -> Wrist)
    arm_points = np.array([shoulder, elbow, target_wrist])
    
    fig.add_trace(go.Scatter(
        x=arm_points[:, 0], y=arm_points[:, 1],
        mode='lines+markers',
        line=dict(color='#00FF00', width=6),
        marker=dict(size=12, color=['white', 'red', 'red']),
        name="Augmented Arm"
    ))

    # Reference "Ghost" Arm (Original position)
    fig.add_trace(go.Scatter(
        x=[0, 0.3, 0.5], y=[0, 0.35, 0.2], # Mock original pose
        mode='lines',
        line=dict(color='rgba(255, 255, 255, 0.2)', dash='dot'),
        name="Original Pose"
    ))

    # Theme Styling
    fig.update_layout(
        template="plotly_dark",
        height=400,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgb(10, 25, 50)",
        xaxis=dict(range=[-0.2, 1.0], gridcolor="rgb(50, 100, 200)", zeroline=False),
        yaxis=dict(range=[-0.2, 0.8], gridcolor="rgb(50, 100, 200)", zeroline=False),
        margin=dict(l=20, r=20, b=20, t=20)
    )
    return fig