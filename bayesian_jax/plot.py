import plotly.graph_objects as go

def plot2d(res, n_init, n):
    # data
    n_total = 1 + n + n_init
    sample = list(range(1, 26))
    target = res.target_all
    x = res.params_all[:,0]
    y = res.params_all[:,1]
    x = list(x)
    y = list(y) 
    # create 2d scatter
    fig = go.Figure(data=[go.Scatter3d(
        x=x,
        y=y,
        z=target,
        mode='markers',
        marker=dict(
            size=5,
            color=sample,  
            colorscale='Viridis',  # Using Viridis Color Mapping
            opacity=0.8
        ),
        text=sample  # Display sample number on mouse hover
    )])

    # Setting the Chart Layout
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Target'
        ),
        title='Two-dimensional Scatter Plot',
        width=800,
        height=600
    )

    fig.show()

def plot3d(res, n_init, n):
    # data
    n_total = 1 + n + n_init
    sample = list(range(1, n_total + 1))
    target = res.target_all
    x = res.params_all[:, 0]
    y = res.params_all[:, 1]
    z = res.params_all[:, 2]
    x = list(x)
    y = list(y) 
    z = list(z)

    # create 3d scatter with color mapping
    fig = go.Figure(data=[go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=5,
            color=target,  
            colorscale='Viridis',  
            colorbar=dict(title='Target'),  
            opacity=0.8
        ),
        text=sample  # Display sample number on mouse hover
    )])

    # Setting the Chart Layout
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
        ),
        title='Three-dimensional Scatter Plot with Color Mapping',
        width=800,
        height=600
    )

    fig.show()
