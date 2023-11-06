import plotly.graph_objects as go


def plot3d(res, n_init, n):

    # 数据
    n_total = 1 + n + n_init
    sample = list(range(1, 26))
    target = res.target_all
    x = res.params_all[:,0]
    y = res.params_all[:,1]
    x = list(x)
    y = list(y) 
    # 创建三维散点图
    fig = go.Figure(data=[go.Scatter3d(
        x=x,
        y=y,
        z=target,
        mode='markers',
        marker=dict(
            size=5,
            color=sample,  # 以不同的样本编号为颜色
            colorscale='Viridis',  # 使用Viridis颜色映射
            opacity=0.8
        ),
        text=sample  # 鼠标悬停时显示样本编号
    )])

    # 设置图表布局
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Target'
        ),
        title='Three-dimensional Scatter Plot',
        width=800,
        height=600
    )

    # 显示图表
    fig.show()


