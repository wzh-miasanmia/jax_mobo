import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

def mccormick_function(x, y):
    return np.sin(x + y) + (x - y)**2 - 1.5*x + 2.5*y + 1

# 生成 x, y 坐标值
x = np.linspace(-1.5, 4, 400)
y = np.linspace(-3, 3, 300)
x, y = np.meshgrid(x, y)

# 计算函数值
z = mccormick_function(x, y)

# 绘制图形
fig, ax = plt.subplots()
contour = ax.contourf(x, y, z, levels=1000, cmap='viridis')
cbar = plt.colorbar(contour)

# 设置指数刻度
cbar.set_ticks(np.arange(0, 36, 5))  # 这里设置了从0到30，每5个单位一个刻度


# 添加等高线
contour_lines = plt.contour(x, y, z, colors='white', linewidths=0.5)
plt.clabel(contour_lines, inline=True, fontsize=8)

# 添加标签和标题
plt.xlabel('x')
plt.ylabel('y')
plt.title('McCormick Function')

# 显示图形
plt.savefig('/home/wzhmiasanmia/ma_workspace/jax_mobo/mobo_np/McCormick.png')