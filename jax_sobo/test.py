import jax
import jax.numpy as jnp

# 设置随机数种子
key = jax.random.PRNGKey(0)

# 均值向量
mean = jnp.array([0.0, 1.0])

# 协方差矩阵
cov = jnp.array([[1.0, 0.5], [0.5, 2.0]])

# 生成 3 个样本
samples = jax.random.multivariate_normal(key, mean, cov, shape=(3,))

print(samples)