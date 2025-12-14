"""
JAX 关键概念示例代码
这些示例展示了 DRQ 实现中使用的核心 JAX 技术
"""

import jax
import jax.numpy as jnp
from functools import partial

print("=" * 60)
print("1. 函数式编程：不可变数据结构")
print("=" * 60)

# JAX 数组是不可变的
x = jnp.array([1, 2, 3])
# x[0] = 10  # ❌ 这会报错！

# ✅ 正确方式：使用 .at[] 创建新数组
x_new = x.at[0].set(10)
print(f"原数组: {x}")
print(f"新数组: {x_new}")
print(f"原数组未改变: {x}")

print("\n" + "=" * 60)
print("2. RNG 状态管理")
print("=" * 60)

# JAX 使用显式 RNG 状态
rng = jax.random.PRNGKey(42)

# 每次使用随机数都需要分割 key
rng, key1 = jax.random.split(rng)
rng, key2 = jax.random.split(rng)

random1 = jax.random.normal(key1, (3,))
random2 = jax.random.normal(key2, (3,))

print(f"随机数1: {random1}")
print(f"随机数2: {random2}")
print(f"注意：每次使用后 rng 都会更新")

# 可以一次分割多个 key
rng, *keys = jax.random.split(rng, 5)
print(f"\n一次分割 5 个 key: {len(keys)} 个")

print("\n" + "=" * 60)
print("3. JIT 编译")
print("=" * 60)

# 普通函数
def slow_function(x):
    return jnp.sum(x ** 2)

# JIT 编译的函数
@jax.jit
def fast_function(x):
    return jnp.sum(x ** 2)

x = jnp.ones((1000, 1000))

# 第一次调用 JIT 函数会编译（较慢）
result1 = fast_function(x)
print("JIT 函数第一次调用完成（已编译）")

# 后续调用很快
result2 = fast_function(x)
print("JIT 函数后续调用很快")

# 静态参数示例
@partial(jax.jit, static_argnames=("multiplier",))
def multiply(x, multiplier):
    return x * multiplier

result = multiply(jnp.array([1, 2, 3]), multiplier=2)
print(f"静态参数示例: {result}")

print("\n" + "=" * 60)
print("4. 自动微分")
print("=" * 60)

def loss_fn(params, x, y):
    """简单的 MSE 损失"""
    pred = params * x
    return jnp.mean((pred - y) ** 2)

# 计算梯度
grad_fn = jax.grad(loss_fn, argnums=0)  # 对第一个参数求导

params = jnp.array(2.0)
x = jnp.array([1.0, 2.0, 3.0])
y = jnp.array([2.0, 4.0, 6.0])

grad = grad_fn(params, x, y)
print(f"参数: {params}")
print(f"梯度: {grad}")

# 带辅助信息的梯度
def loss_with_aux(params, x, y):
    pred = params * x
    loss = jnp.mean((pred - y) ** 2)
    aux = {"pred": pred, "loss": loss}
    return loss, aux

grad_fn_aux = jax.grad(loss_with_aux, has_aux=True, argnums=0)
grad, aux = grad_fn_aux(params, x, y)
print(f"梯度: {grad}")
print(f"辅助信息: {aux}")

print("\n" + "=" * 60)
print("5. tree_map: 操作嵌套结构")
print("=" * 60)

# 嵌套参数字典（类似神经网络参数）
params = {
    "layer1": {
        "weight": jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        "bias": jnp.array([0.1, 0.2])
    },
    "layer2": {
        "weight": jnp.array([[5.0, 6.0]]),
        "bias": jnp.array([0.3])
    }
}

# 对所有参数应用函数
def scale_by_09(x):
    return x * 0.9

scaled_params = jax.tree_map(scale_by_09, params)
print("原始参数 layer1.weight:")
print(params["layer1"]["weight"])
print("\n缩放后 layer1.weight:")
print(scaled_params["layer1"]["weight"])

# 计算参数总数
def count_params(x):
    return x.size

total = jax.tree_map(count_params, params)
print(f"\n每层参数数量: {total}")
print(f"总参数数: {sum(jax.tree_leaves(total))}")

print("\n" + "=" * 60)
print("6. vmap: 向量化映射")
print("=" * 60)

# 单个样本的处理函数
def process_single(x, rng):
    noise = jax.random.normal(rng, x.shape) * 0.1
    return x + noise

# 使用 vmap 批量处理
batch_x = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
batch_rng = jax.random.split(jax.random.PRNGKey(42), 3)

# 手动循环（慢）
result_manual = jnp.array([
    process_single(batch_x[i], batch_rng[i]) 
    for i in range(3)
])

# 使用 vmap（快，可并行）
vmap_process = jax.vmap(process_single, in_axes=(0, 0))
result_vmap = vmap_process(batch_x, batch_rng)

print("手动循环结果:")
print(result_manual)
print("\nvmap 结果:")
print(result_vmap)
print("\n结果相同:", jnp.allclose(result_manual, result_vmap))

print("\n" + "=" * 60)
print("7. scan: 函数式循环")
print("=" * 60)

# 使用 scan 实现循环（JIT 友好）
def scan_body(carry, x):
    """循环体：累加并返回新状态"""
    new_carry = carry + x
    output = carry * 2  # 输出
    return new_carry, output

initial_carry = 0
xs = jnp.array([1, 2, 3, 4, 5])

final_carry, outputs = jax.lax.scan(scan_body, initial_carry, xs)

print(f"初始状态: {initial_carry}")
print(f"输入序列: {xs}")
print(f"最终状态: {final_carry}")
print(f"输出序列: {outputs}")

# 在 DRQ 中，scan 用于多次 critic 更新
print("\n在 DRQ 中，scan 用于:")
print("  - 将 batch 分割为多个 minibatch")
print("  - 对每个 minibatch 进行 critic 更新")
print("  - 累积更新后的 agent 状态")

print("\n" + "=" * 60)
print("8. 设备管理")
print("=" * 60)

# 获取可用设备
devices = jax.local_devices()
print(f"可用设备: {devices}")

# 将数据放到设备
x = jnp.array([1, 2, 3])
x_on_device = jax.device_put(x, device=devices[0])
print(f"数据已放到设备: {x_on_device.device()}")

# 从设备取回（如果需要）
x_cpu = jax.device_get(x_on_device)
print(f"数据已取回 CPU: {type(x_cpu)}")

print("\n" + "=" * 60)
print("9. 实际应用：简化的损失函数")
print("=" * 60)

# 模拟 DRQ 中的损失函数结构
def critic_loss_fn(params, batch, rng):
    """
    简化的 critic 损失函数
    在实际 DRQ 中，这会调用神经网络
    """
    # 模拟预测 Q 值
    rng, key = jax.random.split(rng)
    predicted_q = jax.random.normal(key, (batch["rewards"].shape[0],))
    
    # 模拟目标 Q 值
    target_q = batch["rewards"] + 0.99 * batch["masks"] * predicted_q
    
    # MSE 损失
    loss = jnp.mean((predicted_q - target_q) ** 2)
    aux = {"loss": loss, "q_mean": predicted_q.mean()}
    
    return loss, aux

# 创建模拟 batch
batch = {
    "rewards": jnp.array([1.0, 2.0, 3.0]),
    "masks": jnp.array([1.0, 1.0, 0.0]),
}

# 计算梯度
params = {"dummy": jnp.array(1.0)}  # 模拟参数
rng = jax.random.PRNGKey(42)

grad_fn = jax.grad(critic_loss_fn, has_aux=True, argnums=0)
grads, aux = grad_fn(params, batch, rng)

print(f"损失值: {aux['loss']}")
print(f"Q 值均值: {aux['q_mean']}")
print(f"梯度结构: {list(grads.keys())}")

print("\n" + "=" * 60)
print("10. 数据增强示例（简化版）")
print("=" * 60)

def random_crop_simple(img, rng, padding=4):
    """简化的随机裁剪"""
    h, w = img.shape[:2]
    crop_h = jax.random.randint(rng, (), 0, 2 * padding + 1)
    crop_w = jax.random.randint(rng, (), 0, 2 * padding + 1)
    
    # 简化：直接裁剪（实际实现会先 padding）
    return img[crop_h:h-crop_h, crop_w:w-crop_w]

# 批量随机裁剪
def batched_crop(imgs, rng, padding=4):
    """对批次中的每张图像应用随机裁剪"""
    rngs = jax.random.split(rng, imgs.shape[0])
    return jax.vmap(
        lambda img, r: random_crop_simple(img, r, padding),
        in_axes=(0, 0)
    )(imgs, rngs)

# 示例
imgs = jnp.ones((3, 64, 64, 3))  # 3 张 64x64 RGB 图像
rng = jax.random.PRNGKey(42)

cropped = batched_crop(imgs, rng, padding=4)
print(f"原始形状: {imgs.shape}")
print(f"裁剪后形状: {cropped.shape}")
print("每张图像都应用了不同的随机裁剪")

print("\n" + "=" * 60)
print("总结")
print("=" * 60)
print("""
这些是 DRQ 实现中使用的核心 JAX 概念：

1. 函数式编程：所有操作都是纯函数，无副作用
2. RNG 管理：显式管理随机数状态
3. JIT 编译：加速关键函数
4. 自动微分：计算梯度
5. tree_map：操作嵌套结构
6. vmap：向量化批量操作
7. scan：函数式循环
8. 设备管理：CPU/GPU/TPU 数据管理

理解这些概念后，你就能更好地理解和修改 DRQ 代码了！
""")

