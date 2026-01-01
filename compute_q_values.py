"""
计算轨迹Q值的工具函数

轨迹长度为100，前面t步reward为0，后面reward为1。
Q值计算不截断：一旦出现reward=1，后续按无穷步计算（1 + gamma + gamma^2 + ... = 1/(1-gamma)）。
"""


def compute_q_value(reward_one_count: int, step: int, gamma: float) -> float:
    """
    计算某个时刻的Q值（按无穷步计算）
    
    参数:
        reward_one_count: reward=1的数量，表示从step (100 - reward_one_count)开始，所有reward都是1
        step: 需要计算Q值的step（0-99）
        gamma: 折扣因子
    
    返回:
        该step的Q值（按无穷步计算）
    """
    trajectory_length = 100
    
    # 计算reward开始为1的起始step
    reward_one_start = trajectory_length - reward_one_count
    
    # 如果step已经超过了轨迹长度
    if step >= trajectory_length:
        return 0.0
    
    # 如果step在reward=1的区间内（或之后）
    if step >= reward_one_start:
        # 从step开始，后面无穷步都是1
        # Q = 1 + gamma + gamma^2 + ... = 1/(1-gamma) (当gamma < 1时)
        if gamma == 1.0:
            # 如果gamma=1，无穷级数发散，返回一个很大的值或特殊标记
            return float('inf')
        else:
            # 无穷几何级数求和: 1/(1-gamma)
            return 1.0 / (1.0 - gamma)
    
    # 如果step在reward=0的区间内
    else:
        # 从step到reward_one_start-1都是0，从reward_one_start开始后面无穷步都是1
        steps_until_one = reward_one_start - step
        if gamma == 1.0:
            # 如果gamma=1，从reward_one_start开始后面无穷步都是1，所以是无穷
            return float('inf')
        else:
            # Q = gamma^steps_until_one * (1 + gamma + gamma^2 + ...)
            #    = gamma^steps_until_one * 1/(1-gamma)
            return (gamma ** steps_until_one) / (1.0 - gamma)


def compute_average_q_value(reward_one_count: int, gamma: float) -> float:
    """
    计算0-100每一步Q值的平均
    
    参数:
        reward_one_count: reward=1的数量，表示从step (100 - reward_one_count)开始，所有reward都是1
        gamma: 折扣因子
    
    返回:
        所有step（0-99）Q值的平均值
    """
    trajectory_length = 100
    total_q = 0.0
    
    # 计算每一步的Q值并累加
    for step in range(trajectory_length):
        total_q += compute_q_value(reward_one_count, step, gamma)
    
    # 返回平均值
    return total_q / trajectory_length


# 示例使用
if __name__ == "__main__":
    # 测试参数
    reward_one_count = 73  # 最后50步reward为1
    gamma = 0.95
    
    # 测试计算某个step的Q值
    step = 1
    q_value = compute_q_value(reward_one_count, step, gamma)
    print(f"Step {step}的Q值: {q_value:.4f}")
    
    # 测试计算平均Q值
    avg_q = compute_average_q_value(reward_one_count, gamma)
    print(f"所有step的平均Q值: {avg_q:.4f}")
    
    # 测试边界情况
    print("\n边界情况测试:")
    print(f"Step 0的Q值: {compute_q_value(reward_one_count, 0, gamma):.4f}")
    print(f"Step 49的Q值: {compute_q_value(reward_one_count, 49, gamma):.4f}")
    print(f"Step 99的Q值: {compute_q_value(reward_one_count, 99, gamma):.4f}")
