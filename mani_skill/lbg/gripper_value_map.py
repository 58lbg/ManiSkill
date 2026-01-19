def action_to_joint(x, xmin=0.0559, xmax=0.11899):
    # 1. 限幅
    x_clamped = max(xmin, min(xmax, x))

    # 2. 映射
    y = 2 * (x_clamped - xmin) / (xmax - xmin) - 1
    return y

print(action_to_joint(0.08))