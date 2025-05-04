def find_nearest_point(new_point, X, mapped_labels):
    """
    找出距离新数据点最近的训练数据点及其映射标签
    
    参数:
    new_point - 一个新的数据点，形状为(1, 2)的数组
    X - 所有训练数据点的坐标，形状为(n_samples, 2)的数组
    mapped_labels - 所有训练数据点的映射标签，长度为n_samples的数组
    
    返回:
    nearest_point - 距离最近的训练数据点坐标
    nearest_label - 该点的映射标签
    distance - 最小距离
    """
    import numpy as np
    
    # 确保输入是numpy数组并且维度正确
    new_point = np.array(new_point).reshape(1, -1)
    
    # 计算欧氏距离
    distances = np.sqrt(np.sum((X - new_point)**2, axis=1))
    
    # 找出最小距离的索引
    min_index = np.argmin(distances)
    
    # 获取最近点、其标签和距离
    nearest_point = X[min_index]
    nearest_label = mapped_labels[min_index]
    min_distance = distances[min_index]
    
    return nearest_point, nearest_label, min_distance