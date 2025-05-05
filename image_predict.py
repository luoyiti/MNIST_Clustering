def convert_image_to_mnist_format(image, label=None):
    import cv2
    import numpy as np
    import pandas as pd
    """
    Convert an image to MNIST format.
    
    Parameters:
    image: Path to an image file or a numpy array of shape (height, width) or (height, width, channels)
    label: Optional digit label (0-9)
    
    Returns:
    pandas.DataFrame: A dataframe in MNIST format with 'label' column and 784 pixel columns
    """
    
    # Read the image if it's a file path
    if isinstance(image, str):
        img = cv2.imread(image)
    else:
        img = image.copy()
    
    # Convert to grayscale if it's color
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize to 28x28 if needed
    if img.shape != (28, 28):
        img = cv2.resize(img, (28, 28))
    
    # Invert if it's black digits on white background
    if img.mean() > 127:
        img = 255 - img
    
    # Create a dataframe with 'label' and 784 pixel columns
    data = {'label': label if label is not None else 0}
    
    # Add pixel values with the same column names as in the original MNIST data
    for i in range(28):
        for j in range(28):
            data[f"{i+1}x{j+1}"] = int(img[i, j])
    
    return pd.DataFrame([data])

def img_to_tsne(img, X_train, tsne_df):
    import numpy as np
    """
    通过寻找最相似的训练图像，将新图像映射到t-SNE空间
    
    参数:
    img - 28x28的图像数组或784维的向量
    X_train - 原始训练数据
    tsne_df - 包含t-SNE坐标的DataFrame
    
    返回:
    tsne_point - 预测的t-SNE坐标 [x, y]
    """
    # 确保img是一维向量
    if len(img.shape) > 1:
        img_vector = img.flatten()
    else:
        img_vector = img
        
    # 计算与训练集中所有图像的欧氏距离
    X_train_array = X_train.to_numpy()
    distances = np.sqrt(np.sum((X_train_array - img_vector)**2, axis=1))
    
    # 找到最相似图像的索引
    most_similar_idx = np.argmin(distances)
    
    # 获取该图像在t-SNE空间中的坐标
    tsne_point = tsne_df.iloc[most_similar_idx][['x', 'y']].values
    
    return tsne_point, most_similar_idx

def visualize_nearest_prediction(example_tsne, X, clustered_data, example, 
                                 figsize=(10, 8), title=None, save_path=None, show=True):
    """
    可视化新数据点及其在t-SNE空间中最近邻的预测结果
    
    参数:
    example_tsne - 新数据点的t-SNE坐标
    X - 所有t-SNE点的坐标，形状为(n_samples, 2)的数组
    clustered_data - 所有点的聚类标签数组
    example - 包含原始标签的数据对象
    figsize - 图像大小，默认为(10, 8)
    title - 图像标题，如果为None则使用默认标题
    save_path - 保存图像的路径，如果为None则不保存
    show - 是否显示图像，默认为True
    
    返回:
    tuple - (nearest_point, nearest_label, distance) 最近点信息
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from find_nearest_point import find_nearest_point
    
    # 找到最近的点及其标签
    nearest_point, nearest_label, distance = find_nearest_point(example_tsne, X, clustered_data)
    
    # 打印结果
    print(f"新数据点: {example_tsne[0]}")
    print(f"最近的训练点: {nearest_point}")
    print(f"该点的标签: {nearest_label}")
    print(f"距离: {distance:.4f}")
    
    # 创建可视化
    plt.figure(figsize=figsize)
    
    # 绘制所有数据点
    plt.scatter(X[:, 0], X[:, 1], c=clustered_data, cmap='tab10', alpha=0.3, s=10)
    
    # 绘制最近点
    plt.scatter(nearest_point[0], nearest_point[1], c=nearest_label, cmap='tab10', 
                s=200, edgecolors='black')
    
    # 绘制新数据点
    plt.scatter(example_tsne[0], example_tsne[1], c='red', marker='x', s=200)
    
    # 绘制连接线
    plt.plot([example_tsne[0], nearest_point[0]], [example_tsne[1], nearest_point[1]], 'k--')
    
    # 添加标注
    plt.annotate(f"Predicted Label: {nearest_label}\nOriginal Label: {example['label'].values[0]}",
                 xy=(example_tsne[0], example_tsne[1]), 
                 xytext=(example_tsne[0] + 5, example_tsne[1] + 5),
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3"))
    
    # 设置标题
    if title is None:
        title = f'Distance to Nearest Point: {distance:.4f}, Label: {nearest_label}'
    plt.title(title)
    
    # 添加颜色条和网格
    plt.colorbar(label='Digit Label')
    plt.grid(True, alpha=0.3)
    
    # 保存图像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # 显示图像
    if show:
        plt.show()
    
    return nearest_point, nearest_label, distance

