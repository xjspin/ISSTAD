import numpy as np

# 创建一个包含 float 数据的 NumPy 数组（示例数据）
data = np.random.rand(224, 224)  # 示例数据

# 计算数据的最小值和最大值
min_value = np.min(data)
max_value = np.max(data)

# 对数据进行归一化操作
normalized_data = (data - min_value) / (max_value - min_value)

# 现在，normalized_data 数组中的值将在 [0, 1] 范围内

# 将 NumPy 数组转换为 PIL 图像对象
from PIL import Image

# 将数据缩放到 [0, 255] 范围以保存为图像
scaled_data = (normalized_data * 255.0).astype(np.uint8)
image = Image.fromarray(scaled_data, mode='L')  # 'L' 表示灰度图像

# 保存为 TIFF 文件
image.save('./outtest/output.tif', 'TIFF')

# 关闭图像对象
image.close()