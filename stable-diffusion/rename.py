import os

# 指定要处理的目录
directory = '/data2/ljq/do_train/stable-diffusion/rel_image'

# 获取目录下的所有文件，并过滤出图片文件（假设处理所有的图片格式）
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
images = [f for f in os.listdir(directory) if f.lower().endswith(valid_extensions)]

# 按名称排序
images.sort()

# 遍历图片并重命名
for i, image in enumerate(images):
    # 获取图片的完整路径
    old_path = os.path.join(directory, image)
    
    # 创建新的文件名，如 2_1.png, 2_2.png, ...
    new_name = f'2_{i+1}.png'
    new_path = os.path.join(directory, new_name)
    
    # 重命名文件
    os.rename(old_path, new_path)

print("重命名完成！")
