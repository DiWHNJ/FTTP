from datasets import load_dataset

# 尝试加载一个数据集
try:
    dataset = load_dataset("imdb")
    print("成功访问 Hugging Face Hub 并加载数据集！")
except Exception as e:
    print(f"访问 Hugging Face Hub 失败：{e}")

