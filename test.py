import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 数据准备
data = [
    {'Dataset': 'Cora', 'Strategy': 'Overwriting', 'Original PPL': 84.28, 'Poisoned PPL': 179.77},
    {'Dataset': 'Cora', 'Strategy': 'Appending', 'Original PPL': 82.77, 'Poisoned PPL': 15.57},
    {'Dataset': 'PubMed', 'Strategy': 'Overwriting', 'Original PPL': 18.64, 'Poisoned PPL': 5.61},
    {'Dataset': 'PubMed', 'Strategy': 'Appending', 'Original PPL': 18.03, 'Poisoned PPL': 5.50},
    {'Dataset': 'Arxiv', 'Strategy': 'Overwriting', 'Original PPL': 45.41, 'Poisoned PPL': 5.97},
    {'Dataset': 'Arxiv', 'Strategy': 'Appending', 'Original PPL': 49.61, 'Poisoned PPL': 7.14},
]
df = pd.DataFrame(data)

# # 构造标签：如 "Cora-Ov", "Cora-App"
# df["Group"] = df["Dataset"] + "-" + df["Strategy"].str.slice(0, 3)
#
# # melt 原始 vs 投毒 PPL
# df_melt = df.melt(id_vars=["Group", "Dataset", "Strategy"],
#                   value_vars=["Original PPL", "Poisoned PPL"],
#                   var_name="Type", value_name="Perplexity")

# # 画图
# plt.figure(figsize=(10, 6))
# sns.barplot(data=df_melt, x="Group", y="Perplexity", hue="Type", palette="Set2")
# plt.title("Original vs. Poisoned Perplexity by Dataset and Strategy")
# plt.xlabel("Dataset-Strategy")
# plt.tight_layout()
# plt.show()

# 添加 perplexity delta
df["PPL Change"] = df["Poisoned PPL"] - df["Original PPL"]

# 绘图
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="Dataset", y="PPL Change", hue="Strategy", palette="Set2")
plt.axhline(0, color='gray', linestyle='--')
plt.title("Perplexity Change (Poisoned - Original) by Strategy")
plt.ylabel("Perplexity Change")
plt.tight_layout()
plt.show()
