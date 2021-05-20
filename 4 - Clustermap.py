import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


path = "C:/Wz Python Projects/Wonder_tk_MSc/DPES Rhoda Final/Specific Objective 3.xlsx"

dataset = pd.read_excel(path)


# Hierarchically clustered heatmap
sns.clustermap(dataset, cmap='Dark2_r')
plt.show()
