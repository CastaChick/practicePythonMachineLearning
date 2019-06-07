from classifier import AdalineGD, AdalineSGD
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from decisionPlot import plot_decision_regions as pdr

# pandasでirisデータを取得
df = pd.read_csv('https://archive.ics.uci.edu/ml/'+'machine-learning-databases/iris/iris.data', header = None)

# 1-100行目の目的変数の抽出,(1, -1)のラベルを付与
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

# 説明変数の抽出. 1, 3行目を使用
X = df.iloc[0:100, [0, 2]].values

# データを標準化してADALINEを適用
X_std = np.copy(X)
# 平均0, 標準偏差1に変換
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

ada = AdalineGD(eta = 0.01, n_iter = 15) #  バッチ勾配降下法
ada.fit(X_std, y)

adas = AdalineSGD(eta = 0.01, n_iter = 15)  # 確立的勾配降下法
adas.fit(X_std, y)

# バッチ勾配降下法と確立的勾配降下法の比較
plt.figure(figsize = (8, 8))
plt.subplots_adjust(wspace = 0.4, hspace = 0.6)
plt.subplot(221)
# plot_decision_regionsを用いて決定領域をプロット
pdr(X_std, y, classifier = ada)
plt.title('ADALINE - Gradient Descent')
plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.legend(loc = 'upper left')
plt.subplot(222)
# エポック毎のコストをプロット
plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker = 'o')
plt.xlim(0, 16)
plt.xticks(range(0, 18, 2))
plt.ylim(0, max(ada.cost_)*1.1)
plt.xlabel('Epoch')
plt.ylabel('SSE')
plt.subplot(223)
# plot_decision_regionsを用いて決定領域をプロット
pdr(X_std, y, classifier = adas)
plt.title('ADALINE - Stochastic Gradient Descent')
plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.legend(loc = 'upper left')
plt.subplot(224)
# エポック毎のコストをプロット
plt.plot(range(1, len(adas.cost_) + 1), adas.cost_, marker = 'o')
plt.xlim(0, 16)
plt.xticks(range(0, 18, 2))
plt.ylim(0, max(adas.cost_)*1.1)
plt.xlabel('Epoch')
plt.ylabel('Average Cost')
plt.show()


