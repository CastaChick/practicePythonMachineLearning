from classifier import Perceptron
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

"""
# setosaを赤丸,versicolorを青バツで散布図を作成
plt.scatter(X[:50, 0], X[:50, 1], color = 'red', marker = 'o', label = 'setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color = 'blue', marker = 'x', label = 'versicolor')
plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.grid()
plt.legend(loc = 'upper left')
plt.show()
"""

# X, yを用いてパーセプトロンをトレーニング
ppn = Perceptron(0.01, 10)
ppn.fit(X, y)

"""
# 各エポックのエラー数をプロット
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker = 'o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()
"""

"""plot_decision_regionsを用いて決定領域をプロット"""
pdr(X, y, classifier = ppn)
plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.legend(loc = 'upper left')
plt.show()


