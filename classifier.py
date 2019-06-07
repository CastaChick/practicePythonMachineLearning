import numpy as np
from numpy.random import seed
class Perceptron(object):  # パーセプトロン分類器
    """
    パラメータ
    ------------
    eta: float
        学習率
    n_iter: int
        トレーニング回数

    属性
    ------------
    w_: ベクトル
        学習後の重みベクトル
    errors_: リスト
        各エポックでの誤分類数

    末尾に_がついているものはオブジェクトの初期化時ではなくメソッドの呼び出しにより作成される
    """
    def __init__(self, eta = 0.01, n_iter = 10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):  # モデルをデータに適合させる
        """
        パラメータ
        ---------------
        X: 行列 shape = [n_samples, n_features]
            トレーニングデータ
            サンプル数*特徴量数の次元を持つ
        y: ベクトル len = n_samples
            正解ラベル

        戻り値
        ---------------
        self: object
        """
        self.w_ = np.zeros(1 + X.shape[1])  # 切片を含めた[特徴量+1]次元
        self.errors_ = []

        for _ in range(self.n_iter):  # トレーニングを反復
            errors = 0
            for xi, target in zip(X, y):  # 各サンプルについて重みを更新
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0)
            self.errors_.append(errors)

        return self

    def net_input(self, X):
        """総入力を計算"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """
        1ステップ後のクラスラベルを返す
        Xが行列の時各サンプルについての予測値を並べたベクトルを返す
        Xがあるサンプルについての説明変数の組のベクトルの時そのサンプルについての予測値を返す
        """
        return np.where(self.net_input(X) >= 0.0, 1, -1)

class AdalineGD(object):  # ADALINE分類器
    """
    パラメータ
    ---------------
    eta: float
        学習率
    n_iter: int
        トレーニング回数

    属性
    ---------------
    w_: ベクトル
        学習後の重みベクトル
    errors_: リスト
        各エポックでの誤分類数
    """
    def __init__(self, eta = 0.01, n_iter = 50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):  # モデルの適合
        """
        パラメータ
        ---------------
        X: 行列 shape = [n_samples, n_features]
            トレーニングデータ
            サンプル数*特徴量数の次元を持つ
        y: ベクトル len = n_samples
            正解ラベル

        戻り値
        ---------------
        self: object
        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []  # 各サンプルについてコストを格納
        for i in range(self.n_iter):
            output = self.net_input(X)  # 活性化関数の出力, outputはlen = n_samplesのベクトル
            errors = y - output  # 誤差の計算
            """
            w1 ~ wnを更新
            Δwj = η * Σerrors[i] * xj[i]
            """
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()  # Δw0は学習率*誤差の総和

            """コスト関数の計算, 格納"""
            cost = (errors**2).sum() / 2.0  # Sum of Squared Error: SSE
            self.cost_.append(cost)

        return self

    def net_input(self, X):
        """総入力を計算"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """活性化関数"""
        return self.net_input(X)

    def predict(self, X):
        """1ステップ後のクラスラベルを返す"""
        return np.where(self.activation(X) >= 0.0, 1, -1)

class AdalineSGD(AdalineGD):  # 確立的勾配降下法で重みを更新するADALINE分類器
    """
    AdalineGDを継承

    追加の属性
    ---------------
    shuffle: bool (デフォルト: True)
        循環を回避するためにトレーニングデータをシャッフル
    random_state: int (デフォルト: None)
        シャッフルに使用するseedを設定
    """
    def __init__(self, eta = 0.01, n_iter = 10, shuffle = True, random_state = None):
        super().__init__(eta, n_iter)
        self.shuffle = shuffle
        if random_state:
            seed(random_state)

    def fit(self, X, y):
        """
        基本的に親と同じ
        重みベクトルの更新を_update_weightsで行う
        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []  # 各サンプルのコストを格納

            """各サンプルについて計算, 重み更新"""
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))

            ave_cost = sum(cost) / len(y)  # 各エポックのコストはサンプルの平均コストとする
            self.cost_.append(ave_cost)
        return self

    def _shuffle(self, X, y):
        """トレーニングデータをシャッフル"""
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def _update_weights(self, xi, target):
        """重みを更新し, そのサンプルの誤差の二乗/2をコストとして返す"""
        output = self.net_input(xi)
        error = target - output
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error ** 2
        return cost


