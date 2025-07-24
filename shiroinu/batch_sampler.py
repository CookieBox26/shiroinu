import numpy as np


class BatchSampler:
    # データを全くシャッフルせず順にバッチに切り出していく基本のバッチサンプラーです.
    argnames = ['batch_size']
    @classmethod
    def filter_kwargs(cls, kwargs):
        if type(kwargs) is int:
            return {'batch_size': kwargs}
        return {k: v for k, v in kwargs.items() if k in cls.argnames}
    def __init__(self, n_sample, batch_size, print_info=False):
        self.n_sample = n_sample
        self.batch_size = batch_size
        self.n_batch = int(np.ceil(self.n_sample / batch_size))
        if print_info:
            print(f'===== {self.__class__.__name__} instantiated. =====')
            print(f'{self.n_sample=}')
            print(f'{self.batch_size=}')
            print(f'{self.n_batch=}')
    def __len__(self):
        return self.n_batch
    def __iter__(self):
        self.i_batch_actual = -1
        return self
    def _get_i_batch(self, i_batch):
        # 順にバッチに切り出していくとき i 番目のバッチに所属するサンプルインデックスのリストを返します.
        # i <= n_batch - 1 であることはコール側で保証してください.
        list_indices = [i_batch * self.batch_size + i for i in range(self.batch_size)]
        if i_batch == self.n_batch - 1:
            list_indices = [i for i in list_indices if i <= self.n_sample - 1]
        return list_indices
    def __next__(self):
        self.i_batch_actual += 1
        if self.i_batch_actual == self.n_batch:
            raise StopIteration()
        return self._get_i_batch(self.i_batch_actual)


class BatchSamplerShuffle(BatchSampler):
    # データを全てシャッフルするバッチサンプラーです.
    # 各バッチ内もばらばらなサンプルインデックスのデータの集まりになります.
    def __init__(self, n_sample, batch_size):
        super().__init__(n_sample, batch_size)
        # サンプルインデックスの列をかきまぜておきます.
        self.sample_ids_shuffled = [i for i in range(self.n_sample)]
    def __iter__(self):
        np.random.shuffle(self.sample_ids_shuffled)
        return super().__iter__()
    def __next__(self):
        # 基底クラスの出力を得てから, かきまぜたサンプルインデックスにマッピングして返します.
        list_indices = super().__next__()
        return [self.sample_ids_shuffled[i] for i in list_indices]


class BatchSamplerBatchShuffle(BatchSampler):
    # バッチ順序だけシャッフルするバッチサンプラーです.
    # 各バッチ内は隣接したサンプルインデックスのデータでまとまります.
    def __init__(self, n_sample, batch_size):
        super().__init__(n_sample, batch_size)
        # バッチインデックスの列をかきまぜておきます.
        self.batch_ids_shuffled = [i for i in range(self.n_batch)]
        np.random.shuffle(self.batch_ids_shuffled)
    def __next__(self):
        self.i_batch_actual += 1
        if self.i_batch_actual == self.n_batch:
            raise StopIteration()
        # 現在の何番目のバッチかを, かきまぜたバッチインデックスにマッピングします.
        return self._get_i_batch(self.batch_ids_shuffled[self.i_batch_actual])


class BatchSamplerPeriodic(BatchSampler):
    # 自然数 period に関して合同なインデックスどうしのみ同じバッチにすることを許可するバッチサンプラーです.
    # つまり, 日次データで priod=7 とすれば, バッチ内が全て同じ曜日になります.
    # いまのところ, 連続する同じ曜日のデータが同じバッチになります.
    argnames = ['batch_size', 'period']
    def __init__(self, n_sample, batch_size, period):
        super().__init__(n_sample, batch_size, print_info=False)
        # すべてのサンプルインデックスを period で割ったあまりが同じものたちにグループ分けします.
        self.sample_ids_grouped = [[] for r in range(period)]
        for i_sample in range(n_sample):
            self.sample_ids_grouped[i_sample % period].append(i_sample)
        # 改めて以下の変数を取ります.
        self.n_batch = 0  # バッチ総数を再計算します (バッチがグループをまたがない制約のため増える可能性があります).
        self.batch_ids_shuffled = []  # グループ番号とグループ内バッチ番号の組をすべて格納します.
        for r in range(period):
            n_batch_ = int(np.ceil(len(self.sample_ids_grouped[r]) / batch_size))
            self.n_batch += n_batch_
            self.batch_ids_shuffled += [(r, i_batch_) for i_batch_ in range(n_batch_)]
        np.random.shuffle(self.batch_ids_shuffled)  # かきまぜます.
        if print_info:
            print(f'===== {self.__class__.__name__} instantiated. =====')
            print(f'{self.n_sample=}')
            print(f'{self.batch_size=}')
            print(f'{period=}')
            print(f'{self.n_batch=}')
    def _get_i_batch(self, r, i_batch_):
        # r グループの i_batch_ 番目のバッチに所属するサンプルインデックスのリストを返します.
        sample_ids_ = self.sample_ids_grouped[r]
        list_indices = [
            sample_ids_[i_batch_ * self.batch_size + i] for i in range(self.batch_size)
            if i_batch_ * self.batch_size + i <= len(sample_ids_) - 1]
        return list_indices
    def __next__(self):
        self.i_batch_actual += 1
        if self.i_batch_actual == self.n_batch:
            raise StopIteration()
        r, i_batch_ = self.batch_ids_shuffled[self.i_batch_actual]
        return self._get_i_batch(r, i_batch_)
