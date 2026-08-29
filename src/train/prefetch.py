# -*- coding: utf-8 -*-
"""单线程预取迭代器：后台线程准备 batch（含增强），主线程只做 H2D + GPU 计算，
让 CPU 准备与 GPU 计算重叠。Windows 上替代 DataLoader(num_workers>0)
（spawn 进程会整份拷贝内存数据集 ~10GB，不可行；线程共享内存无此问题）。"""
import queue
import threading
from torch.utils.data._utils.collate import default_collate


class PrefetchLoader:
    def __init__(self, dataset, batch_size, prefetch=2, drop_last=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.prefetch = prefetch
        self.drop_last = drop_last

    def __len__(self):
        n = len(self.dataset) // self.batch_size
        if not self.drop_last and len(self.dataset) % self.batch_size:
            n += 1
        return n

    def __iter__(self):
        ds = self.dataset
        n_batches = len(self)
        q = queue.Queue(maxsize=self.prefetch)

        def worker():
            for b in range(n_batches):
                s = b * self.batch_size
                s1 = min(s + self.batch_size, len(ds))
                batch = default_collate([ds[i] for i in range(s, s1)])
                q.put(batch)
            q.put(None)

        threading.Thread(target=worker, daemon=True).start()
        while True:
            batch = q.get()
            if batch is None:
                return
            yield batch
