import logging
import torch

from funasr.register import tables


# @tables.register("dataloader_classes", "DataloaderMapStyle")
def DataloaderMapStyle(frontend=None, tokenizer=None, **kwargs):
    # dataset
    logging.info("Build dataloader")
    dataset_class = tables.dataset_classes.get(kwargs.get("dataset", "AudioDataset"))
    dataset_tr = dataset_class(
        kwargs.get("train_data_set_list"),
        frontend=frontend,
        tokenizer=tokenizer,
        is_training=True,
        **kwargs.get("dataset_conf"),
    )
    dataset_val = dataset_class(
        kwargs.get("valid_data_set_list"),
        frontend=frontend,
        tokenizer=tokenizer,
        is_training=False,
        **kwargs.get("dataset_conf"),
    )

    # sample
    # 取出batch_sampler配置
    batch_sampler = kwargs["dataset_conf"].get("batch_sampler", "BatchSampler")
    batch_sampler_val = None
    # 如果batch_sampler不为空，则创建batch_sampler对象和batch_sampler_val对象
    if batch_sampler is not None:
        batch_sampler_class = tables.batch_sampler_classes.get(batch_sampler)
        # 创建batch_sampler对象
        batch_sampler = batch_sampler_class(dataset_tr, **kwargs.get("dataset_conf"))
        # 创建batch_sampler_val对象
        batch_sampler_val = batch_sampler_class(
            dataset_val, is_training=False, **kwargs.get("dataset_conf")
        )

    # dataloader
    dataloader_tr = torch.utils.data.DataLoader(
        dataset_tr, collate_fn=dataset_tr.collator, **batch_sampler
    )

    dataloader_val = torch.utils.data.DataLoader(
        dataset_val, collate_fn=dataset_val.collator, **batch_sampler_val
    )

    return dataloader_tr, dataloader_val


@tables.register("dataloader_classes", "DataloaderMapStyle")
class DataloaderMapStyle:
    def __init__(self, frontend=None, tokenizer=None, **kwargs):
        # dataset
        logging.info("Build dataloader")

        dataset_class = tables.dataset_classes.get(kwargs.get("dataset", "AudioDataset"))
        dataset_tr = None
        # split dataset
        self.data_split_num = kwargs["dataset_conf"].get("data_split_num", 1)
        if self.data_split_num == 1:
            dataset_tr = dataset_class(
                kwargs.get("train_data_set_list"),
                frontend=frontend,
                tokenizer=tokenizer,
                is_training=True,
                **kwargs.get("dataset_conf"),
            )
        dataset_val = dataset_class(
            kwargs.get("valid_data_set_list"),
            frontend=frontend,
            tokenizer=tokenizer,
            is_training=False,
            **kwargs.get("dataset_conf"),
        )

        self.dataset_tr = dataset_tr
        self.dataset_val = dataset_val
        self.kwargs = kwargs

        self.dataset_class = dataset_class
        self.frontend = frontend
        self.tokenizer = tokenizer
        self.kwargs = kwargs
        
        # 计算训练样本数量（trainer需要）
        if dataset_tr is not None:
            self.num_samples = len(dataset_tr)
        else:
            # 如果数据集被分片，尝试从文件计算
            train_data_path = kwargs.get("train_data_set_list")
            try:
                with open(train_data_path, 'r', encoding="utf-8") as f:
                    self.num_samples = len(f.readlines())
            except Exception as e:
                logging.warning(f"Failed to count samples: {e}")
                self.num_samples = 0

    def build_iter(self, epoch=0, data_split_i=0, start_step=0, **kwargs):

        # reload dataset slice，更新dataset_tr
        dataset_tr = self.dataset_tr
        if self.data_split_num > 1:
            # del self.dataset_tr
            dataset_tr = self.dataset_class(
                self.kwargs.get("train_data_set_list"),
                frontend=self.frontend,
                tokenizer=self.tokenizer,
                is_training=True,
                **self.kwargs.get("dataset_conf"),
                data_split_i=data_split_i,
            )

        # dataloader
        batch_sampler = self.kwargs["dataset_conf"].get("batch_sampler", "BatchSampler")
        batch_sampler_val = None
        if batch_sampler is not None:
            batch_sampler_class = tables.batch_sampler_classes.get(batch_sampler)
            batch_sampler = batch_sampler_class(
                dataset_tr, start_step=start_step, **self.kwargs.get("dataset_conf")
            )
            batch_sampler_val = batch_sampler_class(
                self.dataset_val, is_training=False, **self.kwargs.get("dataset_conf")
            )

        batch_sampler["batch_sampler"].set_epoch(epoch)
        batch_sampler_val["batch_sampler"].set_epoch(epoch)
        dataloader_tr = torch.utils.data.DataLoader(
            dataset_tr, collate_fn=dataset_tr.collator, **batch_sampler
        )
        dataloader_val = torch.utils.data.DataLoader(
            self.dataset_val, collate_fn=self.dataset_val.collator, **batch_sampler_val
        )

        return dataloader_tr, dataloader_val


class EpochDataLoader:
    """Give streaming ranks equal step counts and skip consumed batches on resume."""

    def __init__(self, loader, steps, start_step=0):
        self.loader = loader
        self.batch_sampler = self
        self.steps = int(steps)
        self.start_step = int(start_step)
        if self.steps <= 0 or not 0 <= self.start_step <= self.steps:
            raise ValueError("Invalid streaming epoch steps or resume offset.")

    def set_epoch(self, epoch):
        dataset = self.loader.dataset
        if getattr(dataset, "is_training", False) and hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)

    def __len__(self):
        return self.steps - self.start_step

    def __iter__(self):
        iterator = iter(self.loader)
        for step in range(self.steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(self.loader)
                try:
                    batch = next(iterator)
                except StopIteration as exc:
                    raise RuntimeError("No usable audio batches on this rank; check shards and length filters.") from exc
            if step >= self.start_step:
                yield batch


@tables.register("dataloader_classes", "DataloaderIterable")
class DataloaderIterable:
    def __init__(self, frontend=None, tokenizer=None, **kwargs):
        self.kwargs = kwargs
        self.frontend = frontend
        self.tokenizer = tokenizer
        self.data_split_num = 1
        self.train_data_set_list = kwargs.get("train_data_set_list")
        conf = kwargs.get("dataset_conf", {})
        with open(self.train_data_set_list, encoding="utf-8") as stream:
            num_shards = sum(bool(line.strip()) for line in stream)
        self.num_samples = int(conf.get("estimated_total_samples", num_shards * conf.get("estimated_samples_per_shard", 1)))

    def build_iter(self, epoch=0, data_split_i=0, start_step=0, **kwargs):
        dataset_class = tables.dataset_classes[self.kwargs["dataset"]]
        conf = dict(self.kwargs.get("dataset_conf", {}))
        num_workers = conf.get("num_workers", 0)
        loaders = []
        for training, path in [(True, self.train_data_set_list), (False, self.kwargs["valid_data_set_list"])]:
            dataset = dataset_class(path, frontend=self.frontend, tokenizer=self.tokenizer,
                                    is_training=training, **conf)
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(epoch if training else 0)
            if hasattr(dataset, "shard_list") and len(dataset.shard_list) < getattr(dataset, "world_size", 1):
                raise ValueError("Provide at least one tar shard per distributed rank.")
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=None, num_workers=num_workers,
                pin_memory=conf.get("pin_memory", True),
                prefetch_factor=conf.get("prefetch_factor", 2) if num_workers else None,
                persistent_workers=conf.get("persistent_workers", True) if num_workers else False,
                generator=torch.Generator().manual_seed(conf.get("seed", 0) + (epoch if training else 0)),
            )
            steps = conf["batch_num_epoch"] if training else conf.get("valid_batch_num_epoch", conf["batch_num_epoch"])
            loaders.append(EpochDataLoader(loader, steps, start_step if training else 0))
        return tuple(loaders)
