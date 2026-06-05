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


@tables.register("dataloader_classes", "DataloaderIterable")
class DataloaderIterable:
    def __init__(self, frontend=None, tokenizer=None, **kwargs):
        logging.info("Build dataloader")
        self.kwargs = kwargs
        self.frontend = frontend
        self.tokenizer = tokenizer
        self.data_split_num = 1 # iter style does not support split

        self.train_data_set_list = kwargs.get("train_data_set_list")
        dataset_conf = kwargs.get("dataset_conf", {})
        estimated_total_samples = dataset_conf.get("estimated_total_samples")
        estimated_samples_per_shard = dataset_conf.get("estimated_samples_per_shard")

        try:
            with open(self.train_data_set_list, 'r', encoding="utf-8") as f:
                num_shards = len(f.readlines())
        except Exception as e:
            logging.warning(f"Failed to count samples in {self.train_data_set_list}: {e}")
            num_shards = 0

        if estimated_total_samples is not None:
            self.num_samples = int(estimated_total_samples)
        elif estimated_samples_per_shard is not None and num_shards > 0:
            self.num_samples = int(num_shards * estimated_samples_per_shard)
        else:
            self.num_samples = num_shards # fallback to shard count for streaming data

    def build_iter(self, epoch=0, data_split_i=0, start_step=0, **kwargs):
        dataset_class = tables.dataset_classes.get(self.kwargs.get("dataset", "LargeAudioDataset"))
        
        dataset_tr = dataset_class(
            self.train_data_set_list,
            frontend=self.frontend,
            tokenizer=self.tokenizer,
            is_training=True,
            **self.kwargs.get("dataset_conf"),
        )

        dataset_val = dataset_class(
            self.kwargs.get("valid_data_set_list"),
            frontend=self.frontend,
            tokenizer=self.tokenizer,
            is_training=False,
            **self.kwargs.get("dataset_conf"),
        )
        
        num_workers = self.kwargs.get("dataset_conf", {}).get("num_workers", 1)
        prefetch_factor = self.kwargs.get("dataset_conf", {}).get("prefetch_factor", 2)
        persistent_workers = self.kwargs.get("dataset_conf", {}).get("persistent_workers", True)
        pin_memory = self.kwargs.get("dataset_conf", {}).get("pin_memory", True)
        
        dataloader_tr = torch.utils.data.DataLoader(
            dataset_tr,
            batch_size=None,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers if num_workers > 0 else False,
        )

        dataloader_val = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=None,
            num_workers=self.kwargs.get("dataset_conf", {}).get("num_workers", 0),
            pin_memory=self.kwargs.get("dataset_conf", {}).get("pin_memory", False),
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers if num_workers > 0 else False,
        )

        return dataloader_tr, dataloader_val