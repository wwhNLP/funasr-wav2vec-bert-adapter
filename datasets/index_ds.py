import os
import json
import torch
import logging

import librosa
import random
import torch.distributed as dist

from funasr.register import tables


@tables.register("index_ds_classes", "SelfSupervisedIndexDSJsonl")
class SelfSupervisedIndexDSJsonl(torch.utils.data.Dataset):
    """
    PyTorch Dataset，用于加载 JSONL 或 JSON 格式的数据集。
    数据可以包含源（source）、目标（target）、文本、语言、情感标签等字段。支持长度过滤、数据分片，以及多种额外字段。
    最小子集："source",  "source_len", 
    """
    def __init__(self, path: str, **kwargs):
        """
        初始化数据集。
        Args:
            path (str): JSONL 文件路径或包含多个 JSON 文件路径的列表文件。
            kwargs:
                max_source_length (int): 源数据最大长度，默认 2048
                min_source_length (int): 源数据最小长度，默认 0
        功能:
            1. 读取 JSONL 或 JSON 文件列表
            2. 根据长度过滤 source/target
            3. 构建标准化字典 self.contents
        """
        super().__init__()
        self.max_source_length = kwargs.get("max_source_length", 2048)
        self.min_source_length = kwargs.get("min_source_length", 0)

        is_training = kwargs.get("is_training", True)
        # 判断 path 类型，分片处理
        if not (path.endswith(".jsonl") or path.endswith(".json")):
            # jsonl list file
            data_split_num = kwargs.get("data_split_num", 1)
            data_split_i = kwargs.get("data_split_i", 0)

            if not is_training:
                data_split_num = 1
                data_split_i = 0
            with open(path, encoding="utf-8") as fin:
                file_list_all = fin.readlines()

                num_per_slice = (len(file_list_all) - 1) // data_split_num + 1  # 16
                file_list = file_list_all[
                    data_split_i * num_per_slice : (data_split_i + 1) * num_per_slice
                ]
                logging.info(
                    f"is_training: {is_training}, data_split_num: {data_split_num}, data_split_i: {data_split_i}, \nfile_list: {file_list}, \nfile_list_all: {file_list_all}"
                )

        else:
            file_list = [path]

        # 遍历文件，解析每条 JSON 样本并过滤
        contents = []
        for file_json in file_list:
            with open(file_json.strip(), encoding="utf-8") as fin:
                for line in fin:
                    data = json.loads(line.strip())
                    if "source" in data:  # for speech lab pretrain
                        source = data["source"].replace(
                            "/cpfs01", "/cpfs_speech/data"
                        )  # only use in alibaba gpu group: .replace("/cpfs01", "/cpfs_speech/data")
                        
                        source_len = data.get("source_len", 1)
                        if (
                            source_len < self.min_source_length
                            or source_len > self.max_source_length
                        ):
                            continue
                        contents_i = {
                            # 构建标准化字典
                            "source": source,
                            "source_len": source_len,
                        }
                        contents.append(contents_i)

        self.contents = contents

        logging.info("total_num of samplers: {}, {}".format(len(self.contents), path))

    def __len__(self):
        return len(self.contents)

    def __getitem__(self, index):

        data = self.contents[index]

        return data

    def get_source_len(self, data_dict):
        return data_dict.get("source_len", 1)
