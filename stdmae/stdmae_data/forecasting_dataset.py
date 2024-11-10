import os

import torch
from torch.utils.data import Dataset
from basicts.utils import load_pkl


class ForecastingDataset(Dataset):
    """Time series forecasting dataset."""

    def __init__(self, data_file_path: str, index_file_path: str, mode: str, seq_len:int) -> None:
        """Init the dataset in the forecasting stage.

        Args:
            data_file_path (str): data file path.
            index_file_path (str): index file path.
            mode (str): train, valid, or test.
            seq_len (int): the length of long term historical data.
        """

        super().__init__()
        assert mode in ["train", "valid", "test"], "error mode"
        self._check_if_file_exists(data_file_path, index_file_path)
        # read raw data (normalized)
        data = load_pkl(data_file_path)

        processed_data = data["processed_data"]
        # print("-------------modexxxx--------------",mode)
        # print("---------processed_data-----", processed_data.shape)
        # print("---------processed_data-----", processed_data[0,0,:])
        self.data = torch.from_numpy(processed_data).float()
        # read index
        # print("---------self.data-xxxxxx------",self.data.shape)
        self.index = load_pkl(index_file_path)[mode]
        # length of long term historical data
        self.seq_len = seq_len
        # print("----------self.seq_len------",self.seq_len)
        # mask
        # print("---------self.data.shape------", self.data.shape)
        # print("---------self.data.shape------", self.data[0,0,:])
        # print("-------------------------------------------")
        # print("---------self.data.shape[0]------", self.data.shape[0])
        # print("---------self.data.shape[1]------",self.data.shape[1])
        # print("---------self.data.shape[2]------", self.data.shape[2])
        self.mask = torch.zeros(self.seq_len, self.data.shape[1], self.data.shape[2])
        # print("---------self.mask------", self.mask.shape)

    def _check_if_file_exists(self, data_file_path: str, index_file_path: str):
        """Check if data file and index file exist.

        Args:
            data_file_path (str): data file path
            index_file_path (str): index file path

        Raises:
            FileNotFoundError: no data file
            FileNotFoundError: no index file
        """
        # print("--------------数据集这里走了吗--xxxx-------------------")
        if not os.path.isfile(data_file_path):
            raise FileNotFoundError("BasicTS can not find data file {0}".format(data_file_path))
        if not os.path.isfile(index_file_path):
            raise FileNotFoundError("BasicTS can not find index file {0}".format(index_file_path))

    def __getitem__(self, index: int) -> tuple:
        """Get a sample.

        Args:
            index (int): the iteration index (not the self.index)

        Returns:
            tuple: (future_data, history_data), where the shape of each is L x N x C.
        """

        idx = list(self.index[index])

        history_data = self.data[idx[0]:idx[1]]     # 12
        # print()
        # print("----------------idx[0]---------------", idx[0])
        # print("----------------idx[1]--------------", idx[1])
        future_data = self.data[idx[1]:idx[2]]      # 12
        # print("---------------idx[2]--------------", idx[2])
        if idx[1] - self.seq_len < 0:
            # print("-------------a--------------")
            long_history_data = self.mask
        else:
            long_history_data = self.data[idx[1] - self.seq_len:idx[1]]     # 11
            # print("-------------b--------------")  走的这里
            # print("----------------idx[1] - self.seq_len-------------",idx[1] - self.seq_len)
            # print("----------------idx[1]---------------", idx[1])



        # print("------------future_data----------",future_data.shape)
        # print("------------future_data----------", future_data[0,0,:])
        # print("-------------history_data------------", history_data.shape)
        # print("------------history_data----------", history_data[0, 0, :])
        # print("------------long_history_data-----------", long_history_data.shape)
        # print("------------long_history_data----------", long_history_data[0, 0, :])
        return future_data, history_data, long_history_data

    def __len__(self):
        """Dataset length

        Returns:
            int: dataset length
        """

        return len(self.index)
