from pde_matching.data.trajectory import DataTrajectory
from pde_matching.envs import Env
from torch_geometric.data.lightning.datamodule import LightningDataModule
from torch_geometric.loader import DataLoader


class OfflineTrajectoryDataModule(LightningDataModule):
    def __init__(
        self,
        env,
        batch_size: int,
        root: str,
        window_size: int,
        window_shift: int,
        val_window_size: int,
        val_window_shift: int,
        train_n_sequences: int = None,
        val_n_sequences: int = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        env_kwargs: dict = None,
        **kwargs,
    ):
        super().__init__(has_val=True, has_test=True, **kwargs)

        self.dataset_cls = Env.get_env(env)
        self.batch_size = batch_size
        self.root = root
        self.window_size = window_size
        self.window_shift = window_shift
        self.val_window_size = val_window_size
        self.val_window_shift = val_window_shift
        self.train_n_sequences = train_n_sequences
        self.val_n_sequences = val_n_sequences
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.env_kwargs = env_kwargs

    @property
    def standardizer(self):
        return self._standardizer

    @property
    def input_keys(self):
        return self._input_keys

    @property
    def output_keys(self):
        return self._output_keys

    @property
    def loss_mask(self):
        return self._loss_mask

    @property
    def output_mask(self):
        return self._output_mask

    def setup(self, stage=None):
        if stage in (None, "train", "fit"):
            self.train_data = self.dataset_cls(
                root=self.root,
                stage="train",
                window_size=self.window_size,
                window_shift=self.window_shift,
                n_sequences=self.train_n_sequences,
                **self.env_kwargs,
            )
        if stage in (None, "validate", "fit"):
            self.val_data = self.dataset_cls(
                root=self.root,
                stage="val",
                window_size=self.val_window_size,
                window_shift=self.val_window_shift,
                n_sequences=self.val_n_sequences,
                **self.env_kwargs,
            )
        if stage in (None, "test"):
            self.test_data = self.dataset_cls(
                root=self.root,
                stage="test",
                window_size=self.val_window_size,
                window_shift=self.val_window_shift,
                n_sequences=self.val_n_sequences,
                **self.env_kwargs,
            )

        # All datasets share the same standardizer
        train_data = self.train_data
        if train_data is None:
            train_data = self.dataset_cls(
                root=self.root,
                stage="train",
                window_size=self.window_size,
                window_shift=self.window_shift,
                n_sequences=self.train_n_sequences,
                **self.env_kwargs,
            )

        self._standardizer = train_data.standardizer
        self._input_keys = train_data.input_keys
        self._output_keys = train_data.output_keys
        self._loss_mask = train_data.loss_mask
        self._output_mask = train_data.output_mask

    def train_dataloader(self):
        # Due to the lazy loading of the dataset from files,
        # we manually shuffle the data in the training loop
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def get_interesting_batch(self, batch_size: int = None, stage: str = "val"):
        if stage == "val":
            return self.val_data.get_interesting_batch(batch_size)
        elif stage == "test":
            return self.test_data.get_interesting_batch(batch_size)
        else:
            raise ValueError(f"Unknown stage: {stage}")

    def get_example_data(self):
        data = self.dataset_cls(
            root=self.root,
            stage="test",
            window_size=2,
            window_shift=1,
            n_sequences=2,
            **self.env_kwargs,
        )

        return DataTrajectory.from_batch_list(
            data[0], data.input_keys, data.output_keys, data.standardizer
        )
