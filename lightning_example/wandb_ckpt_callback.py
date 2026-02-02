import glob, shutil
import os, wandb, pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

class UploadCkpt(pl.Callback):

    def __init__(self, model_checkpoint: ModelCheckpoint):
        super().__init__()
        self._model_checkpoint = model_checkpoint
        self.dirpath = model_checkpoint.dirpath
        self.sync_dir = os.path.join(self.dirpath, "wandb_ckpt")
        if not os.path.exists(self.sync_dir):
            os.makedirs(self.sync_dir)
        self.k = model_checkpoint.save_top_k

    def _atomic_copy(self, src: str, dst: str):
        tmp = dst + ".tmp"
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)  # atomic within the same filesystem

    def _stage(self):
        if not (wandb.run and os.path.isdir(self.dirpath)):
            return

        # 1) best_k from PL's tracker
        items = list(self._model_checkpoint.best_k_models.items())  # [(path, score), ...]
        if not items:
            self._stage_last_only()
        else:
            reverse = (self._model_checkpoint.mode == "max")
            items.sort(key=lambda kv: float(kv[1]), reverse=reverse)
            if self.k is not None and self.k > 0:
                items = items[: self.k]

            for i, (src, _) in enumerate(items, 1):
                if os.path.isfile(src):
                    dst = os.path.join(self.sync_dir, f"best_{i}.ckpt")
                    self._atomic_copy(src, dst)

            # 2) rolling last.ckpt (optional)
            last_src = os.path.join(self.dirpath, "last.ckpt")
            if os.path.isfile(last_src):
                self._atomic_copy(last_src, os.path.join(self.sync_dir, "last.ckpt"))

        # 3) upload (no artifacts): keep a stable relative path
        #    ensures files appear under "wandb_ckpt/" in the run
        wandb.save(os.path.join(self.sync_dir, "*.ckpt"),
                   base_path=self.dirpath, policy="now")

    def _stage_last_only(self):
        last_src = os.path.join(self.dirpath, "last.ckpt")
        if os.path.isfile(last_src):
            self._atomic_copy(last_src, os.path.join(self.sync_dir, "last.ckpt"))

    def on_validation_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            self._stage()

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            self._stage()
