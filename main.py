import hydra
import os
import random
import numpy as np
import torch
import json
from datetime import datetime
from pathlib import Path
from pde_matching.models import ModelBuilder
from hydra.utils import instantiate
from lightning_example.data_module import OfflineTrajectoryDataModule
from lightning_example.experiment import Experiment
from lightning_example.experiment_single_step import ExperimentSingleStep
from lightning_example.plots_callback import PlotsCallback
from lightning_example.plots_single_step_callback import PlotsSingleStepCallback
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning import seed_everything
from tqdm import tqdm
import shutil


@hydra.main(config_path="configs/", config_name="config.yaml", version_base=None)
def main(config: DictConfig) -> None:
    # Set seed for reproducibility
    if config.get("seed") is not None:
        seed_everything(config.seed, workers=True)
        # Additional manual seeding for extra safety
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

    train_single_step = config.get("single_step", False)

    # Check for $TMPDIR
    tmpdir = os.environ.get("TMPDIR")
    if tmpdir and tmpdir != "/tmp" and config.get("checkpoint_to_test_path", None) is None:
        src_root = config.dataset.root
        dst_root = os.path.join(tmpdir, os.path.basename(src_root))
        if not os.path.exists(dst_root):
            shutil.copytree(src_root, dst_root)
        config.dataset.root = dst_root
        torch.set_float32_matmul_precision("high")
        print(f"Using dataset from {dst_root}.")
    
    # In OfflineTrajectoryDataModule, the env is tied to the dataset
    kwargs = OmegaConf.to_container(config.dataset)
    kwargs["batch_size"] = config.batch_size
    kwargs["num_workers"] = config.num_workers
    env_name = config.env.name
    del config.env.name
    data_module = OfflineTrajectoryDataModule(
        env=env_name, env_kwargs=config.env, **kwargs
    )
    example_data = data_module.get_example_data()

    # Create the model
    model = ModelBuilder(config=config.model).build(example_data)
    # model = torch.compile(model)

    logger = instantiate(config.logger)

    # Setup callbacks from config
    callbacks = []
    checkpoint_callback = None
    for callback_name, callback_config in config.callbacks.items():
        callback = instantiate(callback_config)
        callbacks.append(callback)
        if callback_name == "model_checkpoint":
            checkpoint_callback = callback

    if config.logger:
        if config.get("checkpoint_to_test_path") is not None:
            config.plot.animation.save_local_path = config.checkpoint_to_test_path
        
        if train_single_step:
            callbacks.append(PlotsSingleStepCallback(config.plot))
        else:
            callbacks.append(PlotsCallback(config.plot))

        if "Wandb" in config.logger["_target_"] and checkpoint_callback:
            from lightning_example.wandb_ckpt_callback import UploadCkpt
            callbacks.append(UploadCkpt(checkpoint_callback))

    trainer = instantiate(
        config.trainer,
        logger=logger,
        callbacks=callbacks,
    )

    full_config = OmegaConf.to_container(config, resolve=True)

    if config.get("checkpoint_to_test_path") is not None:
        exp_cls = ExperimentSingleStep if train_single_step else Experiment
        exp = exp_cls.load_from_checkpoint(config.checkpoint_to_test_path, config=full_config, model=model)
        test_results = trainer.test(exp, datamodule=data_module)
        
        # Save test results to JSON file
        if test_results:
            results_dict = {
                "checkpoint_path": config.checkpoint_to_test_path,
                "experiment_name": config.get("experiment_name", "unknown"),
                "config_name": config.get("_config_name", "unknown"),
                "dataset_root": config.dataset.root,
                "model_name": config.model.name,
                "seed": config.get("seed", "unknown"),
                "timestamp": datetime.now().isoformat(),
                "test_metrics": test_results[0]  # First test result dict
            }
            
            # Create results directory
            results_dir = Path(config.checkpoint_to_test_path).absolute().parent

            # Generate unique filename
            checkpoint_name = Path(config.checkpoint_to_test_path).name.replace('.ckpt', '')
            results_file = results_dir / f"{checkpoint_name}_test_results.json"
            
            with open(results_file, 'w') as f:
                json.dump(results_dict, f, indent=2)
            
            print(f"Test results saved to: {results_file}")
        
        return 
    
    if train_single_step:
        exp = ExperimentSingleStep(full_config, model)
    else:
        exp = Experiment(full_config, model)

    # Analyze dataset statistics
    if config.get("analyze_dataset_stats", False):
        data_module.setup("validate")
        val_data = data_module.val_data
        num_samples = len(val_data)
        
        total_nodes = 0
        total_edges = 0
        node_type_counts = {}
        edge_type_counts = {}
        node_type_min = {}
        node_type_max = {}
        edge_type_min = {}
        edge_type_max = {}
        
        for data_list in tqdm(val_data, desc="Analyzing dataset"):
            data = data_list[0]
            data_node_type = data.properties[:, :val_data.node_type_dim]
            data_edge_type = data.edge_type
            
            # Basic counts
            total_nodes += data_node_type.size(0)
            total_edges += data_edge_type.size(0)
            
            # Node type distribution
            unique_nodes, counts = torch.unique(data_node_type, return_counts=True)
            for node_type, count in zip(unique_nodes.tolist(), counts.tolist()):
                node_type_counts[node_type] = node_type_counts.get(node_type, 0) + count / num_samples
                node_type_min[node_type] = min(node_type_min.get(node_type, float('inf')), count)
                node_type_max[node_type] = max(node_type_max.get(node_type, 0), count)

            # Edge type distribution
            unique_edges, counts = torch.unique(data_edge_type, return_counts=True)
            for edge_type, count in zip(unique_edges.tolist(), counts.tolist()):
                edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + count / num_samples
                edge_type_min[edge_type] = min(edge_type_min.get(edge_type, float('inf')), count)
                edge_type_max[edge_type] = max(edge_type_max.get(edge_type, 0), count)
        
        print(f"Dataset Statistics (validation set):")
        print(f"Total samples: {num_samples}")
        print(f"Avg nodes per sample: {total_nodes / num_samples:.1f}")
        print(f"Avg edges per sample: {total_edges / num_samples:.1f}")
        print(f"Avg node type counts per sample: {dict(sorted(node_type_counts.items()))}")
        print(f"Min node type counts per sample: {dict(sorted(node_type_min.items()))}")
        print(f"Max node type counts per sample: {dict(sorted(node_type_max.items()))}")
        print(f"Avg edge type counts per sample: {dict(sorted(edge_type_counts.items()))}")
        print(f"Min edge type counts per sample: {dict(sorted(edge_type_min.items()))}")
        print(f"Max edge type counts per sample: {dict(sorted(edge_type_max.items()))}")
        
        return
    else:
        trainer.fit(exp, datamodule=data_module)


if __name__ == "__main__":
    main()
