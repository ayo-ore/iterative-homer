import torch
from abc import abstractmethod
from hydra.utils import call
from torch.utils.data import DataLoader, random_split

from src.experiments.base_experiment import BaseExperiment
from src.utils.datasets import HomerData
from src.utils.trainer import Trainer


class TrainingExperiment(BaseExperiment):

    def run(self):

        # initialize model
        if self.cfg.train or self.cfg.evaluate:

            if not hasattr(self, "model"):
                self.init_model()

            # preprocessing
            self.transforms = self.init_preprocessing()
            if self.cfg.data.on_gpu:
                self.transforms = [t.to(self.device) for t in self.transforms]
            self.log.info(f"Loaded preprocessing transforms:\n{self.transforms}")

        # initialize train/val dataloaders
        self.dataloaders = {}
        if self.cfg.train:

            self.log.info(f"Creating train/val DataLoaders")
            self.dataloaders["val"], self.dataloaders["train"] = self.init_dataloader(
                self.cfg.data.path_exp, self.cfg.data.path_sim, training=True
            )
            if self.model.bayesian:
                self.model.train_size += len(self.dataloaders["train"].dataset)

        # train model
        if self.cfg.train:
            self.log.info("Running training")

            trainer = Trainer(
                model=self.model,
                dataloaders=self.dataloaders,
                cfg=self.cfg.training,
                exp_dir=self.exp_dir,
                device=self.device,
                use_amp=self.cfg.use_amp,
            )
            trainer.run_training()

        del self.dataloaders
        torch.cuda.empty_cache()

        # evaluate model
        if self.cfg.evaluate:

            # load model state
            self.log.info(f"Loading model state from {self.exp_dir}.")
            self.model.load(self.exp_dir, self.device)
            self.model.eval()

            self.log.info("Running evaluation on test dataset")
            dataloader = self.init_dataloader(
                self.cfg.data.path_exp_test, self.cfg.data.path_sim_test
            )
            self.evaluate(dataloader)
            del dataloader
            torch.cuda.empty_cache()

            self.log.info("Running evaluation on train dataset")
            dataloader = self.init_dataloader(
                self.cfg.data.path_exp, self.cfg.data.path_sim
            )
            self.evaluate(dataloader, tag="train")
            del dataloader
            torch.cuda.empty_cache()

        # make plots
        if self.cfg.plot:
            self.log.info("Making plots")
            self.plot()

        # print memory usage
        self.log_resources()

    def init_model(self):
        self.log.info("Initializing model")

        self.model = call(self.cfg.model, self.cfg)
        self.model = self.model.to(self.device)
        model_name = (
            f"{self.model.__class__.__name__}[{self.model.net.__class__.__name__}]"
        )
        num_params = sum(w.numel() for w in self.model.trainable_parameters)
        self.log.info(f"Model ({model_name}) has {num_params} trainable parameters")

    def init_dataloader(self, path_exp, path_sim, training: bool = False):

        dcfg = self.cfg.data
        tcfg = self.cfg.training

        # read data
        dset = self.init_dataset(path_exp, path_sim)

        # optionally move dataset to gpu
        on_gpu = dcfg.on_gpu and self.cfg.use_gpu
        if on_gpu:
            dset = dset.to(self.device)

        self.log.info(f"Read dataset:\n{dset}")

        # preprocess
        for transform in self.transforms:
            dset = transform.forward(dset)

        # optionally create validation split
        if training:
            # create a validation split
            assert dcfg.val_frac > 0, "A validation split is required"

            # seed data split to avoid leakage across iterations
            fixed_rng = torch.Generator().manual_seed(1729)

            # split
            dsets = list(
                random_split(
                    dset, [dcfg.val_frac, 1 - dcfg.val_frac], generator=fixed_rng
                )
            )

            del dset
            torch.cuda.empty_cache()
        else:
            dsets = [dset]

        # create dataloaders
        dataloaders = []
        num_workers = (
            0 if on_gpu or not self.cfg.train else max(self.cfg.num_cpus - 1, 0)
        )
        use_mp = self.cfg.train and num_workers > 0
        for i, d in enumerate(dsets):

            is_trn = i > 0
            batch_size = tcfg.batch_size if is_trn else tcfg.test_batch_size

            dataloaders.append(
                DataLoader(
                    d,
                    shuffle=is_trn,
                    drop_last=is_trn,
                    batch_size=batch_size,
                    collate_fn=self.collate_fn,
                    num_workers=num_workers,
                    pin_memory=not on_gpu,
                    multiprocessing_context="spawn" if use_mp else None,
                    persistent_workers=use_mp,
                )
            )

        return dataloaders if training else dataloaders[0]

    def collate_fn(self, batch: HomerData):
        "Perform experiment-specific collation. Can help to avoid CPU-GPU sync during training."
        return batch

    @abstractmethod
    def init_dataset(self, path_exp, path_sim):
        "Read and return a dataset. To be implemented by the child class"
        pass

    @abstractmethod
    def evaluate(self):
        "Iterate dataset and save model predictions. To be implemented by the child class"
        pass

    @abstractmethod
    def plot(self):
        "Create and save evaluation plots. To be implemented by the child class"
        pass
