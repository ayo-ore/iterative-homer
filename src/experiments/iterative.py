import os

from omegaconf import OmegaConf

from src.experiments import StepOneExperiment, StepTwoExperiment
from src.experiments.base_experiment import BaseExperiment


class IterativeExperiment(BaseExperiment):

    def run(self):

        # dispatch settings to steps 1 and 2
        self.resolve_configs()

        # iterate
        for it in range(1, self.cfg.iterations + 1):

            self.log.info(f"Starting iteration {it}")

            # point to previous iteration if it exists
            self.w_prev_path = step_two_dir if it > 1 else self.cfg.w_prev_path

            # run step one
            if (self.cfg.w_class_path is None) or it > 1:
                step_one_dir = self.run_step_one(it)
            else:
                # skip step one if path provided
                # TODO: Assert consistency of w_class_path and w_prev_path
                self.log.info(f"Skipping step one")
                step_one_dir = self.cfg.w_class_path

            # run step two
            step_two_dir = self.run_step_two(it, step_one_dir)

    def run_step_one(self, it):

        cfg = self.cfg.step_one

        # create experiment directory
        step_one_dir = os.path.join(self.exp_dir, f"step_one_{it}")
        os.makedirs(step_one_dir + "/.hydra")

        # link previous iteration
        cfg.w_prev_path = self.w_prev_path

        # optionally decay learning rate
        if (decay := self.cfg.step_one_lr_decay) is not None:
            cfg.training.lr *= decay ** (it - 1)

        # write config
        with open(step_one_dir + "/.hydra/config.yaml", "w") as f:
            f.write(OmegaConf.to_yaml(cfg))

        # initialize and run
        step_one = StepOneExperiment(cfg, step_one_dir)
        step_one.run()

        return step_one_dir

    def run_step_two(self, it, step_one_dir):

        cfg = self.cfg.step_two

        # create experiment directory
        step_two_dir = os.path.join(self.exp_dir, f"step_two_{it}")
        os.makedirs(step_two_dir + "/.hydra")

        # link step one and previous iteration
        cfg.w_class_path = step_one_dir
        cfg.w_prev_path = self.w_prev_path

        # optionally decay learning rate
        if (decay := self.cfg.step_two_lr_decay) is not None:
            cfg.training.lr *= decay ** (it - 1)

        # write config
        with open(step_two_dir + "/.hydra/config.yaml", "w") as f:
            f.write(OmegaConf.to_yaml(cfg))

        # initialize
        step_two = StepTwoExperiment(cfg, step_two_dir)

        # optionally warm start
        if self.cfg.warm_start_step_two and self.w_prev_path:
            assert cfg.iterate_target
            self.log.info(f"Warm starting step_two nets from iteration {it-1}")
            step_two.init_model()
            step_two.model.load_nets(self.w_prev_path, self.device)

        # run
        step_two.run()

        return step_two_dir

    def resolve_configs(self):

        # dataset
        self.cfg.step_one.dataset = self.cfg.dataset
        self.cfg.step_two.dataset = self.cfg.dataset

        # gpu
        self.cfg.step_one.use_gpu = self.cfg.use_gpu
        self.cfg.step_two.use_gpu = self.cfg.use_gpu

        # f32 precision
        self.cfg.step_one.use_tf32 = self.cfg.use_tf32
        self.cfg.step_two.use_tf32 = self.cfg.use_tf32

        # mixed precision
        self.cfg.step_one.use_amp = self.cfg.use_amp
        self.cfg.step_two.use_amp = self.cfg.use_amp
