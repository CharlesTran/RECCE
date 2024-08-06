import yaml
import argparse
import os
import torch.multiprocessing as mp
import torch.distributed as dist
from trainer import ExpMultiGpuTrainer

def arg_parser():
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument("--config",
                        type=str,
                        default="config/Recce.yml",
                        help="Specified the path of configuration file to be used.")
    return parser.parse_args()

def setup(rank, world_size, config):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend=config["config"]["distribute"]["backend"], rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, config):
    setup(rank, world_size, config)
    config["config"]["local_rank"] = rank
    trainer = ExpMultiGpuTrainer(config, stage="Train")
    trainer.train()
    cleanup()

if __name__ == '__main__':
    import torch

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    args = arg_parser()
    config = args.config

    with open(config) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    world_size = 4  # 总共使用四张GPU
    mp.spawn(train,
             args=(world_size, config),
             nprocs=world_size,
             join=True)
