import argparse
from omegaconf import OmegaConf
from trainer import PPOTrainer

from utils import set_seed

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="Run the training script",
                        action="store_true")
    parser.add_argument("--save", help="Run the training script",
                        action="store_true")
    parser.add_argument("--play", help="Run the inference script",
                        action='store_true')
    parser.add_argument("--resume", help="Resume from given savepoint",
                        action="store_true")
    parser.add_argument("--resume-episode",
                        type=str, default="latest")
    parser.add_argument("--device", help="Which device to use",
                        type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument("--seed", help="Random seed",
                        type=int, default=777)
    args = parser.parse_args()
    return args

def main(args):
    cfg = OmegaConf.load('config.yaml')
    cfg['device'] = args.device
    trainer = PPOTrainer(cfg)
    set_seed(args.seed)

    if args.train:
        if args.resume:
            trainer.load(cfg.data_path, args.resume_episode)
        trainer.train(cfg.train)
        
    elif args.play:
        assert args.resume
        trainer.load(cfg.data_path, args.resume_episode)
        trainer.play()

if __name__ == "__main__":
    main(get_args())