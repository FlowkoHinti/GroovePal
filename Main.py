import argparse
import torch
from omegaconf import OmegaConf

from Configs import BASE_PATH, MAX_SEQUENCE_LENGTH, RNG_SEED
from GrooveModel.Learner.MultiTaskDnaLearner import MultiTaskDNALearner


def compute_embedding_dim(embedding_cfg):
    return sum([
        embedding_cfg.instruments.embedding_dim,
        embedding_cfg.velocities.embedding_dim,
        embedding_cfg.offsets.embedding_dim,
        embedding_cfg.time_signature.embedding_dim,
        embedding_cfg.grid_factor.embedding_dim,
        embedding_cfg.bpm.embedding_dim,
        embedding_cfg.beat_units.embedding_dim,
    ])


def inject_paths(cfg):
    """Resolve and inject base path into dataset and train directories."""
    if isinstance(cfg.dataset.dna_path, str):
        cfg.dataset.dna_path = (BASE_PATH / cfg.dataset.dna_path).resolve()
    if isinstance(cfg.train.save_dir, str):
        cfg.train.save_dir = (BASE_PATH / cfg.train.save_dir).resolve()
    return cfg


def prepare_config(cfg):
    """Inject dynamic values like embedding_dim and context_length."""
    cfg.model.context_length = MAX_SEQUENCE_LENGTH
    cfg.model.embedding_dim = compute_embedding_dim(cfg.embedding)
    cfg = inject_paths(cfg)
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", help="Name of the YAML config file in Experiments/ (without extension)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true", help="Run in training mode")
    group.add_argument("--test", action="store_true", help="Run in testing mode")

    args = parser.parse_args()

    config_path = BASE_PATH / "Experiments" / f"{args.experiment}.yml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    torch.manual_seed(RNG_SEED)

    # Load and patch config
    cfg = OmegaConf.load(config_path)
    cfg = prepare_config(cfg)

    # Instantiate your learner
    learner = MultiTaskDNALearner(cfg)

    #TODO: SAVE A COPY OF PRETRAINED/INTERMEDIATE/MODELS -> TO COMPARE

    # Start training
    if args.train:
        learner.train()
    if args.test:
        learner.test()


if __name__ == "__main__":
    main()
