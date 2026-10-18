import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catstudio.training import cache_features, train

p = argparse.ArgumentParser()
p.add_argument("phase", choices=["cache", "train"])
p.add_argument("--run-name", default="full")
p.add_argument("--max-steps", type=int, default=3000)
p.add_argument("--accumulation", type=int, default=4)
p.add_argument("--rank", type=int, default=8)
p.add_argument("--learning-rate", type=float, default=1e-4)
p.add_argument("--checkpoint-every", type=int, default=500)
p.add_argument("--resume", help="Trusted local checkpoint directory containing trainer_state.pt")
a = p.parse_args()
if a.phase == "cache":
    cache_features()
else:
    train(
        a.run_name,
        a.max_steps,
        a.accumulation,
        a.rank,
        a.learning_rate,
        a.checkpoint_every,
        a.resume,
    )
