"""Training recovery and policy replay at episode boundaries."""
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from warptracer.ppo import PPO, PPOConfig
from warptracer.rl_env import RacingEnv, EpisodeConfig
from warptracer.train import main, evaluate, save_checkpoint

torch.set_num_threads(2)


def test_evaluation_records_initial_and_terminal_state_for_one_step_episode(tmp_path):
    env = RacingEnv(EpisodeConfig(num_envs=1, max_steps=1), device="cpu")
    trainer = PPO(env, PPOConfig(rollout_steps=8, hidden=16))
    checkpoint = tmp_path / "checkpoint.pt"
    save_checkpoint(trainer, checkpoint)
    output = tmp_path / "evaluation"
    evaluate([str(checkpoint), "--device", "cpu", "--output", str(output)])
    metrics = json.loads(output.with_suffix(".json").read_text())
    assert metrics["termination_reason"] == 3 and metrics["transitions"] == 1
    assert output.with_suffix(".npz").exists()
    with np.load(output.with_suffix(".npz")) as recording:
        np.testing.assert_allclose(recording["times"], [0, 1/60])
        assert recording["poses"].shape == (2, 7)
        assert np.linalg.norm(recording["poses"][1, :3] - recording["poses"][0, :3]) < .02


def test_interrupt_during_first_update_preserves_initial_checkpoint(tmp_path, monkeypatch):
    def interrupt(self):
        raise KeyboardInterrupt
    monkeypatch.setattr(PPO, "iterate", interrupt)
    main(["smoke", "--output", str(tmp_path)])
    state = torch.load(tmp_path / "checkpoint.pt", weights_only=True)
    assert state["iteration"] == 0 and state["global_step"] == 0


def test_resume_removes_log_suffix_beyond_saved_update(tmp_path):
    main(["smoke", "--updates", "1", "--output", str(tmp_path)])
    log = tmp_path / "training.jsonl"
    with log.open("a") as stream:
        stream.write(json.dumps({"iteration": 2, "transitions": 512}) + "\n")
        stream.write('{"iteration":')  # an interrupted log write is also discarded
    main(["smoke", "--updates", "1", "--resume", str(tmp_path / "checkpoint.pt"), "--output", str(tmp_path)])
    assert [json.loads(line)["iteration"] for line in log.read_text().splitlines()] == [1, 2]
