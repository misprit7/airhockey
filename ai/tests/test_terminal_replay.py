import numpy as np
from airhockey.sequence_replay import SequenceReplay


def episode(number, length, terminal=True):
    obs = np.column_stack((np.full(length, number), np.arange(length))).astype(
        np.float32
    )
    return dict(
        obs=obs,
        action=obs[:, 1:],
        reward=np.r_[np.zeros(length - 1), 100.0],
        terminated=np.r_[np.zeros(length - 1), float(terminal)],
        demo=np.ones(length),
    )


def test_terminal_sampling_keeps_endings_without_crossing_overwrites():
    b = SequenceReplay(101, 2, 1, 4, 128, device="cpu", terminal_fraction=0.5)
    for i in range(100):
        b.add(episode(i, 17 + i % 11))
        obs, _, rew, term, _, _ = b.sample_with_demo()
        o = obs.numpy()
        assert (np.diff(o[:, :, 0], axis=0) == 0).all()
        assert (np.diff(o[:, :, 1], axis=0) == 1).all()
        assert (term[-1, :, 0] == 1).sum() >= 64
        assert (rew[-1, :, 0] == 100).sum() >= 64


def test_overwritten_terminal_anchors_do_not_survive_nonterminal_games():
    b = SequenceReplay(101, 2, 1, 4, 128, device="cpu", terminal_fraction=1.0)
    b.add(episode(0, 20))
    for i in range(1, 6):
        b.add(episode(i, 80, terminal=False))
    obs, _, _, term, _, _ = b.sample_with_demo()
    assert not term.any()
    assert (obs[:, :, 0] >= 4).all()
    assert b.terminal_head == b.terminal_tail
