"""Stream API contract checks that can run without a CUDA driver.

These simulate stream handles/dependencies, not GPU execution. Native default
and nondefault stream PPO coverage lives in test_ppo.py.
"""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')
import warp as wp
from warptracer.rl_env import RacingEnv


@pytest.mark.parametrize('caller_handle', [0, 17])
def test_graph_capture_uses_nonnull_stream_and_preserves_setup_dependencies(monkeypatch, caller_handle):
    class Stream:
        def __init__(self, device, handle=99):
            self.cuda_stream = handle
            self.waited = []
        def wait_stream(self, other):
            self.waited.append(other)

    device = SimpleNamespace(is_cuda=True)
    caller = Stream(device, caller_handle)
    current = [caller]
    env = RacingEnv.__new__(RacingEnv)
    env.device = device
    env._capture_stream = None
    graph = object()
    operations = []

    @contextmanager
    def scoped_stream(stream, **kwargs):
        previous = current[0]
        current[0] = stream
        try:
            yield stream
        finally:
            current[0] = previous

    @contextmanager
    def capture(*, device, stream=None, **kwargs):
        active = stream if stream is not None else current[0]
        if active.cuda_stream == 0:
            raise RuntimeError('Warp error: unknown stream')
        assert active is current[0]
        assert caller in active.waited  # setup writes are ordered before capture
        yield SimpleNamespace(graph=graph)

    monkeypatch.setattr(wp, 'Stream', Stream)
    monkeypatch.setattr(wp, 'get_stream', lambda device: current[0])
    monkeypatch.setattr(wp, 'ScopedStream', scoped_stream)
    monkeypatch.setattr(wp, 'ScopedCapture', capture)
    env._operations = lambda: operations.append(current[0])
    env._capture_graph()
    assert env.graph is graph
    assert operations == [env._capture_stream]
    assert env._capture_stream is not caller
    assert current[0] is caller
    assert env._capture_stream in caller.waited  # caller sees capture setup completion


def test_nested_torch_scopes_reuse_registered_external_stream_wrapper(monkeypatch):
    env = RacingEnv.__new__(RacingEnv)
    env.device = SimpleNamespace(is_cuda=True)
    env.torch_device = 'cuda:0'
    caller = SimpleNamespace(cuda_stream=17)
    previous = SimpleNamespace(cuda_stream=99)
    current = [previous]
    conversions = []

    @contextmanager
    def scoped_stream(stream):
        saved = current[0]
        current[0] = stream
        try:
            yield stream
        finally:
            current[0] = saved

    def convert(stream):
        wrapper = SimpleNamespace(cuda_stream=stream.cuda_stream)
        conversions.append(wrapper)
        return wrapper

    monkeypatch.setattr(torch.cuda, 'current_stream', lambda device: caller)
    monkeypatch.setattr(wp, 'get_stream', lambda device: current[0])
    monkeypatch.setattr(wp, 'stream_from_torch', convert)
    monkeypatch.setattr(wp, 'ScopedStream', scoped_stream)
    with env._scope() as outer:
        with env._scope() as inner:
            assert inner is outer
            assert current[0] is outer
    assert len(conversions) == 1
    assert current[0] is previous
