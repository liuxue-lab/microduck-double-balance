"""Same transition cases on real host Torch/TD or labelled development doubles."""
from types import SimpleNamespace
from integration.durable import require
from integration.transitions import TerminalBridge, process_transition
from integration.process_vector import ProcessVector, tensor_wire


def fake_env(torch, cache_kind, td, terminated=False, truncated=False):
    class Env:
        num_envs, device, max_episode_length = 1, "cpu", 500
        cfg = SimpleNamespace(auto_reset=False)
        def __init__(self):
            self.calls = {"step": 0, "reset": 0, "compute": 0}
            values = {"actor": torch.zeros(1, 61), "critic": torch.zeros(1, 85)}
            self.values = td(values, batch_size=[1]) if cache_kind == "TensorDict" else values
            self.reward = torch.zeros(1)
            self.terminated = torch.tensor([False], dtype=torch.bool)
            self.truncated = torch.tensor([False], dtype=torch.bool)
        def reset(self, env_ids=None):
            require(env_ids == [0], "Selective IDs differ")
            self.calls["reset"] += 1
            for v in self.values.values():
                v.fill_(-10)
            self.reward.fill_(0)
            self.terminated.fill_(False)
            self.truncated.fill_(False)
            return self.values, {"reset_count": self.calls["reset"]}
        def step(self, action):
            self.calls["step"] += 1
            for v in self.values.values():
                v.fill_(7)
            self.reward.fill_(3)
            self.terminated.fill_(terminated)
            self.truncated.fill_(truncated)
            return self.values, self.reward, self.terminated, self.truncated, {"reward_alias": self.reward}
    return Env()


def terminal_case(torch, td, cache_kind, terminated, truncated):
    env = fake_env(torch, cache_kind, td, terminated, truncated)
    bridge = TerminalBridge(env, torch, td, 4)
    before = bridge.get_observations()
    obs, reward, done, extras = bridge.step(torch.zeros(1, 14))
    row = extras["stage12_transition"]
    require(reward.tolist() == [3] and extras["reward_alias"].tolist() == [3], "Reward aliased reset")
    require(done.tolist() == [int(terminated or truncated)], "Done OR differs")
    require(row["terminated"].tolist() == [terminated] and row["truncated"].tolist() == [truncated],
            "Reset mutated terminal flags")
    require(row["bootstrap_mask"].tolist() == [truncated and not terminated], "Wrong bootstrap precedence")
    require(bool((row["terminal_observation"]["critic"] == 7).all()), "Terminal observation lost")
    expected = -10 if terminated or truncated else 7
    require(bool((obs["actor"] == expected).all()), "Next observation differs")
    require(bool((before["actor"] == -10).all()), "Old observation mutated")
    require(env.calls == {"step": 1, "reset": 1 + int(terminated or truncated), "compute": 0},
            "Duplicate reset/observation computation")
    critic_calls = []
    class Critic:
        is_recurrent = False
        def __call__(self, values):
            critic_calls.append(values["critic"].clone())
            return values["critic"][:, :1]
    seen = []
    alg = SimpleNamespace(critic=Critic(), gamma=.9,
                          process_env_step=lambda *args: seen.append(args))
    corrected = process_transition(alg, obs, reward, done, extras)
    target = 9.3 if truncated and not terminated else 3.
    require(abs(float(corrected.tolist()[0]) - target) < 1e-5, "Terminal bootstrap value differs")
    require(len(critic_calls) == int(truncated and not terminated) and len(seen) == 1,
            "Critic/storage call count differs")
    require("time_outs" not in seen[0][3] and "stage12_transition" not in seen[0][3],
            "Stock double-bootstrap path remains")
    try:
        process_transition(alg, obs, reward, done, extras)
    except RuntimeError:
        pass
    else:
        raise RuntimeError("Duplicate transition accepted")
    bridge.step(torch.zeros(1, 14))
    require(bool((row["terminal_observation"]["critic"] == 7).all()) and reward.tolist() == [3],
            "Later buffer reuse mutated terminal transition")
    return dict(cache=cache_kind, terminated=terminated, truncated=truncated, passed=True,
                real_environment=False, real_PPO=False)


class Loopback:
    """In-memory protocol fixture, not a process, CUDA test, or native world."""
    def __init__(self, owner, bridge, torch):
        self.owner, self.bridge, self.torch = owner, bridge, torch
        self.reply = None
    def send(self, message):
        from integration.process_vector import wire_tensor
        op, payload = message["operation"], message["payload"]
        if op == "step":
            value = self.bridge.step(wire_tensor(payload, self.torch, "cpu"))
        elif op == "observe":
            value = self.bridge.get_observations()
        elif op == "reset":
            value = self.bridge.reset() if payload else (self.bridge.get_observations(), {})
        else:
            value = {}
        self.reply = dict(sequence=message["sequence"], owner=self.owner, ok=True, payload=tensor_wire(value))
    def poll(self, timeout):
        return self.reply is not None
    def recv(self):
        r, self.reply = self.reply, None
        return r


def vector_case(torch, td):
    envs = [fake_env(torch, "TensorDict", td, False, i == 1) for i in (0, 1)]
    bridges = [TerminalBridge(e, torch, td, 5) for e in envs]
    v = ProcessVector([Loopback(i, b, torch) for i, b in enumerate(bridges)], torch, td)
    before = v.get_observations()
    obs, reward, done, extras = v.step(torch.zeros(2, 14))
    require(done.tolist() == [0, 1] and reward.tolist() == [3, 3], "Vector concatenation differs")
    require(extras["stage12_transition"]["bootstrap_mask"].tolist() == [False, True], "Cross-owner timeout mixed")
    require(bool((obs["actor"][0] == 7).all()) and bool((obs["actor"][1] == -10).all()), "Owner ordering differs")
    require(bool((before["actor"] == -10).all()), "Vector snapshot aliases")
    old = envs[1].calls["reset"]
    v.reset([0])
    require(envs[0].calls["reset"] == 2 and envs[1].calls["reset"] == old, "Selective reset touched sibling")
    counts = [dict(e.calls) for e in envs]
    v.reset([])
    require([e.calls for e in envs] == counts, "Empty reset called owner")
    try:
        v.reset([0, 0])
    except RuntimeError:
        pass
    else:
        raise RuntimeError("Duplicate owner reset accepted")
    return dict(passed=True, owners=2, native_nworld_tested=False, transport="in-memory-protocol-only")


def all_cases(torch, td):
    """Collect every independent failure in one invocation, never fail-fast."""
    cases = [("transition/%s/%d/%d" % (kind, t, x),
              lambda k=kind, a=t, b=x: terminal_case(torch, td, k, a, b))
             for kind in ("dict", "TensorDict") for t, x in ((False, False), (True, False),
                                                            (False, True), (True, True))]
    cases.append(("two-owner-protocol", lambda: vector_case(torch, td)))
    rows = []
    for name, operation in cases:
        try:
            rows.append(dict(name=name, passed=True, result=operation()))
        except Exception as exc:
            import traceback
            rows.append(dict(name=name, passed=False, error=repr(exc), traceback=traceback.format_exc()))
    return rows
