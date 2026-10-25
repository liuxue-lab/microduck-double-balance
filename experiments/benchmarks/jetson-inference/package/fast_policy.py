"""Same FP32 graph with fixed CPU buffers and non-aliasing LSTM state ping-pong."""
from deploy_contract import OnnxPolicy, require, vector


class BufferedPolicy:
    def __init__(self, path):
        original = OnnxPolicy(path)
        self.np, self.session = original.np, original.session
        np = self.np
        self.obs = np.zeros((1, 61), dtype=np.float32)
        self.hs = [np.zeros((1, 1, 256), dtype=np.float32) for _ in range(2)]
        self.cs = [np.zeros_like(self.hs[0]) for _ in range(2)]
        self.actions = np.zeros((1, 14), dtype=np.float32)
        self.bindings = []
        for i in range(2):
            binding = self.session.io_binding()
            binding.bind_cpu_input('obs', self.obs)
            binding.bind_cpu_input('h_in', self.hs[i])
            binding.bind_cpu_input('c_in', self.cs[i])
            for name, value in [('actions', self.actions), ('h_out', self.hs[1-i]), ('c_out', self.cs[1-i])]:
                binding.bind_output(name, device_type='cpu', device_id=0, element_type=np.float32,
                                    shape=value.shape, buffer_ptr=value.ctypes.data)
            self.bindings.append(binding)
        self.reset()

    @property
    def h(self):
        return self.hs[self.index]

    @property
    def c(self):
        return self.cs[self.index]

    def reset(self):
        for value in (*self.hs, *self.cs):
            value.fill(0)
        self.index = 0

    def step(self, obs, reset=False):
        vector(obs, 61)
        if reset:
            self.reset()
        self.obs[0, :] = obs
        self.session.run_with_iobinding(self.bindings[self.index])
        self.index = 1 - self.index
        require(all(self.np.isfinite(x).all() for x in (self.actions, self.h, self.c)), 'Nonfinite bound inference')
        return self.actions[0].tolist()
