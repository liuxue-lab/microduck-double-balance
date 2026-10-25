"""Run the pinned native zero-command sampling body with private uniform draws.

Preserves inherited forward=0.2, flags, clamp and body/world-reference order.
Native compute/_update_command subsequently zeros the standing command. The
extension's turn-in-place branch is disabled by its original p=0 setting.
"""
import inspect
from pathlib import Path
from runtime_accounting import require
from source_contracts import compile_method, method_node


class PrivateTorch:
    def __init__(self, bank, torch):
        self.bank, self.torch = bank, torch
    def empty(self, count, *, device):
        require(count == 1 and str(device) in ('cuda', 'cuda:0'), 'Command draw scope differs')
        bank = self.bank
        class Uniform:
            def uniform_(self, low, high):
                return bank.draw('command_value', 0, (1,))*(high-low)+low
        return Uniform()
    def __getattr__(self, name):
        return getattr(self.torch, name)


def private_command_method(bank, torch):
    from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand
    original = UniformVelocityCommand._resample_command
    source = Path(inspect.getsourcefile(original)).read_text()
    return bind_command(source, original.__globals__, bank, torch)


def bind_command(source, namespace, bank, torch):
    node = method_node(source, 'UniformVelocityCommand', '_resample_command')
    return compile_method(node, {**namespace, 'torch': PrivateTorch(bank, torch)},
                          'g11_private_native_command.py')
