import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

from level03_memory_efficient import tensor_memory_bytes


def test_int8_memory_less_than_fp32():
    n = 1_000_000
    int8_mem = tensor_memory_bytes(n, 8)
    fp32_mem = tensor_memory_bytes(n, 32)
    assert int8_mem < fp32_mem



def test_phase_c_stress_33_tensor_memory_exact_formula():
    assert tensor_memory_bytes(1024, 8) == 1024
    assert tensor_memory_bytes(1024, 16) == 2048

