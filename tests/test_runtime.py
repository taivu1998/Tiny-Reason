import unittest

import torch

from src.runtime import ensure_supported_4bit_runtime, resolve_device


class RuntimeTests(unittest.TestCase):
    def test_resolve_device_prefers_model_device(self):
        model = torch.nn.Linear(1, 1)
        device = resolve_device(model)

        self.assertEqual(device.type, "cpu")

    def test_ensure_supported_4bit_runtime_rejects_cpu(self):
        with self.assertRaises(RuntimeError):
            ensure_supported_4bit_runtime(torch.device("cpu"), load_in_4bit=True)

    def test_ensure_supported_4bit_runtime_allows_cpu_when_not_quantized(self):
        ensure_supported_4bit_runtime(torch.device("cpu"), load_in_4bit=False)


if __name__ == "__main__":
    unittest.main()
