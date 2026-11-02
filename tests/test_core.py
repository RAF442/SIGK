"""Testy protokołu danych, degradacji, metryk i uczenia (unittest)."""

import importlib.util
import math
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from src.common import dump_json, image_seed, load_json
from src.data import extract_archive, official_split
from src.metrics import errors

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_CV2 = importlib.util.find_spec("cv2") is not None
HAS_SKIMAGE = importlib.util.find_spec("skimage") is not None


class ProtocolTests(unittest.TestCase):
    def test_disjoint_complete_split(self):
        split = official_split()
        self.assertEqual([len(split[k]) for k in ("train", "val", "test")], [720, 80, 100])
        all_names = sum(split.values(), [])
        self.assertEqual(len(set(all_names)), 900)
        self.assertEqual(split["test"][0], "0801.png")

    def test_zip_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / "bad.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("../escaped.txt", "bad")
            with self.assertRaises(ValueError):
                extract_archive(archive, Path(temp) / "target")
            self.assertFalse((Path(temp) / "escaped.txt").exists())

    def test_seed_stable_and_condition_specific(self):
        self.assertEqual(image_seed(42, "0801.png", "dn001"), image_seed(42, "0801.png", "dn001"))
        self.assertNotEqual(image_seed(42, "0801.png", "dn001"), image_seed(42, "0801.png", "dn003"))

    def test_atomic_standard_json(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "a.json"
            dump_json(path, {"PSNR": float("inf"), "value": None})
            self.assertEqual(load_json(path), {"PSNR": "inf", "value": None})
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_metrics_known_constant_error(self):
        reference = np.ones((8, 8, 3), np.float32) * 0.5
        prediction = reference + 0.1
        result = errors(reference, prediction)
        self.assertAlmostEqual(result["PSNR"], 20, places=4)
        self.assertAlmostEqual(result["MSE"], 0.01, places=6)
        self.assertAlmostEqual(result["SSE"], 1.92, places=5)
        self.assertAlmostEqual(result["SNE"], 1.92, places=5)
        self.assertAlmostEqual(result["SNE"], reference.size * result["MSE"])
        self.assertTrue(math.isinf(errors(reference, reference)["PSNR"]))
        self.assertEqual(errors(reference, reference)["SNE"], 0.0)
        self.assertNotIn("SNE", errors(reference, prediction, None))

    def test_sne_black_reference_has_no_division_by_zero(self):
        reference = np.zeros((2, 2, 3), np.float32)
        prediction = np.full_like(reference, 0.25)
        result = errors(reference, prediction)
        self.assertEqual(result["SNE"], 0.75)
        self.assertTrue(math.isfinite(result["SNE"]))
        self.assertEqual(errors(reference, reference)["SNE"], 0.0)

    def test_sne_rejects_definitions_inconsistent_with_lecture(self):
        reference = np.zeros((2, 2, 3), np.float32)
        for definition in ("mse", "relative-squared-error"):
            with self.assertRaises(ValueError):
                errors(reference, reference, definition)

    def test_cli_defaults_to_lecture_sne(self):
        from src.cli import parser
        self.assertEqual(parser().parse_args(["evaluate"]).sne_definition, "sse")

    def test_metric_rejects_nonfinite(self):
        image = np.zeros((8, 8, 3), np.float32)
        image[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            errors(image, image)


@unittest.skipUnless(HAS_CV2, "OpenCV nie jest zainstalowane")
class DegradationTests(unittest.TestCase):
    def test_shapes_and_noise_statistics(self):
        from src.data import fixed_degradation
        clean = np.full((256, 256, 3), 0.5, np.float32)
        for condition, size in (("sr4", 64), ("sr8", 32)):
            low, base = fixed_degradation(clean, condition, 42, "a.png")
            self.assertEqual(low.shape, (size, size, 3))
            np.testing.assert_allclose(base, clean, atol=1e-6)
        for condition, sigma in (("dn001", 0.01), ("dn003", 0.03)):
            noisy, _ = fixed_degradation(clean, condition, 42, "a.png")
            again, _ = fixed_degradation(clean, condition, 42, "a.png")
            np.testing.assert_array_equal(noisy, again)
            self.assertAlmostEqual(float((noisy - clean).std()), sigma, delta=sigma * 0.01)

    def test_manifest_detects_tampering(self):
        from src.data import verify_data
        from src.demo import make_demo
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "demo"
            make_demo(root)
            manifest = verify_data(root)
            self.assertEqual(manifest["dataset"], "SYNTHETIC-SMOKE")
            image = next((root / "test").glob("*.png"))
            image.write_bytes(b"modified")
            with self.assertRaises(ValueError):
                verify_data(root)


@unittest.skipUnless(HAS_TORCH, "PyTorch nie jest zainstalowany")
class LearningTests(unittest.TestCase):
    def test_shapes_gradient_and_learning(self):
        import torch
        from src.models import Restorer
        torch.set_num_threads(2)
        for condition, size in (("sr4", 64), ("sr8", 32), ("dn001", 256), ("dn003", 256)):
            torch.manual_seed(7)
            model = Restorer(condition, width=4, blocks=1)
            image = torch.zeros(1, 3, size, size)
            base = torch.full((1, 3, 256, 256), 0.5)
            target = base + 0.05
            np.testing.assert_allclose(model(image, base).detach().numpy(), base.numpy())
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            before = float((model(image, base) - target).square().mean().detach())
            for _ in range(4):
                optimizer.zero_grad()
                loss = (model(image, base) - target).square().mean()
                loss.backward()
                self.assertIsNotNone(model.tail.weight.grad)
                optimizer.step()
            after = float((model(image, base) - target).square().mean().detach())
            self.assertLess(after, before)

    def test_checkpoint_roundtrip(self):
        import torch
        from src.models import Restorer, load_model
        model = Restorer("sr4", width=4, blocks=1)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.pt"
            torch.save({"format_version": 1, "condition": "sr4", "architecture": {"width": 4, "blocks": 1},
                        "model": model.state_dict()}, path)
            restored, _ = load_model(path, "cpu")
            for left, right in zip(model.parameters(), restored.parameters()):
                self.assertTrue(torch.equal(left, right))

    def test_finished_run_resume_restores_history(self):
        import torch
        from argparse import Namespace
        from src.common import sha256
        from src.demo import make_demo
        from src.engine import train
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "data"
            make_demo(data)
            args = Namespace(device="cpu", threads=2, seed=42, data=str(data),
                             runs=str(root / "runs"), condition="dn001", resume=False,
                             width=4, blocks=1, lr=0.001, epochs=1, batch_size=2,
                             repeat=1, workers=0, loss="mse")
            train(args)
            folder = root / "runs" / "dn001"
            checkpoint_hash = sha256(folder / "last.pt")
            (folder / "history.csv").unlink()
            args.resume = True
            train(args)
            self.assertTrue((folder / "history.csv").exists())
            self.assertEqual(sha256(folder / "last.pt"), checkpoint_hash)
            checkpoint = torch.load(folder / "best.pt", weights_only=True)
            self.assertEqual(checkpoint["epoch"], 1)
            args.lr = 0.01
            with self.assertRaises(ValueError):
                train(args)


@unittest.skipUnless(HAS_SKIMAGE, "scikit-image nie jest zainstalowane")
class StructuralTests(unittest.TestCase):
    def test_identity_ssim(self):
        from src.metrics import Metrics
        image = np.random.default_rng(1).random((32, 32, 3), dtype=np.float32)
        result = Metrics(skip_lpips=True)(image, image)
        self.assertAlmostEqual(result["SSIM"], 1.0)
        self.assertIsNone(result["LPIPS"])


if __name__ == "__main__":
    unittest.main()
