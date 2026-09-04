from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import main


class MainTest(unittest.TestCase):
    def test_evaluation_routes(self):
        for backbone in ("clipreid", "transreid"):
            for dataset in ("mvti", "trc31k"):
                with self.subTest(backbone=backbone, dataset=dataset):
                    argv = ["main.py", "evaluate", "--backbone", backbone,
                            "--dataset", dataset, "--data-root", "../dataset",
                            "--weight", f"weights/cabreid_{backbone}_{dataset}.pth",
                            "--", "TEST.IMS_PER_BATCH", "16"]
                    with patch.object(sys, "argv", argv), patch.object(main.subprocess, "run") as run:
                        main.main()
                    command = run.call_args.args[0]
                    config = command[command.index("--config_file") + 1]
                    self.assertIn(f"/{dataset}/", config)
                    self.assertTrue((main.ROOT / backbone / config).is_file())
                    self.assertEqual(command[command.index("MODEL.PRETRAIN_CHOICE") + 1], "self")
                    self.assertEqual(command[command.index("DATASETS.ROOT_DIR") + 1],
                                     str(Path("../dataset").resolve()))
                    self.assertEqual(command[-2:], ["TEST.IMS_PER_BATCH", "16"])
                    self.assertEqual(run.call_args.kwargs["cwd"], main.ROOT / backbone)

    def test_evaluation_requires_checkpoint(self):
        with patch.object(sys, "argv", ["main.py", "evaluate", "--backbone", "clipreid"]):
            with self.assertRaises(SystemExit):
                main.parse_args()
