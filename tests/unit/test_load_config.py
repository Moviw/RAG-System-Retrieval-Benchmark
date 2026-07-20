from pathlib import Path

import yaml


def test_load_config_declares_required_workloads() -> None:
    config = yaml.safe_load(Path("benchmarks/configs/load.yaml").read_text())

    assert config["concurrency"] == [1, 5, 10, 25, 50, 100]
    assert config["workloads"]["read-only"]["search"] == 100
    assert config["workloads"]["mostly-read"]["search"] == 95
    assert config["workloads"]["mixed"]["delete"] == 5
