from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

_MPL_CACHE_DIR = Path(tempfile.gettempdir()) / "overall_validation_mplconfig"
_XDG_CACHE_DIR = Path(tempfile.gettempdir()) / "overall_validation_xdg_cache"
_MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(_XDG_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import scipy.stats
import torch
import torch.nn as nn


NLCD_CLASSES = (
    11,
    12,
    21,
    22,
    23,
    24,
    31,
    41,
    42,
    43,
    52,
    71,
    81,
    82,
    90,
    95,
)
MAX_HEIGHT_M = 75.0
EPS = 1e-8
FAMILY_NAME_MAP = {
    "1": "UNETBaseline",
    "2A": "cGANRandomVecFixed",
    "3": "cGANMultiRandomDiversity",
}
TARGET_LABEL_MAP = {"BF": "bf", "BH": "bh"}
DEFAULT_SAMPLE_MODES = {"BF": "standard", "BH": "pipeline_bf"}
POINT_STAGE = "point"
DIVERSITY_STAGE = "diversity"
STANDARD_MODE = "standard"
ORACLE_MODE = "oracle_bf"
PIPELINE_MODE = "pipeline_bf"


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    root: Path
    bf_condition_dir: str
    bf_target_dir: str
    bh_condition_dir: str
    bh_target_dir: str
    manifest_path: Path | None


@dataclass(frozen=True)
class GroupSpec:
    group_id: str
    training_regime: str
    family: str
    dataset_name: str
    learning_rate: float
    checkpoint: int
    bf_run_dir: Path
    bh_run_dir: Path
    bf_checkpoint_path: Path
    bh_checkpoint_path: Path
    bf_metadata: dict[str, Any]
    bh_metadata: dict[str, Any]


class FolderCache:
    def __init__(self, folder: Path, preload: bool) -> None:
        self.folder = folder
        self.filenames = sorted(p.name for p in folder.glob("*.tif"))
        if not self.filenames:
            raise ValueError(f"No TIFF files found in {folder}")
        self._arrays = (
            {name: self._read(folder / name) for name in self.filenames}
            if preload else None
        )

    @staticmethod
    def _read(path: Path) -> np.ndarray:
        with rasterio.open(path) as src:
            return src.read(1)

    def get(self, filename: str) -> np.ndarray:
        if self._arrays is not None:
            return self._arrays[filename]
        return self._read(self.folder / filename)


class DatasetBundle:
    def __init__(self, spec: DatasetConfig, cache_mode: str = "all") -> None:
        preload = cache_mode == "all"
        self.spec = spec
        self.central_cache = FolderCache(spec.root / spec.bf_condition_dir, preload)
        self.bf_cache = FolderCache(spec.root / spec.bf_target_dir, preload)
        self.bh_cache = FolderCache(spec.root / spec.bh_target_dir, preload)
        common = (
            set(self.central_cache.filenames)
            & set(self.bf_cache.filenames)
            & set(self.bh_cache.filenames)
        )
        self.filenames = sorted(common)
        if not self.filenames:
            raise ValueError(f"No matched filenames found for {spec.name}")

    def limited_filenames(self, quick_tiles: int | None = None) -> list[str]:
        if quick_tiles is None or quick_tiles <= 0:
            return list(self.filenames)
        return list(self.filenames[:quick_tiles])

    def get_tile(self, filename: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            self.central_cache.get(filename),
            self.bf_cache.get(filename),
            self.bh_cache.get(filename),
        )


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=not norm,
            ),
        ]
        if norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=False))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: bool = False) -> None:
        super().__init__()
        self.relu = nn.ReLU(inplace=False)
        self.deconv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(0.5) if dropout else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(x)
        x = self.deconv(x)
        x = self.bn(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class DecoderBlockFiLM(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_dim: int,
        dropout: bool = False,
    ) -> None:
        super().__init__()
        self.relu = nn.ReLU(inplace=False)
        self.deconv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(0.5) if dropout else None
        self.film = nn.Linear(latent_dim, out_channels * 2)
        nn.init.normal_(self.film.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.film.bias)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = self.relu(x)
        x = self.deconv(x)
        x = self.bn(x)
        gamma, beta = self.film(z).chunk(2, dim=1)
        x = x * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class GeneratorUNet(nn.Module):
    def __init__(self, condition_channels: int) -> None:
        super().__init__()
        self.encoder1 = EncoderBlock(condition_channels, 64, norm=False)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)
        self.encoder5 = EncoderBlock(512, 512)
        self.encoder6 = EncoderBlock(512, 512)
        self.encoder7 = EncoderBlock(512, 512)
        self.encoder8 = EncoderBlock(512, 512, norm=False)
        self.decoder8 = DecoderBlock(512, 512, dropout=True)
        self.decoder7 = DecoderBlock(1024, 512, dropout=True)
        self.decoder6 = DecoderBlock(1024, 512, dropout=True)
        self.decoder5 = DecoderBlock(1024, 512)
        self.decoder4 = DecoderBlock(1024, 256)
        self.decoder3 = DecoderBlock(512, 128)
        self.decoder2 = DecoderBlock(256, 64)
        self.decoder1 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        e1 = self.encoder1(cond)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        e6 = self.encoder6(e5)
        e7 = self.encoder7(e6)
        e8 = self.encoder8(e7)
        d8 = torch.cat([self.decoder8(e8), e7], dim=1)
        d7 = torch.cat([self.decoder7(d8), e6], dim=1)
        d6 = torch.cat([self.decoder6(d7), e5], dim=1)
        d5 = torch.cat([self.decoder5(d6), e4], dim=1)
        d4 = torch.cat([self.decoder4(d5), e3], dim=1)
        d3 = torch.cat([self.decoder3(d4), e2], dim=1)
        d2 = torch.cat([self.decoder2(d3), e1], dim=1)
        return self.decoder1(d2)


class GeneratorRandomVec(nn.Module):
    def __init__(self, condition_channels: int, latent_dim: int = 8) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder1 = EncoderBlock(condition_channels + latent_dim, 64, norm=False)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)
        self.encoder5 = EncoderBlock(512, 512)
        self.encoder6 = EncoderBlock(512, 512)
        self.encoder7 = EncoderBlock(512, 512)
        self.encoder8 = EncoderBlock(512, 512, norm=False)
        self.decoder8 = DecoderBlock(512, 512, dropout=True)
        self.decoder7 = DecoderBlock(1024, 512, dropout=True)
        self.decoder6 = DecoderBlock(1024, 512, dropout=True)
        self.decoder5 = DecoderBlock(1024, 512)
        self.decoder4 = DecoderBlock(1024, 256)
        self.decoder3 = DecoderBlock(512, 128)
        self.decoder2 = DecoderBlock(256, 64)
        self.decoder1 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, cond: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        z_img = z[:, :, None, None].expand(-1, -1, cond.size(2), cond.size(3))
        x = torch.cat([cond, z_img], dim=1)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        e6 = self.encoder6(e5)
        e7 = self.encoder7(e6)
        e8 = self.encoder8(e7)
        d8 = torch.cat([self.decoder8(e8), e7], dim=1)
        d7 = torch.cat([self.decoder7(d8), e6], dim=1)
        d6 = torch.cat([self.decoder6(d7), e5], dim=1)
        d5 = torch.cat([self.decoder5(d6), e4], dim=1)
        d4 = torch.cat([self.decoder4(d5), e3], dim=1)
        d3 = torch.cat([self.decoder3(d4), e2], dim=1)
        d2 = torch.cat([self.decoder2(d3), e1], dim=1)
        return self.decoder1(d2)


class GeneratorMultiRandom(nn.Module):
    def __init__(self, condition_channels: int, latent_dim: int = 16) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder1 = EncoderBlock(condition_channels + latent_dim, 64, norm=False)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)
        self.encoder5 = EncoderBlock(512, 512)
        self.encoder6 = EncoderBlock(512, 512)
        self.encoder7 = EncoderBlock(512, 512)
        self.encoder8 = EncoderBlock(512, 512, norm=False)
        self.decoder8 = DecoderBlockFiLM(512, 512, latent_dim, dropout=True)
        self.decoder7 = DecoderBlockFiLM(1024, 512, latent_dim, dropout=True)
        self.decoder6 = DecoderBlockFiLM(1024, 512, latent_dim, dropout=True)
        self.decoder5 = DecoderBlockFiLM(1024, 512, latent_dim)
        self.decoder4 = DecoderBlockFiLM(1024, 256, latent_dim)
        self.decoder3 = DecoderBlockFiLM(512, 128, latent_dim)
        self.decoder2 = DecoderBlockFiLM(256, 64, latent_dim)
        self.decoder1 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, cond: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        z_img = z[:, :, None, None].expand(-1, -1, cond.size(2), cond.size(3))
        x = torch.cat([cond, z_img], dim=1)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        e6 = self.encoder6(e5)
        e7 = self.encoder7(e6)
        e8 = self.encoder8(e7)
        d8 = torch.cat([self.decoder8(e8, z), e7], dim=1)
        d7 = torch.cat([self.decoder7(d8, z), e6], dim=1)
        d6 = torch.cat([self.decoder6(d7, z), e5], dim=1)
        d5 = torch.cat([self.decoder5(d6, z), e4], dim=1)
        d4 = torch.cat([self.decoder4(d5, z), e3], dim=1)
        d3 = torch.cat([self.decoder3(d4, z), e2], dim=1)
        d2 = torch.cat([self.decoder2(d3, z), e1], dim=1)
        return self.decoder1(d2)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def ensure_results_dirs(results_root: Path) -> dict[str, Path]:
    dirs = {
        "root": results_root,
        "manifests": results_root / "manifests",
        "metrics": results_root / "metrics",
        "diversity": results_root / "diversity",
        "tables": results_root / "tables",
        "figures": results_root / "figures",
        "logs": results_root / "logs",
        "samples": results_root / "samples",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_int_seed(*parts: Any) -> int:
    text = "||".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.blake2b(text, digest_size=8).digest()
    return int.from_bytes(digest, "big") & 0x7FFFFFFF


def normalize_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    return {key.replace("module.", ""): value for key, value in state_dict.items()}


def choose_device(device_value: str) -> torch.device:
    if device_value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_value)


def set_torch_determinism() -> None:
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def family_sort_key(family: str) -> tuple[int, str]:
    family = str(family)
    order = {"1": 1, "2A": 2, "3": 3}
    return order[family], family


def model_folder_name(training_regime: str, family: str, target_code: str) -> str:
    suffix = "" if training_regime == "LALegacy" else "_MSASample"
    return f"{family}_{target_code}_{FAMILY_NAME_MAP[family]}{suffix}"


def run_dir_name(learning_rate: float) -> str:
    token = f"{learning_rate:g}".replace(".", "p")
    seed = {
        0.0001: 2026,
        0.0002: 3026,
        0.0005: 4026,
        0.001: 5026,
    }[round(float(learning_rate), 4)]
    return f"lr_{token}_seed_{seed}"


def run_dir_name_for_model(training_regime: str, family: str, target_code: str, learning_rate: float) -> str:
    if (
        training_regime == "MSASample"
        and family == "2A"
        and target_code == "BH"
        and round(float(learning_rate), 4) == 0.001
    ):
        return "lr_0p001_seed_2026"
    return run_dir_name(learning_rate)


def resolve_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (base_dir / path).resolve()


def load_config(config_path: Path) -> dict[str, Any]:
    raw = read_json(config_path)
    config_dir = config_path.resolve().parent
    resolved = dict(raw)
    resolved["config_path"] = config_path.resolve()
    resolved["config_dir"] = config_dir
    resolved["results_root"] = resolve_path(config_dir, raw["results_root"])
    resolved["model_roots"] = {
        key: resolve_path(config_dir, value)
        for key, value in raw["model_roots"].items()
    }
    resolved["dataset_specs"] = {
        key: DatasetConfig(
            name=key,
            root=resolve_path(config_dir, payload["root"]),
            bf_condition_dir=payload["bf_condition_dir"],
            bf_target_dir=payload["bf_target_dir"],
            bh_condition_dir=payload["bh_condition_dir"],
            bh_target_dir=payload["bh_target_dir"],
            manifest_path=resolve_path(config_dir, payload["manifest"])
            if payload.get("manifest") else None,
        )
        for key, payload in raw["dataset_specs"].items()
    }
    resolved.setdefault("device", "auto")
    resolved.setdefault("batch_size", 16)
    resolved.setdefault("cache_mode", "lazy")
    resolved.setdefault("pixel_regression_samples_per_bin", {"BF": 2000, "BH": 2000})
    resolved.setdefault("pixel_regression_tile_budget", {"BF": 128, "BH": 128})
    resolved.setdefault("pixel_regression_candidates_per_tile", 64)
    resolved.setdefault("diversity_subset_per_bin", {"BF": 80, "BH": 60})
    resolved.setdefault("representative_sample_target_modes", DEFAULT_SAMPLE_MODES)
    return resolved


def snapshot_runtime_environment(results_dirs: dict[str, Path], config: dict[str, Any]) -> None:
    payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "rasterio": rasterio.__version__,
        "config_path": str(config["config_path"]),
        "results_root": str(results_dirs["root"]),
        "timestamp_utc": now_utc_iso(),
    }
    write_json(results_dirs["manifests"] / "runtime_environment.json", payload)


def snapshot_dataset_manifests(results_dirs: dict[str, Path], config: dict[str, Any]) -> None:
    payload: dict[str, Any] = {}
    for name, spec in config["dataset_specs"].items():
        payload[name] = read_json(spec.manifest_path) if spec.manifest_path and spec.manifest_path.exists() else {}
    write_json(results_dirs["manifests"] / "dataset_manifests_snapshot.json", payload)


def encode_lulc_raw_batch(arrays: np.ndarray) -> np.ndarray:
    arrays = arrays.astype(np.float32, copy=False)
    arrays = arrays.copy()
    arrays[(arrays < 0) | (arrays > 95)] = 0.0
    return ((arrays / 95.0) * 2.0 - 1.0)[:, None, :, :].astype(np.float32)


def encode_lulc_onehot_batch(arrays: np.ndarray) -> np.ndarray:
    encoded = [(arrays == cls).astype(np.float32) for cls in NLCD_CLASSES]
    return np.stack(encoded, axis=1)


def encode_bf_unit(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(arr, 0.0, 1.0)


def encode_bf_scaled_batch(arrays: np.ndarray) -> np.ndarray:
    unit = np.stack([encode_bf_unit(arr) for arr in arrays], axis=0)
    return ((unit * 2.0) - 1.0)[:, None, :, :].astype(np.float32)


def decode_bf_scaled(arrays: np.ndarray) -> np.ndarray:
    arrays = np.clip(arrays.astype(np.float32), -1.0, 1.0)
    return np.clip((arrays + 1.0) / 2.0, 0.0, 1.0)


def height_feet_to_meters(arrays: np.ndarray) -> np.ndarray:
    return arrays.astype(np.float32) * 0.3048


def decode_height_log1p_to_meters(arrays: np.ndarray) -> np.ndarray:
    arrays = np.clip(arrays.astype(np.float32), -1.0, 1.0)
    return np.expm1((arrays + 1.0) * np.log1p(MAX_HEIGHT_M) / 2.0)


def encode_condition_batch(
    arrays: np.ndarray,
    condition_kind: str,
    condition_channels: int,
) -> np.ndarray:
    if condition_kind == "lulc":
        if condition_channels > 1:
            return encode_lulc_onehot_batch(arrays)
        return encode_lulc_raw_batch(arrays)
    if condition_kind == "bf":
        return encode_bf_scaled_batch(arrays)
    raise ValueError(f"Unsupported condition kind: {condition_kind}")


def decode_prediction_batch(
    arrays: np.ndarray,
    target_kind: str,
) -> np.ndarray:
    if target_kind == "bf":
        return decode_bf_scaled(arrays)
    if target_kind == "height":
        return decode_height_log1p_to_meters(arrays)
    raise ValueError(f"Unsupported target kind: {target_kind}")


def make_latent_batch(
    filenames: list[str],
    latent_dim: int,
    base_seed: int,
) -> torch.Tensor:
    rows = []
    for filename in filenames:
        rng = np.random.default_rng(stable_int_seed("latent", base_seed, filename))
        rows.append(rng.standard_normal(latent_dim, dtype=np.float32))
    return torch.from_numpy(np.stack(rows, axis=0))


def instantiate_generator(
    family: str,
    condition_channels: int,
    latent_dim: int | None = None,
) -> nn.Module:
    if family == "1":
        return GeneratorUNet(condition_channels=condition_channels)
    if family == "2A":
        return GeneratorRandomVec(condition_channels=condition_channels, latent_dim=int(latent_dim or 8))
    if family == "3":
        return GeneratorMultiRandom(condition_channels=condition_channels, latent_dim=int(latent_dim or 16))
    raise ValueError(f"Unsupported family: {family}")


def load_generator_from_run(
    family: str,
    checkpoint_path: Path,
    metadata: dict[str, Any],
    device: torch.device,
) -> nn.Module:
    model = instantiate_generator(
        family=family,
        condition_channels=int(metadata["condition_channels"]),
        latent_dim=metadata.get("latent_dim"),
    )
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(normalize_state_dict(state_dict))
    model.to(device)
    model.eval()
    return model


def batched(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for index in range(0, len(items), batch_size):
        yield items[index:index + batch_size]


def active_pixel_mask(target: str, reference: np.ndarray, prediction: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    if target == "BF":
        threshold = float(config["bf_active_threshold"])
        return (reference >= threshold) | (prediction >= threshold)
    threshold = float(config["bh_active_threshold_m"])
    return (reference > threshold) | (prediction > threshold)


def compute_bh_built_mean_m(arr: np.ndarray) -> float:
    positive = arr[arr > 0.0]
    if positive.size == 0:
        return 0.0
    return float(np.mean(positive))


def safe_linregress(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    reference = np.asarray(reference, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if reference.size < 2 or np.allclose(reference, reference[0]):
        return {
            "slope": math.nan,
            "intercept": math.nan,
            "r2": math.nan,
            "pearson_r": math.nan,
        }
    fit = scipy.stats.linregress(reference, prediction)
    r_value = float(fit.rvalue)
    return {
        "slope": float(fit.slope),
        "intercept": float(fit.intercept),
        "r2": float(r_value * r_value),
        "pearson_r": r_value,
    }


def bootstrap_indices_array(
    results_dirs: dict[str, Path],
    n_tiles: int,
    config: dict[str, Any],
) -> np.ndarray:
    path = results_dirs["manifests"] / "bootstrap_indices.npz"
    seed = int(config["bootstrap_seed"])
    reps = int(config["bootstrap_reps"])
    if path.exists():
        payload = np.load(path)
        if (
            int(payload["n_tiles"]) == n_tiles
            and int(payload["bootstrap_seed"]) == seed
            and int(payload["bootstrap_reps"]) == reps
        ):
            return payload["indices"]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n_tiles, size=(reps, n_tiles), endpoint=False)
    np.savez_compressed(
        path,
        indices=indices,
        n_tiles=np.array(n_tiles),
        bootstrap_seed=np.array(seed),
        bootstrap_reps=np.array(reps),
    )
    return indices


def metric_interval_from_bootstrap(
    values: np.ndarray,
    indices: np.ndarray,
    kind: str,
) -> tuple[float, float]:
    draws = values[indices]
    if kind == "mae":
        stats = np.mean(np.abs(draws), axis=1)
    elif kind == "rmse":
        stats = np.sqrt(np.mean(draws * draws, axis=1))
    elif kind == "mbe":
        stats = np.mean(draws, axis=1)
    else:
        raise ValueError(f"Unsupported bootstrap metric kind: {kind}")
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def pixel_metric_interval_from_bootstrap(
    abs_sums: np.ndarray,
    sq_sums: np.ndarray,
    signed_sums: np.ndarray,
    counts: np.ndarray,
    indices: np.ndarray,
) -> dict[str, float]:
    abs_draws = np.sum(abs_sums[indices], axis=1)
    sq_draws = np.sum(sq_sums[indices], axis=1)
    signed_draws = np.sum(signed_sums[indices], axis=1)
    count_draws = np.maximum(np.sum(counts[indices], axis=1), 1.0)
    mae = abs_draws / count_draws
    rmse = np.sqrt(sq_draws / count_draws)
    mbe = signed_draws / count_draws
    return {
        "pixel_mae_ci_low": float(np.percentile(mae, 2.5)),
        "pixel_mae_ci_high": float(np.percentile(mae, 97.5)),
        "pixel_rmse_ci_low": float(np.percentile(rmse, 2.5)),
        "pixel_rmse_ci_high": float(np.percentile(rmse, 97.5)),
        "pixel_mbe_ci_low": float(np.percentile(mbe, 2.5)),
        "pixel_mbe_ci_high": float(np.percentile(mbe, 97.5)),
    }


class PointAccumulator:
    def __init__(
        self,
        *,
        row_id: str,
        group: GroupSpec,
        target: str,
        evaluation_mode: str,
        sample_lookup: dict[str, np.ndarray],
    ) -> None:
        self.row_id = row_id
        self.group = group
        self.target = target
        self.evaluation_mode = evaluation_mode
        self.sample_lookup = sample_lookup
        self.tile_mean_rows: list[dict[str, Any]] = []
        self.tile_heterogeneity_rows: list[dict[str, Any]] = []
        self.mean_diff: list[float] = []
        self.heterogeneity_diff: list[float] = []
        self.pixel_abs_sums: list[float] = []
        self.pixel_sq_sums: list[float] = []
        self.pixel_signed_sums: list[float] = []
        self.pixel_counts: list[int] = []
        self.pixel_sample_reference: list[np.ndarray] = []
        self.pixel_sample_prediction: list[np.ndarray] = []

    def update(self, filename: str, reference: np.ndarray, prediction: np.ndarray) -> None:
        tile_mean_reference = float(np.mean(reference))
        tile_mean_prediction = float(np.mean(prediction))
        tile_std_reference = float(np.std(reference))
        tile_std_prediction = float(np.std(prediction))
        tile_mean_bias = tile_mean_prediction - tile_mean_reference
        tile_std_bias = tile_std_prediction - tile_std_reference
        diff = prediction - reference
        mask = active_pixel_mask(self.target, reference, prediction, ACTIVE_CONFIG)
        active_diff = diff[mask]
        if active_diff.size == 0:
            pixel_abs_sum = 0.0
            pixel_sq_sum = 0.0
            pixel_signed_sum = 0.0
            pixel_count = 0
            tile_pixel_mae = 0.0
            tile_pixel_rmse = 0.0
            tile_pixel_mbe = 0.0
        else:
            pixel_abs_sum = float(np.sum(np.abs(active_diff)))
            pixel_sq_sum = float(np.sum(active_diff * active_diff))
            pixel_signed_sum = float(np.sum(active_diff))
            pixel_count = int(active_diff.size)
            tile_pixel_mae = pixel_abs_sum / pixel_count
            tile_pixel_rmse = math.sqrt(pixel_sq_sum / pixel_count)
            tile_pixel_mbe = pixel_signed_sum / pixel_count
        reference_bin_value = tile_mean_reference if self.target == "BF" else compute_bh_built_mean_m(reference)
        row_common = {
            "row_id": self.row_id,
            "group_id": self.group.group_id,
            "target": self.target.lower(),
            "evaluation_mode": self.evaluation_mode,
            "training_regime": self.group.training_regime,
            "family": self.group.family,
            "dataset_name": self.group.dataset_name,
            "learning_rate": self.group.learning_rate,
            "checkpoint": self.group.checkpoint,
            "filename": filename,
            "reference_bin_value": reference_bin_value,
            "tile_pixel_mae": tile_pixel_mae,
            "tile_pixel_rmse": tile_pixel_rmse,
            "tile_pixel_mbe": tile_pixel_mbe,
            "pixel_active_count": pixel_count,
            "pixel_abs_error_sum": pixel_abs_sum,
            "pixel_sq_error_sum": pixel_sq_sum,
            "pixel_signed_error_sum": pixel_signed_sum,
        }
        self.tile_mean_rows.append(
            {
                **row_common,
                "reference_value": tile_mean_reference,
                "prediction_value": tile_mean_prediction,
                "bias": tile_mean_bias,
            }
        )
        self.tile_heterogeneity_rows.append(
            {
                **row_common,
                "reference_value": tile_std_reference,
                "prediction_value": tile_std_prediction,
                "bias": tile_std_bias,
            }
        )
        self.mean_diff.append(tile_mean_bias)
        self.heterogeneity_diff.append(tile_std_bias)
        self.pixel_abs_sums.append(pixel_abs_sum)
        self.pixel_sq_sums.append(pixel_sq_sum)
        self.pixel_signed_sums.append(pixel_signed_sum)
        self.pixel_counts.append(pixel_count)
        sample_positions = self.sample_lookup.get(filename)
        if sample_positions is not None and sample_positions.size:
            ref_values = reference[sample_positions[:, 0], sample_positions[:, 1]]
            pred_values = prediction[sample_positions[:, 0], sample_positions[:, 1]]
            sample_mask = active_pixel_mask(self.target, ref_values, pred_values, ACTIVE_CONFIG)
            if np.any(sample_mask):
                self.pixel_sample_reference.append(ref_values[sample_mask])
                self.pixel_sample_prediction.append(pred_values[sample_mask])

    def finalize(self, bootstrap_indices: np.ndarray) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
        mean_reference = np.array([row["reference_value"] for row in self.tile_mean_rows], dtype=np.float64)
        mean_prediction = np.array([row["prediction_value"] for row in self.tile_mean_rows], dtype=np.float64)
        heter_reference = np.array([row["reference_value"] for row in self.tile_heterogeneity_rows], dtype=np.float64)
        heter_prediction = np.array([row["prediction_value"] for row in self.tile_heterogeneity_rows], dtype=np.float64)
        mean_diff = np.array(self.mean_diff, dtype=np.float64)
        heter_diff = np.array(self.heterogeneity_diff, dtype=np.float64)
        pixel_abs_sums = np.array(self.pixel_abs_sums, dtype=np.float64)
        pixel_sq_sums = np.array(self.pixel_sq_sums, dtype=np.float64)
        pixel_signed_sums = np.array(self.pixel_signed_sums, dtype=np.float64)
        pixel_counts = np.array(self.pixel_counts, dtype=np.float64)
        mean_reg = safe_linregress(mean_reference, mean_prediction)
        heter_reg = safe_linregress(heter_reference, heter_prediction)
        if self.pixel_sample_reference and self.pixel_sample_prediction:
            pixel_sample_reference = np.concatenate(self.pixel_sample_reference)
            pixel_sample_prediction = np.concatenate(self.pixel_sample_prediction)
            pixel_reg = safe_linregress(pixel_sample_reference, pixel_sample_prediction)
        else:
            pixel_sample_reference = np.array([], dtype=np.float64)
            pixel_sample_prediction = np.array([], dtype=np.float64)
            pixel_reg = {"slope": math.nan, "intercept": math.nan, "r2": math.nan, "pearson_r": math.nan}
        pixel_count_total = float(np.sum(pixel_counts))
        overall = {
            "row_id": self.row_id,
            "group_id": self.group.group_id,
            "target": self.target.lower(),
            "evaluation_mode": self.evaluation_mode,
            "training_regime": self.group.training_regime,
            "family": self.group.family,
            "dataset_name": self.group.dataset_name,
            "learning_rate": self.group.learning_rate,
            "checkpoint": self.group.checkpoint,
            "n_tiles": len(self.tile_mean_rows),
            "tile_mean_mae": float(np.mean(np.abs(mean_diff))),
            "tile_mean_rmse": float(np.sqrt(np.mean(mean_diff * mean_diff))),
            "tile_mean_mbe": float(np.mean(mean_diff)),
            "tile_mean_slope": mean_reg["slope"],
            "tile_mean_intercept": mean_reg["intercept"],
            "tile_mean_r2": mean_reg["r2"],
            "tile_mean_pearson_r": mean_reg["pearson_r"],
            "tile_heterogeneity_mae": float(np.mean(np.abs(heter_diff))),
            "tile_heterogeneity_rmse": float(np.sqrt(np.mean(heter_diff * heter_diff))),
            "tile_heterogeneity_mbe": float(np.mean(heter_diff)),
            "tile_heterogeneity_slope": heter_reg["slope"],
            "tile_heterogeneity_intercept": heter_reg["intercept"],
            "tile_heterogeneity_r2": heter_reg["r2"],
            "tile_heterogeneity_pearson_r": heter_reg["pearson_r"],
            "pixel_mae": float(np.sum(pixel_abs_sums) / max(pixel_count_total, 1.0)),
            "pixel_rmse": float(np.sqrt(np.sum(pixel_sq_sums) / max(pixel_count_total, 1.0))),
            "pixel_mbe": float(np.sum(pixel_signed_sums) / max(pixel_count_total, 1.0)),
            "pixel_active_count": int(pixel_count_total),
            "pixel_sample_size": int(pixel_sample_reference.size),
            "pixel_regression_slope": pixel_reg["slope"],
            "pixel_regression_intercept": pixel_reg["intercept"],
            "pixel_regression_r2": pixel_reg["r2"],
            "pixel_regression_pearson_r": pixel_reg["pearson_r"],
        }
        overall["tile_mean_mae_ci_low"], overall["tile_mean_mae_ci_high"] = metric_interval_from_bootstrap(mean_diff, bootstrap_indices, "mae")
        overall["tile_mean_rmse_ci_low"], overall["tile_mean_rmse_ci_high"] = metric_interval_from_bootstrap(mean_diff, bootstrap_indices, "rmse")
        overall["tile_mean_mbe_ci_low"], overall["tile_mean_mbe_ci_high"] = metric_interval_from_bootstrap(mean_diff, bootstrap_indices, "mbe")
        overall["tile_heterogeneity_mae_ci_low"], overall["tile_heterogeneity_mae_ci_high"] = metric_interval_from_bootstrap(heter_diff, bootstrap_indices, "mae")
        overall["tile_heterogeneity_rmse_ci_low"], overall["tile_heterogeneity_rmse_ci_high"] = metric_interval_from_bootstrap(heter_diff, bootstrap_indices, "rmse")
        overall["tile_heterogeneity_mbe_ci_low"], overall["tile_heterogeneity_mbe_ci_high"] = metric_interval_from_bootstrap(heter_diff, bootstrap_indices, "mbe")
        overall.update(
            pixel_metric_interval_from_bootstrap(
                pixel_abs_sums,
                pixel_sq_sums,
                pixel_signed_sums,
                pixel_counts,
                bootstrap_indices,
            )
        )
        return overall, self.tile_mean_rows, self.tile_heterogeneity_rows


ACTIVE_CONFIG: dict[str, Any] = {}


def set_active_config(config: dict[str, Any]) -> None:
    ACTIVE_CONFIG.clear()
    ACTIVE_CONFIG.update(config)


def append_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def load_existing_keys(path: Path, key_columns: list[str]) -> set[Any]:
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if len(key_columns) == 1:
        column = key_columns[0]
        return set(df[column].tolist())
    return {tuple(row[column] for column in key_columns) for _, row in df.iterrows()}


def append_log_rows(results_dirs: dict[str, Path], rows: list[dict[str, Any]]) -> None:
    append_csv_rows(results_dirs["logs"] / "completed_rows.csv", rows)


def infer_batch(
    model: nn.Module,
    family: str,
    condition_tensor: torch.Tensor,
    filenames: list[str],
    latent_seed: int,
    device: torch.device,
) -> np.ndarray:
    condition_tensor = condition_tensor.to(device=device, dtype=torch.float32, non_blocking=True)
    with torch.no_grad():
        if family == "1":
            output = model(condition_tensor)
        else:
            latent_dim = int(getattr(model, "latent_dim"))
            latent = make_latent_batch(filenames, latent_dim, latent_seed).to(device=device, dtype=torch.float32)
            output = model(condition_tensor, latent)
    return output.detach().cpu().numpy()[:, 0, :, :]


def make_group_specs(config: dict[str, Any]) -> list[GroupSpec]:
    groups: list[GroupSpec] = []
    for training_regime, model_root in config["model_roots"].items():
        for family in sorted(config["active_families"], key=family_sort_key):
            bf_folder = model_folder_name(training_regime, family, "BF")
            bh_folder = model_folder_name(training_regime, family, "BH")
            for dataset_name in sorted(config["dataset_specs"]):
                for learning_rate in config["learning_rates"]:
                    bf_run_dir = model_root / bf_folder / run_dir_name_for_model(
                        training_regime,
                        family,
                        "BF",
                        float(learning_rate),
                    )
                    bh_run_dir = model_root / bh_folder / run_dir_name_for_model(
                        training_regime,
                        family,
                        "BH",
                        float(learning_rate),
                    )
                    bf_metadata = read_json(bf_run_dir / "metadata.json")
                    bh_metadata = read_json(bh_run_dir / "metadata.json")
                    for checkpoint in config["checkpoints"]:
                        bf_checkpoint = bf_run_dir / f"generator_epoch_{int(checkpoint):04d}.pth"
                        bh_checkpoint = bh_run_dir / f"generator_epoch_{int(checkpoint):04d}.pth"
                        if not bf_checkpoint.exists() or not bh_checkpoint.exists():
                            raise FileNotFoundError(f"Missing checkpoint for {training_regime} {family} lr={learning_rate} checkpoint={checkpoint}")
                        group_id = "::".join(
                            [
                                training_regime,
                                family,
                                dataset_name,
                                f"lr_{learning_rate:g}",
                                f"ckpt_{int(checkpoint):04d}",
                            ]
                        )
                        groups.append(
                            GroupSpec(
                                group_id=group_id,
                                training_regime=training_regime,
                                family=family,
                                dataset_name=dataset_name,
                                learning_rate=float(learning_rate),
                                checkpoint=int(checkpoint),
                                bf_run_dir=bf_run_dir,
                                bh_run_dir=bh_run_dir,
                                bf_checkpoint_path=bf_checkpoint,
                                bh_checkpoint_path=bh_checkpoint,
                                bf_metadata=bf_metadata,
                                bh_metadata=bh_metadata,
                            )
                        )
    groups.sort(
        key=lambda item: (
            item.training_regime,
            family_sort_key(item.family),
            item.dataset_name,
            item.learning_rate,
            item.checkpoint,
        )
    )
    return groups


def evaluation_matrix_rows(groups: list[GroupSpec]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in groups:
        common = {
            "group_id": group.group_id,
            "training_regime": group.training_regime,
            "family": group.family,
            "dataset_name": group.dataset_name,
            "learning_rate": group.learning_rate,
            "checkpoint": group.checkpoint,
        }
        for stage, target, evaluation_mode in [
            (POINT_STAGE, "BF", STANDARD_MODE),
            (POINT_STAGE, "BH", ORACLE_MODE),
            (POINT_STAGE, "BH", PIPELINE_MODE),
            (DIVERSITY_STAGE, "BF", STANDARD_MODE),
            (DIVERSITY_STAGE, "BH", ORACLE_MODE),
            (DIVERSITY_STAGE, "BH", PIPELINE_MODE),
        ]:
            rows.append(
                {
                    **common,
                    "row_id": make_row_id(stage, group, target, evaluation_mode),
                    "stage": stage,
                    "target": target,
                    "evaluation_mode": evaluation_mode,
                    "requires_diversity_inference": stage == DIVERSITY_STAGE and group.family in {"2A", "3"},
                }
            )
    return rows


def write_evaluation_matrix(results_dirs: dict[str, Path], groups: list[GroupSpec]) -> None:
    rows = evaluation_matrix_rows(groups)
    path = results_dirs["manifests"] / "evaluation_matrix.csv"
    if path.exists():
        path.unlink()
    append_csv_rows(path, rows)


def make_row_id(stage: str, group: GroupSpec, target: str, evaluation_mode: str) -> str:
    return "::".join(
        [
            stage,
            target,
            evaluation_mode,
            group.training_regime,
            group.family,
            group.dataset_name,
            f"lr_{group.learning_rate:g}",
            f"ckpt_{group.checkpoint:04d}",
        ]
    )


def tile_bin_label(value: float, bins: list[dict[str, Any]]) -> str | None:
    for entry in bins:
        lower = float(entry["min"])
        upper = entry.get("max")
        upper_value = float(upper) if upper is not None else None
        if entry.get("inclusive_upper", False):
            if value >= lower and (upper_value is None or value <= upper_value):
                return str(entry["label"])
        else:
            if value >= lower and (upper_value is None or value < upper_value):
                return str(entry["label"])
    return None


def compute_reference_tile_values(
    dataset: DatasetBundle,
    filenames: list[str],
    target: str,
) -> dict[str, float]:
    values: dict[str, float] = {}
    for filename in filenames:
        _central, bf_arr, bh_arr = dataset.get_tile(filename)
        if target == "BF":
            values[filename] = float(np.mean(encode_bf_unit(bf_arr)))
        else:
            bh_m = height_feet_to_meters(bh_arr)
            values[filename] = compute_bh_built_mean_m(bh_m)
    return values


def deterministic_sample(items: list[str], sample_size: int, seed_parts: tuple[Any, ...]) -> list[str]:
    if len(items) <= sample_size:
        return list(items)
    keyed = sorted(items, key=lambda value: stable_int_seed(*seed_parts, value))
    return keyed[:sample_size]


def build_diversity_subset(
    dataset: DatasetBundle,
    filenames: list[str],
    target: str,
    config: dict[str, Any],
) -> pd.DataFrame:
    bins = config["bf_bins"] if target == "BF" else config["bh_bins_m"]
    per_bin = int(config["diversity_subset_per_bin"][target])
    values = compute_reference_tile_values(dataset, filenames, target)
    by_bin: dict[str, list[str]] = defaultdict(list)
    for filename, value in values.items():
        label = tile_bin_label(value, bins)
        if label is not None:
            by_bin[label].append(filename)
    chosen: list[dict[str, Any]] = []
    used: set[str] = set()
    for entry in bins:
        label = str(entry["label"])
        selected = deterministic_sample(sorted(by_bin.get(label, [])), per_bin, ("diversity", dataset.spec.name, target, label))
        for filename in selected:
            chosen.append(
                {
                    "dataset_name": dataset.spec.name,
                    "target": target.lower(),
                    "filename": filename,
                    "reference_bin": label,
                    "reference_bin_value": values[filename],
                }
            )
            used.add(filename)
    shortfall = max(0, per_bin * len(bins) - len(chosen))
    if shortfall:
        remaining = [filename for filename in filenames if filename not in used]
        for filename in deterministic_sample(remaining, shortfall, ("diversity-fill", dataset.spec.name, target)):
            chosen.append(
                {
                    "dataset_name": dataset.spec.name,
                    "target": target.lower(),
                    "filename": filename,
                    "reference_bin": "fill",
                    "reference_bin_value": values[filename],
                }
            )
    return pd.DataFrame(chosen)


def write_diversity_subsets(results_dirs: dict[str, Path], datasets: dict[str, DatasetBundle], filenames_by_dataset: dict[str, list[str]], config: dict[str, Any]) -> dict[tuple[str, str], list[str]]:
    out: dict[tuple[str, str], list[str]] = {}
    for dataset_name, dataset in datasets.items():
        for target in ("BF", "BH"):
            subset = build_diversity_subset(dataset, filenames_by_dataset[dataset_name], target, config)
            path = results_dirs["manifests"] / f"diversity_subsets_{dataset_name}_{target.lower()}.csv"
            subset.to_csv(path, index=False)
            out[(dataset_name, target)] = subset["filename"].tolist()
    return out


def build_pixel_regression_sample(
    dataset: DatasetBundle,
    filenames: list[str],
    target: str,
    config: dict[str, Any],
) -> pd.DataFrame:
    if target == "BF":
        bins = [
            {"label": "0-0.01", "min": 0.0, "max": 0.01},
            *config["bf_bins"],
        ]
        tile_bins = config["bf_bins"]
    else:
        bins = [
            {"label": "0-0.5", "min": 0.0, "max": 0.5},
            *config["bh_bins_m"],
        ]
        tile_bins = config["bh_bins_m"]
    tile_budget = int(config["pixel_regression_tile_budget"][target])
    target_per_bin = int(config["pixel_regression_samples_per_bin"][target])
    candidates_per_tile = int(config["pixel_regression_candidates_per_tile"])
    reference_tile_values = compute_reference_tile_values(dataset, filenames, target)
    selected_tiles: list[str] = []
    per_tile_group = max(1, tile_budget // max(len(tile_bins), 1))
    grouped: dict[str, list[str]] = defaultdict(list)
    for filename, value in reference_tile_values.items():
        label = tile_bin_label(value, tile_bins)
        if label is not None:
            grouped[label].append(filename)
    for entry in tile_bins:
        label = str(entry["label"])
        selected_tiles.extend(
            deterministic_sample(
                sorted(grouped.get(label, [])),
                per_tile_group,
                ("pixel-tile-sample", dataset.spec.name, target, label),
            )
        )
    if len(selected_tiles) < tile_budget:
        fill = [filename for filename in filenames if filename not in set(selected_tiles)]
        selected_tiles.extend(deterministic_sample(fill, tile_budget - len(selected_tiles), ("pixel-tile-fill", dataset.spec.name, target)))
    selected_tiles = selected_tiles[:tile_budget]
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(stable_int_seed("pixel-sample", dataset.spec.name, target))
    by_bin_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for filename in selected_tiles:
        _central, bf_arr, bh_arr = dataset.get_tile(filename)
        reference = encode_bf_unit(bf_arr) if target == "BF" else height_feet_to_meters(bh_arr)
        for entry in bins:
            label = str(entry["label"])
            lower = float(entry["min"])
            upper = entry.get("max")
            upper_value = float(upper) if upper is not None else None
            if upper_value is None:
                mask = reference >= lower
            else:
                mask = (reference >= lower) & (reference < upper_value)
            flat_indices = np.flatnonzero(mask)
            if flat_indices.size == 0:
                continue
            take = min(flat_indices.size, candidates_per_tile)
            chosen = rng.choice(flat_indices, size=take, replace=False)
            for flat_index in chosen:
                row_index, col_index = np.unravel_index(int(flat_index), reference.shape)
                by_bin_rows[label].append(
                    {
                        "dataset_name": dataset.spec.name,
                        "target": target.lower(),
                        "filename": filename,
                        "row": int(row_index),
                        "col": int(col_index),
                        "reference_value": float(reference[row_index, col_index]),
                        "reference_bin": label,
                    }
                )
    for entry in bins:
        label = str(entry["label"])
        candidates = by_bin_rows.get(label, [])
        if len(candidates) <= target_per_bin:
            rows.extend(candidates)
        else:
            order = np.argsort(
                [stable_int_seed("pixel-final", dataset.spec.name, target, label, item["filename"], item["row"], item["col"]) for item in candidates]
            )
            rows.extend([candidates[int(index)] for index in order[:target_per_bin]])
    return pd.DataFrame(rows)


def write_pixel_regression_sample(
    results_dirs: dict[str, Path],
    datasets: dict[str, DatasetBundle],
    filenames_by_dataset: dict[str, list[str]],
    config: dict[str, Any],
) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    tables: list[pd.DataFrame] = []
    lookup: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for dataset_name, dataset in datasets.items():
        for target in ("BF", "BH"):
            df = build_pixel_regression_sample(dataset, filenames_by_dataset[dataset_name], target, config)
            tables.append(df)
            mapping: dict[str, list[list[int]]] = defaultdict(list)
            for _, row in df.iterrows():
                mapping[str(row["filename"])].append([int(row["row"]), int(row["col"])])
            lookup[(dataset_name, target)] = {
                key: np.asarray(value, dtype=np.int64)
                for key, value in mapping.items()
            }
    combined = pd.concat(tables, ignore_index=True)
    combined.to_csv(results_dirs["manifests"] / "pixel_regression_sample.csv", index=False)
    return lookup


def evaluate_point_group(
    group: GroupSpec,
    dataset: DatasetBundle,
    filenames: list[str],
    config: dict[str, Any],
    device: torch.device,
    bootstrap_indices: np.ndarray,
    pixel_sample_lookup: dict[tuple[str, str], dict[str, np.ndarray]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], pd.DataFrame]:
    bf_model = load_generator_from_run(group.family, group.bf_checkpoint_path, group.bf_metadata, device)
    bh_model = load_generator_from_run(group.family, group.bh_checkpoint_path, group.bh_metadata, device)
    bf_acc = PointAccumulator(
        row_id=make_row_id(POINT_STAGE, group, "BF", STANDARD_MODE),
        group=group,
        target="BF",
        evaluation_mode=STANDARD_MODE,
        sample_lookup=pixel_sample_lookup[(group.dataset_name, "BF")],
    )
    bh_oracle_acc = PointAccumulator(
        row_id=make_row_id(POINT_STAGE, group, "BH", ORACLE_MODE),
        group=group,
        target="BH",
        evaluation_mode=ORACLE_MODE,
        sample_lookup=pixel_sample_lookup[(group.dataset_name, "BH")],
    )
    bh_pipeline_acc = PointAccumulator(
        row_id=make_row_id(POINT_STAGE, group, "BH", PIPELINE_MODE),
        group=group,
        target="BH",
        evaluation_mode=PIPELINE_MODE,
        sample_lookup=pixel_sample_lookup[(group.dataset_name, "BH")],
    )
    batch_size = int(config["batch_size"])
    primary_seed = int(config["primary_latent_seed"])
    for batch_filenames in batched(filenames, batch_size):
        central_batch = np.stack([dataset.get_tile(filename)[0] for filename in batch_filenames], axis=0)
        bf_reference_batch = np.stack([encode_bf_unit(dataset.get_tile(filename)[1]) for filename in batch_filenames], axis=0)
        bh_reference_batch = np.stack([height_feet_to_meters(dataset.get_tile(filename)[2]) for filename in batch_filenames], axis=0)
        bf_condition_np = encode_condition_batch(
            central_batch,
            condition_kind=str(group.bf_metadata["condition_kind"]),
            condition_channels=int(group.bf_metadata["condition_channels"]),
        )
        bf_predictions_scaled = infer_batch(
            bf_model,
            group.family,
            torch.from_numpy(bf_condition_np),
            batch_filenames,
            primary_seed,
            device,
        )
        bf_predictions = decode_prediction_batch(bf_predictions_scaled, "bf")
        bh_oracle_condition_np = encode_condition_batch(
            bf_reference_batch,
            condition_kind=str(group.bh_metadata["condition_kind"]),
            condition_channels=int(group.bh_metadata["condition_channels"]),
        )
        bh_oracle_scaled = infer_batch(
            bh_model,
            group.family,
            torch.from_numpy(bh_oracle_condition_np),
            batch_filenames,
            primary_seed,
            device,
        )
        bh_oracle_predictions = decode_prediction_batch(bh_oracle_scaled, "height")
        bh_pipeline_condition_np = encode_condition_batch(
            bf_predictions,
            condition_kind=str(group.bh_metadata["condition_kind"]),
            condition_channels=int(group.bh_metadata["condition_channels"]),
        )
        bh_pipeline_scaled = infer_batch(
            bh_model,
            group.family,
            torch.from_numpy(bh_pipeline_condition_np),
            batch_filenames,
            primary_seed,
            device,
        )
        bh_pipeline_predictions = decode_prediction_batch(bh_pipeline_scaled, "height")
        for index, filename in enumerate(batch_filenames):
            bf_acc.update(filename, bf_reference_batch[index], bf_predictions[index])
            bh_oracle_acc.update(filename, bh_reference_batch[index], bh_oracle_predictions[index])
            bh_pipeline_acc.update(filename, bh_reference_batch[index], bh_pipeline_predictions[index])
    bf_overall, bf_tile_mean_rows, bf_tile_heter_rows = bf_acc.finalize(bootstrap_indices)
    bh_oracle_overall, bh_oracle_tile_mean_rows, bh_oracle_tile_heter_rows = bh_oracle_acc.finalize(bootstrap_indices)
    bh_pipeline_overall, bh_pipeline_tile_mean_rows, bh_pipeline_tile_heter_rows = bh_pipeline_acc.finalize(bootstrap_indices)
    bias_rows = compute_bias_rows(
        [
            (bf_acc, bf_tile_mean_rows),
            (bh_oracle_acc, bh_oracle_tile_mean_rows),
            (bh_pipeline_acc, bh_pipeline_tile_mean_rows),
        ],
        config,
    )
    bh_comparison = compare_bh_pipeline_vs_oracle(
        group,
        bh_oracle_tile_mean_rows,
        bh_pipeline_tile_mean_rows,
        bh_oracle_overall,
        bh_pipeline_overall,
    )
    del bf_model
    del bh_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return (
        [bf_overall, bh_oracle_overall, bh_pipeline_overall],
        bf_tile_mean_rows + bh_oracle_tile_mean_rows + bh_pipeline_tile_mean_rows,
        bf_tile_heter_rows + bh_oracle_tile_heter_rows + bh_pipeline_tile_heter_rows,
        bias_rows,
        bh_comparison,
    )


def compute_bias_rows(
    accumulators: list[tuple[PointAccumulator, list[dict[str, Any]]]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for accumulator, tile_rows in accumulators:
        bins = config["bf_bins"] if accumulator.target == "BF" else config["bh_bins_m"]
        frame = pd.DataFrame(tile_rows)
        if frame.empty:
            continue
        for entry in bins:
            label = str(entry["label"])
            lower = float(entry["min"])
            upper = entry.get("max")
            upper_value = float(upper) if upper is not None else None
            if upper_value is None:
                subset = frame[frame["reference_bin_value"] >= lower]
            else:
                subset = frame[(frame["reference_bin_value"] >= lower) & (frame["reference_bin_value"] < upper_value)]
            if subset.empty:
                continue
            rows.append(
                {
                    "row_id": accumulator.row_id,
                    "group_id": accumulator.group.group_id,
                    "target": accumulator.target.lower(),
                    "evaluation_mode": accumulator.evaluation_mode,
                    "training_regime": accumulator.group.training_regime,
                    "family": accumulator.group.family,
                    "dataset_name": accumulator.group.dataset_name,
                    "learning_rate": accumulator.group.learning_rate,
                    "checkpoint": accumulator.group.checkpoint,
                    "reference_bin": label,
                    "n_tiles": int(len(subset)),
                    "mean_bias": float(subset["bias"].mean()),
                    "median_bias": float(subset["bias"].median()),
                    "mean_abs_bias": float(np.abs(subset["bias"]).mean()),
                    "mean_tile_pixel_mae": float(subset["tile_pixel_mae"].mean()),
                }
            )
    return rows


def compare_bh_pipeline_vs_oracle(
    group: GroupSpec,
    oracle_rows: list[dict[str, Any]],
    pipeline_rows: list[dict[str, Any]],
    oracle_overall: dict[str, Any],
    pipeline_overall: dict[str, Any],
) -> pd.DataFrame:
    oracle_frame = pd.DataFrame(oracle_rows)
    pipeline_frame = pd.DataFrame(pipeline_rows)
    merged = oracle_frame.merge(
        pipeline_frame,
        on=["filename"],
        suffixes=("_oracle", "_pipeline"),
    )
    merged["tile_pixel_mae_delta_pipeline_minus_oracle"] = merged["tile_pixel_mae_pipeline"] - merged["tile_pixel_mae_oracle"]
    merged["tile_pixel_mbe_delta_pipeline_minus_oracle"] = merged["tile_pixel_mbe_pipeline"] - merged["tile_pixel_mbe_oracle"]
    summary = pd.DataFrame(
        [
            {
                "group_id": group.group_id,
                "training_regime": group.training_regime,
                "family": group.family,
                "dataset_name": group.dataset_name,
                "learning_rate": group.learning_rate,
                "checkpoint": group.checkpoint,
                "oracle_tile_mean_mae": oracle_overall["tile_mean_mae"],
                "pipeline_tile_mean_mae": pipeline_overall["tile_mean_mae"],
                "oracle_pixel_mae": oracle_overall["pixel_mae"],
                "pipeline_pixel_mae": pipeline_overall["pixel_mae"],
                "pipeline_minus_oracle_tile_mean_mae": pipeline_overall["tile_mean_mae"] - oracle_overall["tile_mean_mae"],
                "pipeline_minus_oracle_pixel_mae": pipeline_overall["pixel_mae"] - oracle_overall["pixel_mae"],
                "mean_tile_pixel_mae_delta_pipeline_minus_oracle": float(merged["tile_pixel_mae_delta_pipeline_minus_oracle"].mean()),
                "mean_tile_pixel_mbe_delta_pipeline_minus_oracle": float(merged["tile_pixel_mbe_delta_pipeline_minus_oracle"].mean()),
            }
        ]
    )
    return summary


def load_point_tile_rows(results_dirs: dict[str, Path]) -> pd.DataFrame:
    path = results_dirs["metrics"] / "tile_level_mean_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype={"family": str}, low_memory=False)


def select_representative_sample_rows(point_df: pd.DataFrame, config: dict[str, Any]) -> set[str]:
    if point_df.empty:
        return set()
    output: set[str] = set()
    sample_modes = config["representative_sample_target_modes"]
    eligible = point_df[
        point_df["family"].isin(["2A", "3"])
        & (
            ((point_df["target"] == "bf") & (point_df["evaluation_mode"] == sample_modes["BF"]))
            | ((point_df["target"] == "bh") & (point_df["evaluation_mode"] == sample_modes["BH"]))
        )
    ].copy()
    if eligible.empty:
        return set()
    eligible["sort_metric"] = eligible["tile_mean_mae"]
    for _, subset in eligible.groupby(["target", "evaluation_mode", "training_regime", "family", "dataset_name"]):
        output.add(str(subset.sort_values(["sort_metric", "learning_rate", "checkpoint"]).iloc[0]["row_id"]))
    return output


def predict_subset_for_seed(
    *,
    model: nn.Module,
    family: str,
    condition_np: np.ndarray,
    filenames: list[str],
    target_kind: str,
    latent_seed: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    for start in range(0, len(filenames), batch_size):
        end = start + batch_size
        batch_filenames = filenames[start:end]
        batch_condition = torch.from_numpy(condition_np[start:end])
        prediction_scaled = infer_batch(model, family, batch_condition, batch_filenames, latent_seed, device)
        outputs.append(decode_prediction_batch(prediction_scaled, target_kind))
    return np.concatenate(outputs, axis=0)


def compute_diversity_tile_metrics(
    *,
    predictions: np.ndarray,
    reference: np.ndarray,
    target: str,
    config: dict[str, Any],
) -> list[dict[str, float]]:
    n_samples, n_tiles, _, _ = predictions.shape
    rows: list[dict[str, float]] = []
    for tile_index in range(n_tiles):
        tile_reference = reference[tile_index]
        tile_predictions = predictions[:, tile_index]
        union_mask = np.zeros_like(tile_reference, dtype=bool)
        for sample_index in range(n_samples):
            union_mask |= active_pixel_mask(target, tile_reference, tile_predictions[sample_index], config)
        if not np.any(union_mask):
            pairwise = 0.0
            pixel_std = 0.0
        else:
            pairwise_values = []
            for left in range(n_samples):
                for right in range(left + 1, n_samples):
                    pairwise_values.append(float(np.mean(np.abs(tile_predictions[left][union_mask] - tile_predictions[right][union_mask]))))
            pairwise = float(np.mean(pairwise_values)) if pairwise_values else 0.0
            pixel_std = float(np.mean(np.std(tile_predictions[:, union_mask], axis=0)))
        tile_means = np.mean(tile_predictions, axis=(1, 2))
        tile_stds = np.std(tile_predictions, axis=(1, 2))
        first_diff = tile_predictions[0] - tile_reference
        first_mask = active_pixel_mask(target, tile_reference, tile_predictions[0], config)
        if np.any(first_mask):
            first_mae = float(np.mean(np.abs(first_diff[first_mask])))
        else:
            first_mae = 0.0
        best_mae = first_mae
        for sample_index in range(1, n_samples):
            diff = tile_predictions[sample_index] - tile_reference
            mask = active_pixel_mask(target, tile_reference, tile_predictions[sample_index], config)
            sample_mae = float(np.mean(np.abs(diff[mask]))) if np.any(mask) else 0.0
            best_mae = min(best_mae, sample_mae)
        rows.append(
            {
                "div_pairwise_mae": pairwise,
                "div_pixel_std": pixel_std,
                "div_tile_mean_sd": float(np.std(tile_means)),
                "div_tile_heterogeneity_sd": float(np.std(tile_stds)),
                "first_sample_tile_mae": first_mae,
                "best_of_k_tile_mae": best_mae,
                "oracle8_tile_mae_gain": first_mae - best_mae,
            }
        )
    return rows


def save_representative_sample_bundle(
    results_dirs: dict[str, Path],
    row_id: str,
    filenames: list[str],
    reference: np.ndarray,
    predictions: np.ndarray,
) -> None:
    tile_index = min(len(filenames) // 2, len(filenames) - 1)
    if tile_index < 0:
        return
    path = results_dirs["samples"] / f"{row_id.replace('::', '__')}.npz"
    np.savez_compressed(
        path,
        filename=np.array(filenames[tile_index]),
        reference=reference[tile_index],
        predictions=predictions[:, tile_index],
    )


def build_zero_diversity_rows_from_point_rows(
    *,
    group: GroupSpec,
    point_tile_rows: pd.DataFrame,
    subset_filenames: list[str],
    target: str,
    evaluation_mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    target_label = target.lower()
    row_id = make_row_id(DIVERSITY_STAGE, group, target, evaluation_mode)
    subset = point_tile_rows[
        (point_tile_rows["group_id"] == group.group_id)
        & (point_tile_rows["target"] == target_label)
        & (point_tile_rows["evaluation_mode"] == evaluation_mode)
        & (point_tile_rows["filename"].isin(subset_filenames))
    ].copy()
    if subset.empty:
        return [], []
    subset = subset.sort_values("filename")
    row_common = {
        "group_id": group.group_id,
        "training_regime": group.training_regime,
        "family": group.family,
        "dataset_name": group.dataset_name,
        "learning_rate": group.learning_rate,
        "checkpoint": group.checkpoint,
        "target": target_label,
        "evaluation_mode": evaluation_mode,
    }
    tile_rows = [
        {
            "row_id": row_id,
            **row_common,
            "filename": row["filename"],
            "div_pairwise_mae": 0.0,
            "div_pixel_std": 0.0,
            "div_tile_mean_sd": 0.0,
            "div_tile_heterogeneity_sd": 0.0,
            "first_sample_tile_mae": float(row["tile_pixel_mae"]),
            "best_of_k_tile_mae": float(row["tile_pixel_mae"]),
            "oracle8_tile_mae_gain": 0.0,
        }
        for _, row in subset.iterrows()
    ]
    overall_rows = [
        {
            "row_id": row_id,
            **row_common,
            "n_tiles": int(len(subset)),
            "div_pairwise_mae": 0.0,
            "div_pixel_std": 0.0,
            "div_tile_mean_sd": 0.0,
            "div_tile_heterogeneity_sd": 0.0,
            "oracle8_tile_mae_gain": 0.0,
            "spread_error_spearman": 0.0,
        }
    ]
    return overall_rows, tile_rows


def evaluate_diversity_group_bf(
    *,
    group: GroupSpec,
    dataset: DatasetBundle,
    subset_filenames: list[str],
    config: dict[str, Any],
    device: torch.device,
    representative_row_ids: set[str],
    point_tile_rows: pd.DataFrame,
    results_dirs: dict[str, Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if group.family == "1":
        return build_zero_diversity_rows_from_point_rows(
            group=group,
            point_tile_rows=point_tile_rows,
            subset_filenames=subset_filenames,
            target="BF",
            evaluation_mode=STANDARD_MODE,
        )
    row_common = {
        "group_id": group.group_id,
        "training_regime": group.training_regime,
        "family": group.family,
        "dataset_name": group.dataset_name,
        "learning_rate": group.learning_rate,
        "checkpoint": group.checkpoint,
    }
    batch_size = int(config["batch_size"])
    bf_model = load_generator_from_run(group.family, group.bf_checkpoint_path, group.bf_metadata, device)
    central = np.stack([dataset.get_tile(filename)[0] for filename in subset_filenames], axis=0)
    bf_reference = np.stack([encode_bf_unit(dataset.get_tile(filename)[1]) for filename in subset_filenames], axis=0)
    bf_condition_np = encode_condition_batch(
        central,
        condition_kind=str(group.bf_metadata["condition_kind"]),
        condition_channels=int(group.bf_metadata["condition_channels"]),
    )
    bf_predictions = [
        predict_subset_for_seed(
            model=bf_model,
            family=group.family,
            condition_np=bf_condition_np,
            filenames=subset_filenames,
            target_kind="bf",
            latent_seed=int(latent_seed),
            batch_size=batch_size,
            device=device,
        )
        for latent_seed in config["diversity_latent_seeds"]
    ]
    bf_predictions_np = np.stack(bf_predictions, axis=0)
    tile_rows: list[dict[str, Any]] = []
    overall_rows = build_diversity_output_rows(
        group=group,
        target="BF",
        evaluation_mode=STANDARD_MODE,
        filenames=subset_filenames,
        reference=bf_reference,
        predictions=bf_predictions_np,
        config=config,
        row_common=row_common,
        out_tile_rows=tile_rows,
    )
    bf_row_id = make_row_id(DIVERSITY_STAGE, group, "BF", STANDARD_MODE)
    if bf_row_id in representative_row_ids:
        save_representative_sample_bundle(
            results_dirs=results_dirs,
            row_id=bf_row_id,
            filenames=subset_filenames,
            reference=bf_reference,
            predictions=bf_predictions_np,
        )
    del bf_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return overall_rows, tile_rows


def evaluate_diversity_group_bh(
    *,
    group: GroupSpec,
    dataset: DatasetBundle,
    subset_filenames: list[str],
    config: dict[str, Any],
    device: torch.device,
    representative_row_ids: set[str],
    point_tile_rows: pd.DataFrame,
    results_dirs: dict[str, Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if group.family == "1":
        overall_rows: list[dict[str, Any]] = []
        tile_rows: list[dict[str, Any]] = []
        for evaluation_mode in (ORACLE_MODE, PIPELINE_MODE):
            mode_rows, mode_tile_rows = build_zero_diversity_rows_from_point_rows(
                group=group,
                point_tile_rows=point_tile_rows,
                subset_filenames=subset_filenames,
                target="BH",
                evaluation_mode=evaluation_mode,
            )
            overall_rows.extend(mode_rows)
            tile_rows.extend(mode_tile_rows)
        return overall_rows, tile_rows
    row_common = {
        "group_id": group.group_id,
        "training_regime": group.training_regime,
        "family": group.family,
        "dataset_name": group.dataset_name,
        "learning_rate": group.learning_rate,
        "checkpoint": group.checkpoint,
    }
    batch_size = int(config["batch_size"])
    bf_model = load_generator_from_run(group.family, group.bf_checkpoint_path, group.bf_metadata, device)
    bh_model = load_generator_from_run(group.family, group.bh_checkpoint_path, group.bh_metadata, device)
    central = np.stack([dataset.get_tile(filename)[0] for filename in subset_filenames], axis=0)
    bf_reference = np.stack([encode_bf_unit(dataset.get_tile(filename)[1]) for filename in subset_filenames], axis=0)
    bh_reference = np.stack([height_feet_to_meters(dataset.get_tile(filename)[2]) for filename in subset_filenames], axis=0)
    bh_oracle_condition_np = encode_condition_batch(
        bf_reference,
        condition_kind=str(group.bh_metadata["condition_kind"]),
        condition_channels=int(group.bh_metadata["condition_channels"]),
    )
    bf_condition_np = encode_condition_batch(
        central,
        condition_kind=str(group.bf_metadata["condition_kind"]),
        condition_channels=int(group.bf_metadata["condition_channels"]),
    )
    bf_predictions = [
        predict_subset_for_seed(
            model=bf_model,
            family=group.family,
            condition_np=bf_condition_np,
            filenames=subset_filenames,
            target_kind="bf",
            latent_seed=int(latent_seed),
            batch_size=batch_size,
            device=device,
        )
        for latent_seed in config["diversity_latent_seeds"]
    ]
    bf_predictions_np = np.stack(bf_predictions, axis=0)
    bh_oracle_predictions = [
        predict_subset_for_seed(
            model=bh_model,
            family=group.family,
            condition_np=bh_oracle_condition_np,
            filenames=subset_filenames,
            target_kind="height",
            latent_seed=int(latent_seed),
            batch_size=batch_size,
            device=device,
        )
        for latent_seed in config["diversity_latent_seeds"]
    ]
    bh_oracle_predictions_np = np.stack(bh_oracle_predictions, axis=0)
    tile_rows: list[dict[str, Any]] = []
    overall_rows = build_diversity_output_rows(
        group=group,
        target="BH",
        evaluation_mode=ORACLE_MODE,
        filenames=subset_filenames,
        reference=bh_reference,
        predictions=bh_oracle_predictions_np,
        config=config,
        row_common=row_common,
        out_tile_rows=tile_rows,
    )
    bh_oracle_row_id = make_row_id(DIVERSITY_STAGE, group, "BH", ORACLE_MODE)
    if bh_oracle_row_id in representative_row_ids:
        save_representative_sample_bundle(
            results_dirs=results_dirs,
            row_id=bh_oracle_row_id,
            filenames=subset_filenames,
            reference=bh_reference,
            predictions=bh_oracle_predictions_np,
        )
    bh_pipeline_predictions = []
    for seed_index, latent_seed in enumerate(config["diversity_latent_seeds"]):
        bh_pipeline_condition_np = encode_condition_batch(
            bf_predictions_np[seed_index],
            condition_kind=str(group.bh_metadata["condition_kind"]),
            condition_channels=int(group.bh_metadata["condition_channels"]),
        )
        bh_pipeline_predictions.append(
            predict_subset_for_seed(
                model=bh_model,
                family=group.family,
                condition_np=bh_pipeline_condition_np,
                filenames=subset_filenames,
                target_kind="height",
                latent_seed=int(latent_seed),
                batch_size=batch_size,
                device=device,
            )
        )
    bh_pipeline_predictions_np = np.stack(bh_pipeline_predictions, axis=0)
    overall_rows.extend(
        build_diversity_output_rows(
            group=group,
            target="BH",
            evaluation_mode=PIPELINE_MODE,
            filenames=subset_filenames,
            reference=bh_reference,
            predictions=bh_pipeline_predictions_np,
            config=config,
            row_common=row_common,
            out_tile_rows=tile_rows,
        )
    )
    bh_pipeline_row_id = make_row_id(DIVERSITY_STAGE, group, "BH", PIPELINE_MODE)
    if bh_pipeline_row_id in representative_row_ids:
        save_representative_sample_bundle(
            results_dirs=results_dirs,
            row_id=bh_pipeline_row_id,
            filenames=subset_filenames,
            reference=bh_reference,
            predictions=bh_pipeline_predictions_np,
        )
    del bf_model
    del bh_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return overall_rows, tile_rows


def build_diversity_output_rows(
    *,
    group: GroupSpec,
    target: str,
    evaluation_mode: str,
    filenames: list[str],
    reference: np.ndarray,
    predictions: np.ndarray,
    config: dict[str, Any],
    row_common: dict[str, Any],
    out_tile_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    row_id = make_row_id(DIVERSITY_STAGE, group, target, evaluation_mode)
    tile_metrics = compute_diversity_tile_metrics(
        predictions=predictions,
        reference=reference,
        target=target,
        config=config,
    )
    spread = np.array([row["div_pairwise_mae"] for row in tile_metrics], dtype=np.float64)
    error = np.array([row["first_sample_tile_mae"] for row in tile_metrics], dtype=np.float64)
    if spread.size >= 2 and not np.allclose(spread, spread[0]) and not np.allclose(error, error[0]):
        spread_error = float(scipy.stats.spearmanr(spread, error).statistic)
    else:
        spread_error = math.nan
    for filename, metrics in zip(filenames, tile_metrics, strict=True):
        out_tile_rows.append(
            {
                "row_id": row_id,
                **row_common,
                "target": target.lower(),
                "evaluation_mode": evaluation_mode,
                "filename": filename,
                **metrics,
            }
        )
    return [
        {
            "row_id": row_id,
            **row_common,
            "target": target.lower(),
            "evaluation_mode": evaluation_mode,
            "n_tiles": len(tile_metrics),
            "div_pairwise_mae": float(np.mean([row["div_pairwise_mae"] for row in tile_metrics])),
            "div_pixel_std": float(np.mean([row["div_pixel_std"] for row in tile_metrics])),
            "div_tile_mean_sd": float(np.mean([row["div_tile_mean_sd"] for row in tile_metrics])),
            "div_tile_heterogeneity_sd": float(np.mean([row["div_tile_heterogeneity_sd"] for row in tile_metrics])),
            "oracle8_tile_mae_gain": float(np.mean([row["oracle8_tile_mae_gain"] for row in tile_metrics])),
            "spread_error_spearman": spread_error,
        }
    ]


def requested_stage_pairs(stage: str) -> list[tuple[str, str]]:
    if stage == POINT_STAGE:
        return [("BF", STANDARD_MODE), ("BH", ORACLE_MODE), ("BH", PIPELINE_MODE)]
    if stage == DIVERSITY_STAGE:
        return [("BF", STANDARD_MODE), ("BH", ORACLE_MODE), ("BH", PIPELINE_MODE)]
    raise ValueError(f"Unsupported stage: {stage}")


def row_matches_output_filters(row: dict[str, Any], filters: dict[str, set[str]]) -> bool:
    if filters.get("target") and str(row["target"]).upper() not in filters["target"]:
        return False
    if filters.get("evaluation_mode") and str(row["evaluation_mode"]) not in filters["evaluation_mode"]:
        return False
    return True


def requested_row_ids_for_group(stage: str, group: GroupSpec, filters: dict[str, set[str]]) -> set[str]:
    row_ids: set[str] = set()
    for target, evaluation_mode in requested_stage_pairs(stage):
        if not row_matches_output_filters({"target": target, "evaluation_mode": evaluation_mode}, filters):
            continue
        row_ids.add(make_row_id(stage, group, target, evaluation_mode))
    return row_ids


def build_pixel_level_summary_rows(overall_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pixel_columns = [
        "row_id",
        "group_id",
        "target",
        "evaluation_mode",
        "training_regime",
        "family",
        "dataset_name",
        "learning_rate",
        "checkpoint",
        "n_tiles",
        "pixel_mae",
        "pixel_rmse",
        "pixel_mbe",
        "pixel_mae_ci_low",
        "pixel_mae_ci_high",
        "pixel_rmse_ci_low",
        "pixel_rmse_ci_high",
        "pixel_mbe_ci_low",
        "pixel_mbe_ci_high",
        "pixel_active_count",
        "pixel_sample_size",
        "pixel_regression_slope",
        "pixel_regression_intercept",
        "pixel_regression_r2",
        "pixel_regression_pearson_r",
    ]
    return [{column: row.get(column) for column in pixel_columns} for row in overall_rows]


def drop_stage_log_rows(results_dirs: dict[str, Path], stage: str) -> None:
    path = results_dirs["logs"] / "completed_rows.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path)
    if "stage" not in frame.columns:
        path.unlink()
        return
    frame = frame[frame["stage"] != stage]
    if frame.empty:
        path.unlink()
    else:
        frame.to_csv(path, index=False)


def reset_stage_outputs(results_dirs: dict[str, Path], stage: str) -> None:
    if stage == POINT_STAGE:
        paths = [
            results_dirs["metrics"] / "overall_point_metrics.csv",
            results_dirs["metrics"] / "tile_level_mean_metrics.csv",
            results_dirs["metrics"] / "tile_level_heterogeneity_metrics.csv",
            results_dirs["metrics"] / "pixel_level_metrics.csv",
            results_dirs["metrics"] / "bh_pipeline_vs_oracle.csv",
            results_dirs["metrics"] / "bias_by_reference_bin.csv",
        ]
    elif stage == DIVERSITY_STAGE:
        paths = [
            results_dirs["diversity"] / "overall_diversity_metrics.csv",
            results_dirs["diversity"] / "tile_level_diversity_metrics.csv",
        ]
        for sample_path in results_dirs["samples"].glob("*.npz"):
            sample_path.unlink()
    else:
        raise ValueError(f"Unsupported stage: {stage}")
    for path in paths:
        if path.exists():
            path.unlink()
    drop_stage_log_rows(results_dirs, stage)


def reset_summary_outputs(results_dirs: dict[str, Path]) -> None:
    for folder in (results_dirs["tables"], results_dirs["figures"]):
        for path in folder.iterdir():
            if path.is_file():
                path.unlink()


def filter_groups(groups: list[GroupSpec], filters: dict[str, set[str]]) -> list[GroupSpec]:
    if not filters:
        return groups
    filtered: list[GroupSpec] = []
    for group in groups:
        if filters.get("training_regime") and group.training_regime not in filters["training_regime"]:
            continue
        if filters.get("family") and group.family not in filters["family"]:
            continue
        if filters.get("dataset_name") and group.dataset_name not in filters["dataset_name"]:
            continue
        if filters.get("learning_rate") and f"{group.learning_rate:g}" not in filters["learning_rate"]:
            continue
        if filters.get("checkpoint") and str(group.checkpoint) not in filters["checkpoint"]:
            continue
        filtered.append(group)
    return filtered


def run_point_stage(
    *,
    config: dict[str, Any],
    results_dirs: dict[str, Path],
    groups: list[GroupSpec],
    quick_tiles: int | None,
    resume: bool,
    filters: dict[str, set[str]],
) -> None:
    set_active_config(config)
    device = choose_device(str(config["device"]))
    set_torch_determinism()
    datasets = {
        name: DatasetBundle(spec, cache_mode=str(config["cache_mode"]))
        for name, spec in config["dataset_specs"].items()
    }
    filenames_by_dataset = {
        name: bundle.limited_filenames(quick_tiles)
        for name, bundle in datasets.items()
    }
    n_tiles = len(next(iter(filenames_by_dataset.values())))
    bootstrap_indices = bootstrap_indices_array(results_dirs, n_tiles, config)
    pixel_sample_lookup = write_pixel_regression_sample(results_dirs, datasets, filenames_by_dataset, config)
    filtered_groups = filter_groups(groups, filters)
    overall_path = results_dirs["metrics"] / "overall_point_metrics.csv"
    completed = load_existing_keys(overall_path, ["row_id"]) if resume else set()
    for group in filtered_groups:
        expected_ids = requested_row_ids_for_group(POINT_STAGE, group, filters)
        if not expected_ids:
            continue
        if resume and expected_ids.issubset(completed):
            continue
        filenames = filenames_by_dataset[group.dataset_name]
        overall_rows, tile_mean_rows, tile_heter_rows, bias_rows, bh_df = evaluate_point_group(
            group,
            datasets[group.dataset_name],
            filenames,
            config,
            device,
            bootstrap_indices,
            pixel_sample_lookup,
        )
        overall_rows = [row for row in overall_rows if row_matches_output_filters(row, filters)]
        tile_mean_rows = [row for row in tile_mean_rows if row_matches_output_filters(row, filters)]
        tile_heter_rows = [row for row in tile_heter_rows if row_matches_output_filters(row, filters)]
        bias_rows = [row for row in bias_rows if row_matches_output_filters(row, filters)]
        if overall_rows:
            append_csv_rows(results_dirs["metrics"] / "overall_point_metrics.csv", overall_rows)
            append_csv_rows(results_dirs["metrics"] / "pixel_level_metrics.csv", build_pixel_level_summary_rows(overall_rows))
        if tile_mean_rows:
            append_csv_rows(results_dirs["metrics"] / "tile_level_mean_metrics.csv", tile_mean_rows)
        if tile_heter_rows:
            append_csv_rows(results_dirs["metrics"] / "tile_level_heterogeneity_metrics.csv", tile_heter_rows)
        if bias_rows:
            append_csv_rows(results_dirs["metrics"] / "bias_by_reference_bin.csv", bias_rows)
        if (
            not bh_df.empty
            and "BH" in {target for target, _mode in requested_stage_pairs(POINT_STAGE)}
            and (not filters.get("target") or "BH" in filters["target"])
            and (
                not filters.get("evaluation_mode")
                or {ORACLE_MODE, PIPELINE_MODE}.issubset(filters["evaluation_mode"])
            )
        ):
            write_header = not (results_dirs["metrics"] / "bh_pipeline_vs_oracle.csv").exists()
            bh_df.to_csv(
                results_dirs["metrics"] / "bh_pipeline_vs_oracle.csv",
                mode="a",
                header=write_header,
                index=False,
            )
        if overall_rows:
            append_log_rows(
                results_dirs,
                [
                    {
                        "timestamp_utc": now_utc_iso(),
                        "stage": POINT_STAGE,
                        "row_id": row["row_id"],
                        "group_id": group.group_id,
                        "status": "completed",
                    }
                    for row in overall_rows
                ],
            )


def run_diversity_stage(
    *,
    config: dict[str, Any],
    results_dirs: dict[str, Path],
    groups: list[GroupSpec],
    quick_tiles: int | None,
    resume: bool,
    filters: dict[str, set[str]],
) -> None:
    set_active_config(config)
    device = choose_device(str(config["device"]))
    set_torch_determinism()
    datasets = {
        name: DatasetBundle(spec, cache_mode=str(config["cache_mode"]))
        for name, spec in config["dataset_specs"].items()
    }
    filenames_by_dataset = {
        name: bundle.limited_filenames(quick_tiles)
        for name, bundle in datasets.items()
    }
    subsets = write_diversity_subsets(results_dirs, datasets, filenames_by_dataset, config)
    point_path = results_dirs["metrics"] / "overall_point_metrics.csv"
    point_tile_path = results_dirs["metrics"] / "tile_level_mean_metrics.csv"
    if not point_path.exists() or not point_tile_path.exists():
        raise FileNotFoundError(
            "Point-stage outputs are required before diversity. Run the `point` stage first."
        )
    point_df = pd.read_csv(point_path)
    point_tile_rows = load_point_tile_rows(results_dirs)
    representative_row_ids = select_representative_sample_rows(point_df, config)
    filtered_groups = filter_groups(groups, filters)
    overall_path = results_dirs["diversity"] / "overall_diversity_metrics.csv"
    completed = load_existing_keys(overall_path, ["row_id"]) if resume else set()
    for group in filtered_groups:
        expected_ids = requested_row_ids_for_group(DIVERSITY_STAGE, group, filters)
        if not expected_ids:
            continue
        if resume and expected_ids.issubset(completed):
            continue
        group_rows: list[dict[str, Any]] = []
        group_tile_rows: list[dict[str, Any]] = []
        subset_filenames_bf = subsets[(group.dataset_name, "BF")]
        subset_filenames_bh = subsets[(group.dataset_name, "BH")]
        bf_rows, bf_tile_rows = evaluate_diversity_group_bf(
            group=group,
            dataset=datasets[group.dataset_name],
            subset_filenames=subset_filenames_bf,
            config=config,
            device=device,
            representative_row_ids=representative_row_ids,
            point_tile_rows=point_tile_rows,
            results_dirs=results_dirs,
        )
        bh_rows, bh_tile_rows = evaluate_diversity_group_bh(
            group=group,
            dataset=datasets[group.dataset_name],
            subset_filenames=subset_filenames_bh,
            config=config,
            device=device,
            representative_row_ids=representative_row_ids,
            point_tile_rows=point_tile_rows,
            results_dirs=results_dirs,
        )
        group_rows.extend(bf_rows)
        group_rows.extend(bh_rows)
        group_tile_rows.extend(bf_tile_rows)
        group_tile_rows.extend(bh_tile_rows)
        group_rows = [row for row in group_rows if row_matches_output_filters(row, filters)]
        group_tile_rows = [row for row in group_tile_rows if row_matches_output_filters(row, filters)]
        if group_rows:
            append_csv_rows(results_dirs["diversity"] / "overall_diversity_metrics.csv", group_rows)
        if group_tile_rows:
            append_csv_rows(results_dirs["diversity"] / "tile_level_diversity_metrics.csv", group_tile_rows)
        if group_rows:
            append_log_rows(
                results_dirs,
                [
                    {
                        "timestamp_utc": now_utc_iso(),
                        "stage": DIVERSITY_STAGE,
                        "row_id": row["row_id"],
                        "group_id": group.group_id,
                        "status": "completed",
                    }
                    for row in group_rows
                ],
            )


def parse_filter_values(value: str | None) -> set[str]:
    if value is None or not value.strip():
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def build_filters(args: Any) -> dict[str, set[str]]:
    filters = {
        "training_regime": parse_filter_values(getattr(args, "training_regime", None)),
        "family": parse_filter_values(getattr(args, "family", None)),
        "dataset_name": parse_filter_values(getattr(args, "dataset", None)),
        "learning_rate": parse_filter_values(getattr(args, "learning_rate", None)),
        "checkpoint": parse_filter_values(getattr(args, "checkpoint", None)),
        "target": {item.upper() for item in parse_filter_values(getattr(args, "target", None))},
        "evaluation_mode": parse_filter_values(getattr(args, "evaluation_mode", None)),
    }
    return {key: value for key, value in filters.items() if value}


def summarise_group_ranges(frame: pd.DataFrame, value_columns: list[str], group_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, subset in frame.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, keys if isinstance(keys, tuple) else (keys,), strict=False))
        for column in value_columns:
            values = subset[column].dropna().to_numpy(dtype=float)
            if values.size == 0:
                row[f"{column}_median"] = math.nan
                row[f"{column}_min"] = math.nan
                row[f"{column}_max"] = math.nan
            else:
                row[f"{column}_median"] = float(np.median(values))
                row[f"{column}_min"] = float(np.min(values))
                row[f"{column}_max"] = float(np.max(values))
        rows.append(row)
    return pd.DataFrame(rows)


def save_heatmap(
    *,
    frame: pd.DataFrame,
    value_column: str,
    title: str,
    out_path: Path,
) -> None:
    learning_rates = [0.0001, 0.0002, 0.0005, 0.001]
    checkpoints = [250, 500, 750, 1000]
    pivot = frame.pivot(index="learning_rate", columns="checkpoint", values=value_column)
    pivot = pivot.reindex(index=learning_rates, columns=checkpoints)
    fig, ax = plt.subplots(figsize=(5.5, 3.8), constrained_layout=True)
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(checkpoints)))
    ax.set_xticklabels([str(item) for item in checkpoints])
    ax.set_yticks(np.arange(len(learning_rates)))
    ax.set_yticklabels([f"{item:g}" for item in learning_rates])
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel("Learning rate")
    ax.set_title(title)
    for row_index in range(pivot.shape[0]):
        for col_index in range(pivot.shape[1]):
            value = pivot.iloc[row_index, col_index]
            if pd.notna(value):
                ax.text(col_index, row_index, f"{value:.3f}", ha="center", va="center", fontsize=7, color="white")
    fig.colorbar(image, ax=ax, shrink=0.9)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def build_training_regime_delta_table(point_df: pd.DataFrame) -> pd.DataFrame:
    pair_columns = ["target", "evaluation_mode", "family", "dataset_name", "learning_rate", "checkpoint"]
    metric_columns = [
        "tile_mean_mae",
        "tile_mean_rmse",
        "tile_mean_mbe",
        "tile_mean_slope",
        "tile_mean_r2",
        "pixel_mae",
        "pixel_mbe",
        "pixel_regression_slope",
        "pixel_regression_r2",
    ]
    paired = point_df[pair_columns + ["training_regime", *metric_columns]].copy()
    paired = paired[paired["training_regime"].isin(["LALegacy", "MSASample"])]
    wide = paired.pivot_table(
        index=pair_columns,
        columns="training_regime",
        values=metric_columns,
        aggfunc="first",
    )
    if wide.empty or "LALegacy" not in wide.columns.get_level_values(1) or "MSASample" not in wide.columns.get_level_values(1):
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for keys, values in wide.iterrows():
        row = dict(zip(pair_columns, keys if isinstance(keys, tuple) else (keys,), strict=False))
        for metric in metric_columns:
            la_value = values.get((metric, "LALegacy"), math.nan)
            msa_value = values.get((metric, "MSASample"), math.nan)
            row[f"{metric}_LALegacy"] = la_value
            row[f"{metric}_MSASample"] = msa_value
            row[f"{metric}_delta_msa_minus_la"] = msa_value - la_value
        row["abs_tile_mean_mbe_LALegacy"] = abs(row["tile_mean_mbe_LALegacy"])
        row["abs_tile_mean_mbe_MSASample"] = abs(row["tile_mean_mbe_MSASample"])
        row["abs_tile_mean_mbe_delta_msa_minus_la"] = (
            row["abs_tile_mean_mbe_MSASample"] - row["abs_tile_mean_mbe_LALegacy"]
        )
        row["abs_pixel_mbe_LALegacy"] = abs(row["pixel_mbe_LALegacy"])
        row["abs_pixel_mbe_MSASample"] = abs(row["pixel_mbe_MSASample"])
        row["abs_pixel_mbe_delta_msa_minus_la"] = row["abs_pixel_mbe_MSASample"] - row["abs_pixel_mbe_LALegacy"]
        rows.append(row)
    return pd.DataFrame(rows)


def save_training_regime_delta_figure(delta_df: pd.DataFrame, out_path: Path) -> None:
    if delta_df.empty:
        return
    delta_df = delta_df[delta_df["checkpoint"] == 1000].copy()
    if delta_df.empty:
        return
    panel_specs = [
        ("bf", "standard", "BF"),
        ("bh", "oracle_bf", "BH"),
    ]
    dataset_order = ["CONUSStratifiedTest", "SanDiegoTestNoOverlap"]
    family_order = ["1", "2A", "3"]
    colors = {"CONUSStratifiedTest": "#315f72", "SanDiegoTestNoOverlap": "#c87533"}
    metric_specs = [
        ("tile_mean_mae_delta_msa_minus_la", "Delta tile mean MAE\n(MSA - LA)", "negative = lower error"),
        ("tile_mean_mbe_delta_msa_minus_la", "Delta tile mean MBE\n(MSA - LA)", "positive = less underprediction"),
        ("pixel_mae_delta_msa_minus_la", "Delta pixel MAE\n(MSA - LA)", "negative = lower error"),
        ("pixel_mbe_delta_msa_minus_la", "Delta pixel MBE\n(MSA - LA)", "positive = less underprediction"),
    ]
    fig, axes = plt.subplots(len(metric_specs), len(panel_specs), figsize=(9.6, 11.0), constrained_layout=False)
    for row_index, (metric, ylabel, subtitle) in enumerate(metric_specs):
        for col_index, (target, evaluation_mode, panel_title) in enumerate(panel_specs):
            ax = axes[row_index, col_index]
            subset = delta_df[
                (delta_df["target"] == target)
                & (delta_df["evaluation_mode"] == evaluation_mode)
            ]
            positions: list[float] = []
            labels: list[str] = []
            data: list[np.ndarray] = []
            box_colors: list[str] = []
            position = 1.0
            for family in family_order:
                for dataset_name in dataset_order:
                    values = subset[
                        (subset["family"] == family)
                        & (subset["dataset_name"] == dataset_name)
                    ][metric].dropna().to_numpy(dtype=float)
                    if values.size == 0:
                        continue
                    positions.append(position)
                    labels.append(f"{family}\n{'CONUS' if dataset_name == 'CONUSStratifiedTest' else 'SD'}")
                    data.append(values)
                    box_colors.append(colors[dataset_name])
                    position += 1.0
                position += 0.45
            if data:
                box = ax.boxplot(data, positions=positions, widths=0.65, patch_artist=True, showfliers=False)
                for patch, color in zip(box["boxes"], box_colors, strict=False):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.72)
                for median in box["medians"]:
                    median.set_color("black")
                    median.set_linewidth(1.2)
            ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
            ax.set_title(panel_title if row_index == 0 else "")
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, fontsize=8)
            if col_index == 0:
                ax.set_ylabel(ylabel)
            ax.text(
                0.02,
                0.96,
                subtitle,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.5},
            )
            ax.grid(axis="y", alpha=0.25)
    handles = [
        plt.Line2D([0], [0], color=colors["CONUSStratifiedTest"], linewidth=8, alpha=0.72, label="CONUS stratified test"),
        plt.Line2D([0], [0], color=colors["SanDiegoTestNoOverlap"], linewidth=8, alpha=0.72, label="San Diego no-overlap test"),
    ]
    fig.subplots_adjust(left=0.105, right=0.995, top=0.91, bottom=0.065, hspace=0.42, wspace=0.16)
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("Paired training-regime comparison at epoch 1000 across learning rates", y=0.985)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def prepare_epoch1000_model_class_tile_frame(tile_df: pd.DataFrame) -> pd.DataFrame:
    frame = tile_df[
        (tile_df["checkpoint"] == 1000)
        & (
            ((tile_df["target"] == "bf") & (tile_df["evaluation_mode"] == STANDARD_MODE))
            | ((tile_df["target"] == "bh") & (tile_df["evaluation_mode"] == ORACLE_MODE))
        )
    ].copy()
    frame["filename"] = frame["filename"].astype(str).str.strip()
    frame["family"] = frame["family"].astype(str)
    frame["tile_mean_abs_error"] = (frame["prediction_value"] - frame["reference_value"]).abs()
    frame["tile_mean_bias"] = frame["prediction_value"] - frame["reference_value"]
    frame["target_label"] = frame["target"].map({"bf": "BF", "bh": "BH"})
    return frame


def build_model_class_metric_summary_epoch1000(mean_tile_df: pd.DataFrame, heterogeneity_tile_df: pd.DataFrame) -> pd.DataFrame:
    mean_frame = prepare_epoch1000_model_class_tile_frame(mean_tile_df)
    heter_frame = prepare_epoch1000_model_class_tile_frame(heterogeneity_tile_df)
    rows: list[dict[str, Any]] = []
    group_columns = ["training_regime", "target", "target_label", "dataset_name", "family", "learning_rate"]
    heter_groups = {
        keys: subset
        for keys, subset in heter_frame.groupby(group_columns, dropna=False)
    }
    for keys, subset in mean_frame.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, keys if isinstance(keys, tuple) else (keys,), strict=False))
        heter_subset = heter_groups.get(keys, pd.DataFrame())
        row["checkpoint"] = 1000
        row["n_tiles"] = int(subset.shape[0])
        row["average_absolute_error"] = float(subset["tile_mean_abs_error"].mean())
        row["average_bias"] = float(subset["tile_mean_bias"].mean())
        row["heterogeneity_absolute_error"] = (
            float(heter_subset["tile_mean_abs_error"].mean()) if not heter_subset.empty else math.nan
        )
        row["heterogeneity_bias"] = (
            float(heter_subset["tile_mean_bias"].mean()) if not heter_subset.empty else math.nan
        )
        rows.append(row)
    return pd.DataFrame(rows)


def model_class_plot_constants() -> tuple[
    list[tuple[str, str, str]],
    list[str],
    dict[str, str],
    dict[str, str],
    dict[str, str],
    dict[float, str],
    dict[str, float],
]:
    panel_specs = [
        ("LALegacy", "BF", "Building Footprint | Los Angeles Model"),
        ("MSASample", "BF", "Building Footprint | CONUS Model"),
        ("LALegacy", "BH", "Building Heights (m) | Los Angeles Model"),
        ("MSASample", "BH", "Building Heights (m) | CONUS Model"),
    ]
    family_order = ["1", "2A", "3"]
    family_labels = {"1": "U-Net", "2A": "Single-latent\ncGAN", "3": "Multi-latent\ncGAN"}
    dataset_colors = {"CONUSStratifiedTest": "#315f72", "SanDiegoTestNoOverlap": "#c87533"}
    dataset_labels = {"CONUSStratifiedTest": "CONUS Test Dataset", "SanDiegoTestNoOverlap": "San Diego Test Dataset"}
    lr_markers = {0.0001: "o", 0.0002: "s", 0.0005: "^", 0.001: "D"}
    offsets = {"CONUSStratifiedTest": -0.16, "SanDiegoTestNoOverlap": 0.16}
    return panel_specs, family_order, family_labels, dataset_colors, dataset_labels, lr_markers, offsets


def save_model_class_grid_figure(
    summary_df: pd.DataFrame,
    *,
    metric_specs: list[tuple[str, str]],
    group_labels: tuple[str, str],
    group_label_rows: tuple[tuple[int, int], tuple[int, int]],
    out_path: Path,
    zero_line_metric_tokens: tuple[str, ...] = (),
    one_line_metric_tokens: tuple[str, ...] = (),
) -> None:
    if summary_df.empty:
        return
    (
        panel_specs,
        family_order,
        family_labels,
        dataset_colors,
        dataset_labels,
        lr_markers,
        offsets,
    ) = model_class_plot_constants()
    y_limits: dict[tuple[str, str], tuple[float, float]] = {}
    for metric, _ylabel in metric_specs:
        for target_label in ("BF", "BH"):
            values = summary_df[summary_df["target_label"] == target_label][metric].dropna().to_numpy(dtype=float)
            if values.size == 0:
                continue
            lower = float(np.min(values))
            upper = float(np.max(values))
            if any(token in metric for token in zero_line_metric_tokens):
                lower = min(lower, 0.0)
                upper = max(upper, 0.0)
            if any(token in metric for token in one_line_metric_tokens):
                lower = min(lower, 1.0)
                upper = max(upper, 1.0)
            span = upper - lower
            pad = 0.06 * span if span > 0.0 else max(abs(upper), 1.0) * 0.06
            y_limits[(target_label, metric)] = (lower - pad, upper + pad)
    fig, axes = plt.subplots(len(metric_specs), len(panel_specs), figsize=(16.5, 11.0), constrained_layout=False)
    for row_index, (metric, ylabel) in enumerate(metric_specs):
        for col_index, (training_regime, target_label, title) in enumerate(panel_specs):
            ax = axes[row_index, col_index]
            panel = summary_df[
                (summary_df["training_regime"] == training_regime)
                & (summary_df["target_label"] == target_label)
            ]
            for dataset_name, dataset_panel in panel.groupby("dataset_name"):
                color = dataset_colors.get(dataset_name, "#666666")
                offset = offsets.get(dataset_name, 0.0)
                for learning_rate, lr_panel in dataset_panel.groupby("learning_rate"):
                    lr_panel = lr_panel.set_index("family").reindex(family_order)
                    x_values = np.arange(len(family_order), dtype=float) + offset
                    y_values = lr_panel[metric].to_numpy(dtype=float)
                    ax.scatter(
                        x_values,
                        y_values,
                        color=color,
                        marker=lr_markers.get(round(float(learning_rate), 4), "o"),
                        s=34,
                        edgecolor="black",
                        linewidth=0.25,
                        alpha=0.88,
                    )
                medians = dataset_panel.groupby("family")[metric].median().reindex(family_order)
                ax.scatter(
                    np.arange(len(family_order), dtype=float) + offset,
                    medians.to_numpy(dtype=float),
                    color=color,
                    marker="_",
                    s=420,
                    linewidth=2.0,
                    zorder=4,
                )
            if any(token in metric for token in zero_line_metric_tokens):
                ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
            if any(token in metric for token in one_line_metric_tokens):
                ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
            if (target_label, metric) in y_limits:
                ax.set_ylim(*y_limits[(target_label, metric)])
            ax.set_xticks(np.arange(len(family_order)))
            if row_index == len(metric_specs) - 1:
                ax.set_xticklabels([family_labels[item] for item in family_order], fontsize=10)
            else:
                ax.set_xticklabels([])
            if col_index == 0:
                ax.set_ylabel(ylabel, fontsize=12, labelpad=24)
                ax.yaxis.set_label_coords(-0.18, 0.5)
            ax.set_title(title if row_index == 0 else "", fontsize=12)
            ax.grid(axis="y", alpha=0.25)
    dataset_handles = [
        plt.Line2D([0], [0], color=color, marker="o", linestyle="", label=dataset_labels.get(dataset, dataset))
        for dataset, color in dataset_colors.items()
    ]
    lr_handles = [
        plt.Line2D([0], [0], color="black", marker=marker, linestyle="", label=f"LR {learning_rate:g}")
        for learning_rate, marker in lr_markers.items()
    ]
    median_handle = plt.Line2D([0], [0], color="black", marker="_", linestyle="", markersize=18, label="Median across LRs")
    fig.subplots_adjust(left=0.085, right=0.995, top=0.91, bottom=0.09, hspace=0.24, wspace=0.18)
    for group_label, (start_row, end_row) in zip(group_labels, group_label_rows, strict=False):
        group_y = 0.5 * (axes[start_row, 0].get_position().y1 + axes[end_row, 0].get_position().y0)
        fig.text(0.015, group_y, group_label, rotation=90, ha="center", va="center", fontsize=13, fontweight="bold")
    fig.legend(
        handles=[*dataset_handles, *lr_handles, median_handle],
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=8,
    )
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_model_class_comparison_figure(summary_df: pd.DataFrame, out_path: Path) -> None:
    save_model_class_grid_figure(
        summary_df,
        metric_specs=[
            ("average_absolute_error", "Mean Absolute Error"),
            ("average_bias", "Mean Bias"),
            ("heterogeneity_absolute_error", "Mean Absolute Error"),
            ("heterogeneity_bias", "Mean Bias"),
        ],
        group_labels=("Average", "Heterogeneity"),
        group_label_rows=((0, 1), (2, 3)),
        out_path=out_path,
        zero_line_metric_tokens=("bias",),
    )


def build_model_class_fit_summary_epoch1000(point_df: pd.DataFrame) -> pd.DataFrame:
    frame = point_df[
        (point_df["checkpoint"] == 1000)
        & (
            ((point_df["target"] == "bf") & (point_df["evaluation_mode"] == STANDARD_MODE))
            | ((point_df["target"] == "bh") & (point_df["evaluation_mode"] == ORACLE_MODE))
        )
    ].copy()
    if frame.empty:
        return frame
    frame["target_label"] = frame["target"].map({"bf": "BF", "bh": "BH"})
    return frame


def save_model_class_fit_figure(summary_df: pd.DataFrame, out_path: Path) -> None:
    save_model_class_grid_figure(
        summary_df,
        metric_specs=[
            ("tile_mean_r2", "R²"),
            ("tile_mean_slope", "Slope"),
            ("tile_heterogeneity_r2", "R²"),
            ("tile_heterogeneity_slope", "Slope"),
        ],
        group_labels=("Average", "Heterogeneity"),
        group_label_rows=((0, 1), (2, 3)),
        out_path=out_path,
        one_line_metric_tokens=("tile_mean_slope",),
    )


def build_summary_outputs(config: dict[str, Any], results_dirs: dict[str, Path]) -> None:
    point_path = results_dirs["metrics"] / "overall_point_metrics.csv"
    diversity_path = results_dirs["diversity"] / "overall_diversity_metrics.csv"
    if not point_path.exists():
        raise FileNotFoundError(f"Missing point metrics: {point_path}")
    point_df = pd.read_csv(point_path, dtype={"family": str}, low_memory=False)
    diversity_df = pd.read_csv(diversity_path, dtype={"family": str}, low_memory=False) if diversity_path.exists() else pd.DataFrame()
    bias_df = pd.read_csv(results_dirs["metrics"] / "bias_by_reference_bin.csv", dtype={"family": str}, low_memory=False) if (results_dirs["metrics"] / "bias_by_reference_bin.csv").exists() else pd.DataFrame()
    bh_df = pd.read_csv(results_dirs["metrics"] / "bh_pipeline_vs_oracle.csv", dtype={"family": str}, low_memory=False) if (results_dirs["metrics"] / "bh_pipeline_vs_oracle.csv").exists() else pd.DataFrame()
    for frame in (point_df, diversity_df, bias_df, bh_df):
        if frame.empty:
            continue
        for column in ("row_id", "group_id", "target", "evaluation_mode", "training_regime", "family", "dataset_name"):
            if column in frame.columns:
                frame[column] = frame[column].astype(str)
    point_summary = summarise_group_ranges(
        point_df,
        [
            "tile_mean_mae",
            "tile_mean_mbe",
            "tile_mean_slope",
            "tile_mean_r2",
            "tile_heterogeneity_mae",
            "tile_heterogeneity_mbe",
            "pixel_mae",
            "pixel_mbe",
            "pixel_regression_slope",
            "pixel_regression_r2",
        ],
        ["target", "evaluation_mode", "training_regime", "family", "dataset_name"],
    )
    point_summary.to_csv(results_dirs["tables"] / "table_point_metric_ranges.csv", index=False)
    regime_delta_df = build_training_regime_delta_table(point_df)
    if not regime_delta_df.empty:
        regime_delta_df.to_csv(results_dirs["tables"] / "table_training_regime_paired_deltas.csv", index=False)
    model_class_fit_summary = build_model_class_fit_summary_epoch1000(point_df)
    if not model_class_fit_summary.empty:
        model_class_fit_summary.to_csv(
            results_dirs["tables"] / "table_model_class_fit_summary_epoch1000.csv",
            index=False,
        )
        save_model_class_fit_figure(
            model_class_fit_summary,
            results_dirs["figures"] / "model_class_fit_epoch1000.png",
        )
    tile_mean_path = results_dirs["metrics"] / "tile_level_mean_metrics.csv"
    tile_heterogeneity_path = results_dirs["metrics"] / "tile_level_heterogeneity_metrics.csv"
    if tile_mean_path.exists() and tile_heterogeneity_path.exists():
        tile_usecols = [
            "target",
            "evaluation_mode",
            "training_regime",
            "family",
            "dataset_name",
            "learning_rate",
            "checkpoint",
            "filename",
            "reference_value",
            "prediction_value",
        ]
        tile_df = pd.read_csv(tile_mean_path, dtype={"family": str}, usecols=tile_usecols, low_memory=False)
        heterogeneity_tile_df = pd.read_csv(
            tile_heterogeneity_path,
            dtype={"family": str},
            usecols=tile_usecols,
            low_memory=False,
        )
        for frame in (tile_df, heterogeneity_tile_df):
            for column in ("target", "evaluation_mode", "training_regime", "family", "dataset_name", "filename"):
                frame[column] = frame[column].astype(str)
        model_class_summary = build_model_class_metric_summary_epoch1000(tile_df, heterogeneity_tile_df)
        if not model_class_summary.empty:
            model_class_summary.to_csv(
                results_dirs["tables"] / "table_model_class_metric_summary_epoch1000.csv",
                index=False,
            )
            save_model_class_comparison_figure(
                model_class_summary,
                results_dirs["figures"] / "model_class_comparison_epoch1000.png",
            )
    if not diversity_df.empty:
        diversity_summary = summarise_group_ranges(
            diversity_df,
            ["div_pairwise_mae", "div_pixel_std", "oracle8_tile_mae_gain", "spread_error_spearman"],
            ["target", "evaluation_mode", "training_regime", "family", "dataset_name"],
        )
        diversity_summary.to_csv(results_dirs["tables"] / "table_diversity_ranges.csv", index=False)
    if not bh_df.empty:
        bh_summary = summarise_group_ranges(
            bh_df,
            ["pipeline_minus_oracle_tile_mean_mae", "pipeline_minus_oracle_pixel_mae"],
            ["training_regime", "family", "dataset_name"],
        )
        bh_summary.to_csv(results_dirs["tables"] / "table_bh_pipeline_vs_oracle.csv", index=False)
    for (target, evaluation_mode, dataset_name), subset in point_df.groupby(["target", "evaluation_mode", "dataset_name"]):
        fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5), constrained_layout=True)
        for row_index, training_regime in enumerate(sorted(subset["training_regime"].unique())):
            for col_index, family in enumerate(sorted(subset["family"].unique(), key=family_sort_key)):
                axis = axes[row_index, col_index]
                panel = subset[
                    (subset["training_regime"] == training_regime)
                    & (subset["family"] == family)
                ]
                learning_rates = [0.0001, 0.0002, 0.0005, 0.001]
                checkpoints = [250, 500, 750, 1000]
                pivot = panel.pivot(index="learning_rate", columns="checkpoint", values="tile_mean_mae")
                pivot = pivot.reindex(index=learning_rates, columns=checkpoints)
                image = axis.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis")
                axis.set_xticks(np.arange(len(checkpoints)))
                axis.set_xticklabels([str(item) for item in checkpoints], fontsize=8)
                axis.set_yticks(np.arange(len(learning_rates)))
                axis.set_yticklabels([f"{item:g}" for item in learning_rates], fontsize=8)
                axis.set_title(f"{training_regime} | {family}", fontsize=10)
                for i in range(pivot.shape[0]):
                    for j in range(pivot.shape[1]):
                        value = pivot.iloc[i, j]
                        if pd.notna(value):
                            axis.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=6, color="white")
        fig.suptitle(f"{dataset_name} | {target.upper()} | {evaluation_mode} | Tile mean MAE")
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8)
        fig.savefig(results_dirs["figures"] / f"performance_matrix_{dataset_name}_{target}_{evaluation_mode}.png", dpi=220)
        plt.close(fig)
    if not bias_df.empty:
        for (target, evaluation_mode, dataset_name), subset in bias_df.groupby(["target", "evaluation_mode", "dataset_name"]):
            fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
            for (training_regime, family), panel in subset.groupby(["training_regime", "family"]):
                panel = panel.sort_values("reference_bin")
                ax.plot(
                    panel["reference_bin"],
                    panel["mean_bias"],
                    marker="o",
                    label=f"{training_regime} | {family}",
                )
            ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
            ax.set_title(f"{dataset_name} | {target.upper()} | {evaluation_mode} | Mean bias by reference bin")
            ax.set_ylabel("Mean bias")
            ax.legend(fontsize=7)
            fig.savefig(results_dirs["figures"] / f"bias_by_bin_{dataset_name}_{target}_{evaluation_mode}.png", dpi=220)
            plt.close(fig)
    if not diversity_df.empty:
        merged = point_df.merge(
            diversity_df,
            on=[
                "row_id",
                "group_id",
                "target",
                "evaluation_mode",
                "training_regime",
                "family",
                "dataset_name",
                "learning_rate",
                "checkpoint",
            ],
            how="inner",
        )
        for (target, evaluation_mode, dataset_name), subset in merged.groupby(["target", "evaluation_mode", "dataset_name"]):
            fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
            for family, marker in [("1", "s"), ("2A", "o"), ("3", "^")]:
                panel = subset[subset["family"] == family]
                if panel.empty:
                    continue
                ax.scatter(
                    panel["tile_mean_mae"],
                    panel["div_pairwise_mae"],
                    label=family,
                    marker=marker,
                    alpha=0.8,
                )
            ax.set_xlabel("Tile mean MAE")
            ax.set_ylabel("Mean pairwise diversity MAE")
            ax.set_title(f"{dataset_name} | {target.upper()} | {evaluation_mode} | Diversity-accuracy frontier")
            ax.legend(title="Family")
            fig.savefig(results_dirs["figures"] / f"diversity_accuracy_frontier_{dataset_name}_{target}_{evaluation_mode}.png", dpi=220)
            plt.close(fig)
    if not bh_df.empty:
        for dataset_name, subset in bh_df.groupby("dataset_name"):
            fig, ax = plt.subplots(figsize=(6.5, 5), constrained_layout=True)
            for (training_regime, family), panel in subset.groupby(["training_regime", "family"]):
                ax.scatter(
                    panel["oracle_tile_mean_mae"],
                    panel["pipeline_tile_mean_mae"],
                    label=f"{training_regime} | {family}",
                    alpha=0.8,
                )
            limits = [
                float(min(subset["oracle_tile_mean_mae"].min(), subset["pipeline_tile_mean_mae"].min())),
                float(max(subset["oracle_tile_mean_mae"].max(), subset["pipeline_tile_mean_mae"].max())),
            ]
            ax.plot(limits, limits, linestyle="--", color="black", linewidth=0.8)
            ax.set_xlabel("BH oracle tile mean MAE")
            ax.set_ylabel("BH pipeline tile mean MAE")
            ax.set_title(f"{dataset_name} | BH pipeline vs oracle")
            ax.legend(fontsize=7)
            fig.savefig(results_dirs["figures"] / f"bh_pipeline_vs_oracle_{dataset_name}.png", dpi=220)
            plt.close(fig)
    sample_files = sorted(results_dirs["samples"].glob("*.npz"))
    for sample_path in sample_files:
        payload = np.load(sample_path)
        reference = payload["reference"]
        predictions = payload["predictions"]
        filename = str(payload["filename"])
        n_samples = predictions.shape[0]
        cols = min(4, n_samples + 1)
        rows = int(math.ceil((n_samples + 1) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3 * rows), constrained_layout=True)
        axes_list = np.atleast_1d(axes).ravel()
        axes_list[0].imshow(reference, cmap="viridis")
        axes_list[0].set_title(f"Reference\n{filename}")
        axes_list[0].axis("off")
        for index in range(n_samples):
            axes_list[index + 1].imshow(predictions[index], cmap="viridis")
            axes_list[index + 1].set_title(f"Sample {index}")
            axes_list[index + 1].axis("off")
        for axis in axes_list[n_samples + 1:]:
            axis.axis("off")
        fig.savefig(results_dirs["figures"] / f"{sample_path.stem}.png", dpi=220)
        plt.close(fig)
