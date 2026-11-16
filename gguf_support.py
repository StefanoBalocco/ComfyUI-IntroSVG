"""Local GGUF discovery and structural validation for IntroSVG."""

from dataclasses import dataclass, field
from pathlib import Path
import json

from gguf import GGUFReader

ARCHITECTURE_KEY = "introsvg.architecture"
CONFIG_KEY = "introsvg.config_json"
ASSET_PREFIX = "introsvg.embedded_file."
ARCHITECTURE = "introsvg"
ASSET_FILES = ("tokenizer.json", "tokenizer_config.json", "generation_config.json", "preprocessor_config.json")
REQUIRED_TENSORS = ("model.embed_tokens.weight", "lm_head.weight", "model.norm.weight")
REQUIRED_PREFIXES = ("model.layers.", "visual.")
MAX_ASSET_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class IntroSVGReference:
    """A validated IntroSVG GGUF: memory-mapped tensors by name, the model config and the embedded files."""

    path: Path
    tensor_index: dict = field(repr=False)
    config: dict = field(repr=False)
    assets: dict[str, str] = field(repr=False)


def read_string(reader, key: str) -> str:
    metadata = reader.get_field(key)
    if metadata is None:
        raise ValueError(f"Invalid IntroSVG GGUF: missing metadata {key}")
    contents = metadata.contents()
    if not isinstance(contents, str) or not contents:
        raise ValueError(f"Invalid IntroSVG GGUF: metadata {key} must be a non-empty string")
    return contents


def inspect_introsvg_gguf(path: Path) -> IntroSVGReference:
    path = Path(path)
    if path.suffix.lower() != ".gguf" or not path.is_file():
        raise ValueError("Expected a local .gguf file")
    # Copy-on-write gives torch.from_numpy a writable mapping whose writes stay in memory.
    reader = GGUFReader(str(path), "c")

    architecture = read_string(reader, ARCHITECTURE_KEY)
    if architecture != ARCHITECTURE:
        raise ValueError(f"{ARCHITECTURE_KEY} is {architecture!r}, expected {ARCHITECTURE!r}")
    try:
        config = json.loads(read_string(reader, CONFIG_KEY))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in GGUF {CONFIG_KEY}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"GGUF {CONFIG_KEY} must contain a JSON object")

    assets = {}
    for name in ASSET_FILES:
        contents = read_string(reader, ASSET_PREFIX + name)
        if len(contents.encode("utf-8")) > MAX_ASSET_BYTES:
            raise ValueError(f"Invalid IntroSVG GGUF: embedded {name} is too large")
        assets[name] = contents

    tensor_index = {tensor.name: tensor for tensor in reader.tensors}
    missing = [name for name in REQUIRED_TENSORS if name not in tensor_index]
    missing += [prefix + "*" for prefix in REQUIRED_PREFIXES if not any(name.startswith(prefix) for name in tensor_index)]
    if missing:
        raise ValueError("This is not a full IntroSVG GGUF: missing tensors " + ", ".join(missing))
    return IntroSVGReference(path=path, tensor_index=tensor_index, config=config, assets=assets)
