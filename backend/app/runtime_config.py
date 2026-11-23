"""Configuración operativa local de APETRIA (sin credenciales)."""
from __future__ import annotations

import json
import os
from pathlib import Path
from threading import RLock
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "instance" / "runtime_settings.json"
SOURCES = ("smd", "nasa", "nab", "hdfs", "bgl")
POLICIES = {
    "smd": ("and", "or", "dual"),
    "nasa": ("and", "or", "dual"),
    "nab": ("and", "or", "dual"),
    "hdfs": ("if", "lstm", "and", "or"),
    "bgl": ("if", "sgd", "and", "or"),
}
SourceId = Literal["smd", "nasa", "nab", "hdfs", "bgl"]


class RuntimeSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled_sources: list[SourceId] = Field(default_factory=lambda: list(SOURCES), min_length=1)
    default_dataset: SourceId = "smd"
    default_policy: str = "and"
    default_limit: int = Field(default=300, ge=30, le=10000)
    theme: Literal["dark", "light"] = "dark"

    @model_validator(mode="after")
    def validate_preferences(self):
        if len(set(self.enabled_sources)) != len(self.enabled_sources):
            raise ValueError("No se permiten fuentes duplicadas")
        if self.default_dataset not in self.enabled_sources:
            raise ValueError("El dataset predeterminado debe permanecer habilitado")
        if self.default_policy not in POLICIES[self.default_dataset]:
            raise ValueError("La política predeterminada no corresponde al dataset")
        return self


_lock = RLock()


def _read_disk() -> RuntimeSettings:
    if not CONFIG_PATH.is_file():
        return RuntimeSettings()
    # Si el archivo local estuviera corrupto, exponemos el error en vez de
    # sobrescribir silenciosamente la decisión del administrador.
    return RuntimeSettings.model_validate_json(CONFIG_PATH.read_text(encoding="utf-8"))


_current = _read_disk()


def get_settings() -> RuntimeSettings:
    with _lock:
        return _current.model_copy(deep=True)


def is_source_enabled(source: str) -> bool:
    with _lock:
        return source in _current.enabled_sources


def require_source_enabled(source: str) -> None:
    from fastapi import HTTPException
    if not is_source_enabled(source):
        raise HTTPException(status_code=403, detail=f"Fuente {source.upper()} deshabilitada por el administrador local")


def save_settings(settings: RuntimeSettings) -> RuntimeSettings:
    global _current
    validated = RuntimeSettings.model_validate(settings.model_dump())
    with _lock:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary = CONFIG_PATH.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(validated.model_dump(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(temporary, CONFIG_PATH)
        _current = validated
        return _current.model_copy(deep=True)
