"""APETRIA 100% - API de demostración reproducible con SMD real."""
from pathlib import Path
import asyncio
import json
from dataclasses import asdict
from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect, Depends
from fastapi.staticfiles import StaticFiles


class SafeFrontendFiles(StaticFiles):
    """Evita un HTTP 500 por rutas de archivos inválidas en Windows."""

    async def get_response(self, path, scope):
        try:
            return await super().get_response(path, scope)
        except OSError as exc:
            # WinError 123: path con caracteres inválidos, p. ej. /**.
            # Se transforma solo ese caso concreto; los otros errores se reportan.
            if getattr(exc, "winerror", None) == 123:
                raise HTTPException(status_code=404, detail="Recurso no encontrado") from exc
            raise
from pydantic import BaseModel, Field
import numpy as np
from backend.app.runtime_config import RuntimeSettings, get_settings, is_source_enabled, require_source_enabled, save_settings
from backend.app.auth_store import (COOKIE_NAME, SESSION_SECONDS, authenticate, change_password, current_session, has_admin, require_csrf, require_user, revoke, ws_allowed)

ROOT = Path(__file__).resolve().parents[2]
app = FastAPI(title="APETRIA", version="1.0.0")
# El frontend y API se sirven desde el mismo origen; CORS abierto no es necesario.
DATASETS = {"smd": "operativo_si_existen_artefactos", "nasa": "pendiente_integracion", "hdfs": "pendiente_integracion", "bgl": "pendiente_integracion", "nab": "pendiente_integracion"}

class ReplayRequest(BaseModel):
    machine: str = Field(default="machine-1-1", pattern=r"^machine-[1-3]-[0-9]{1,2}$")
    partition: str = Field(default="X_test", pattern=r"^(X_test|X_val)$")
    policy: str = Field(default="and", pattern=r"^(and|or|dual)$")
    limit: int = Field(default=100, ge=1, le=10000)
    start_index: int = Field(default=0, ge=0)


def load_real_smd(machine: str, partition: str, start_index: int, limit: int, window_size: int):
    path = ROOT / "data" / "processed" / "smd" / (machine + ".npz")
    if not path.is_file():
        raise FileNotFoundError(f"Falta dataset procesado: {path}")
    with np.load(path, allow_pickle=False) as data:
        if partition not in data.files:
            raise ValueError(f"Partición {partition} ausente. Disponibles: {data.files}")
        total = len(data[partition])
        end_index = start_index + limit
        if start_index >= total or end_index > total:
            raise ValueError(f"Rango [{start_index}, {end_index}) fuera de la partición de {total} observaciones")
        warm_start = max(0, start_index - window_size + 1)
        values = np.asarray(data[partition][warm_start:end_index], dtype=np.float32).copy()
        labels = None
        if partition == "X_test" and "y_test" in data.files:
            labels = np.asarray(data["y_test"][start_index:end_index], dtype=np.int8).copy()
            if len(labels) != limit or not np.isin(labels, [0, 1]).all():
                raise ValueError("Etiquetas y_test no válidas para el segmento solicitado")
    if values.ndim != 2 or values.shape[1] != 38 or not np.isfinite(values).all():
        raise ValueError("SMD requiere matriz finita con 38 características normalizadas")
    return warm_start, values, labels


def run_replay(request: ReplayRequest):
    from backend.app.streaming.telemetry_stream import TelemetryStream
    stream = TelemetryStream(machine=request.machine, policy=request.policy)
    window_size = stream.buffer.config.window_size
    warm_start, values, labels = load_real_smd(
        request.machine, request.partition, request.start_index, request.limit, window_size
    )
    # Procesar solamente el contexto indispensable: a lo sumo window_size - 1
    # observaciones previas. Mantiene secuencia, contadores y buffer coherentes
    # sin manipular atributos privados de TelemetryStream.
    for offset, observation in enumerate(values):
        index = warm_start + offset
        result = stream.process(observation, sequence_id=index)
        if index < request.start_index:
            continue
        payload = asdict(result)
        payload["point_index"] = index  # Índice original para el dashboard
        payload["dataset"] = "smd"
        payload["partition"] = request.partition
        payload["source_index"] = index
        payload["ground_truth"] = int(labels[index - request.start_index]) if labels is not None else None
        yield payload

# Autenticación local. Primera instalación: crear el administrador desde CLI.
class LoginPayload(BaseModel):
    username: str = Field(min_length=1, max_length=48)
    password: str = Field(min_length=1, max_length=256)


class PasswordPayload(BaseModel):
    current_password: str
    new_password: str


@app.get("/api/auth/session")
def auth_session(request: Request):
    session = current_session(request)
    return {"configured": has_admin(), "authenticated": bool(session),
            "username": session["username"] if session else None,
            "csrf_token": session["csrf_token"] if session else None}


@app.post("/api/auth/login")
def auth_login(payload: LoginPayload, request: Request, response: Response):
    if not has_admin():
        raise HTTPException(status_code=503, detail="Creá el administrador local desde la terminal antes de ingresar")
    # Las peticiones de navegador deben provenir del mismo origen.
    origin = request.headers.get("origin")
    if origin and origin != f"{request.url.scheme}://{request.headers.get('host')}":
        raise HTTPException(status_code=403, detail="Origen de login no autorizado")
    token, csrf = authenticate(payload.username, payload.password)
    revoke(request.cookies.get(COOKIE_NAME))
    response.set_cookie(key=COOKIE_NAME, value=token, httponly=True,
                        secure=request.url.scheme == "https", samesite="strict",
                        max_age=SESSION_SECONDS, path="/")
    return {"authenticated": True, "username": payload.username, "csrf_token": csrf}


@app.post("/api/auth/logout", dependencies=[Depends(require_csrf)])
def auth_logout(request: Request, response: Response):
    revoke(request.cookies.get(COOKIE_NAME))
    response.delete_cookie(COOKIE_NAME, path="/", samesite="strict")
    return {"authenticated": False}


@app.post("/api/auth/change-password", dependencies=[Depends(require_csrf)])
def auth_change_password(payload: PasswordPayload, response: Response):
    try:
        change_password(payload.current_password, payload.new_password)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    response.delete_cookie(COOKIE_NAME, path="/", samesite="strict")
    return {"password_changed": True, "authenticated": False}


@app.get("/api/config", dependencies=[Depends(require_user)])
def read_runtime_config():
    return get_settings().model_dump()


@app.put("/api/config", dependencies=[Depends(require_csrf)])
def update_runtime_config(payload: RuntimeSettings):
    return save_settings(payload).model_dump()


@app.get("/api/health")
def health():
    return {"service": "apetria", "status": "online", "mode": "reproduccion_datasets_reales", "synthetic": False}

@app.get("/api/sources", dependencies=[Depends(require_user)])
def sources():
    smd_dir = ROOT / "data" / "processed" / "smd"

    smd_machines = sorted(
        p.stem
        for p in smd_dir.glob("machine-*.npz")
        if not p.stem.endswith("_preprocessing")
    ) if smd_dir.exists() else []

    nasa_dir = ROOT / "data" / "processed" / "nasa"
    nasa_channels = []

    for dataset_file in sorted(nasa_dir.glob("*.npz")):
        channel = dataset_file.stem

        required = [
            ROOT / "models/nasa/isolation_forest" / f"{channel}.joblib",
            ROOT / "models/nasa/isolation_forest" / f"{channel}_metadata.json",
            ROOT / "models/nasa/lstm_autoencoder" / f"{channel}.pt",
            ROOT / "results/calibration/nasa" / f"{channel}_lstm_threshold.json",
        ]

        if all(p.is_file() for p in required):
            nasa_channels.append(channel)

    # NAB: series con telemetria y predicciones historicas disponibles.
    nab_dir = ROOT / "data" / "processed" / "nab" / "aws_v1"
    nab_predictions = ROOT / "results" / "hybrid" / "nab" / "nab_hybrid_predictions.csv.gz"
    nab_series = []

    if nab_predictions.is_file() and nab_dir.is_dir():
        from backend.app.adapters.nab_replay import NABReplay
        nab_series = [
            name for name in NABReplay().series
            if (nab_dir / name).is_file()
        ]

    sources_data = []

    for key in DATASETS:
        if key == "smd":
            machines = smd_machines
        elif key == "nasa":
            machines = nasa_channels
        elif key == "nab":
            machines = nab_series
        else:
            machines = []

        hdfs_ready = (
            ROOT / "results" / "hybrid" / "hdfs"
            / "hdfs_hybrid_predictions.csv"
        ).is_file()

        bgl_ready = (
            ROOT / "results" / "hybrid" / "bgl"
            / "bgl_hybrid_predictions.csv.gz"
        ).is_file()

        status = (
            ("disponible" if bgl_ready else "faltan_datos")
            if key == "bgl"
            else (
            ("disponible" if hdfs_ready else "faltan_datos")
            if key == "hdfs"
            else (
                ("disponible" if machines else "faltan_datos")
                if key in ("smd", "nasa", "nab")
                else "pendiente_integracion"
            )
            )
        )

        artifacts_available = status == "disponible"
        enabled = is_source_enabled(key)
        entry = {
            "id": key,
            "status": status if enabled else "deshabilitada",
            "enabled": enabled,
            "artifacts_available": artifacts_available,
            "machines": machines,
            "note": "No se muestran predicciones ficticias",
        }

        if key == "hdfs":
            entry.update({
                "mode": "historical_offline_replay",
                "unit": "block_session",
                "partition": "historical_test",
                "total_sessions": 115525 if hdfs_ready else 0,
                "policies": ["if", "lstm", "and", "or"],
                "endpoint": "/api/replay/hdfs",
                "websocket": "/ws/replay/hdfs",
            })

        if key == "bgl":
            entry.update({
                "mode": "historical_offline_replay",
                "unit": "log_event",
                "partition": "historical_test",
                "total_events": 949593 if bgl_ready else 0,
                "policies": ["if", "sgd", "and", "or"],
                "endpoint": "/api/replay/bgl",
                "websocket": "/ws/replay/bgl",
            })

        sources_data.append(entry)

    return {"sources": sources_data}

@app.post("/api/replay/smd", dependencies=[Depends(require_csrf)])
def replay_smd(request: ReplayRequest):
    require_source_enabled("smd")
    try:
        return {"source": "smd", "results": list(run_replay(request))}
    except (FileNotFoundError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

@app.websocket("/ws/replay/smd")
async def ws_replay(websocket: WebSocket):
    if not ws_allowed(websocket):
        await websocket.close(code=4401)
        return
    if not is_source_enabled("smd"):
        await websocket.close(code=1008, reason="Fuente deshabilitada")
        return
    await websocket.accept()
    try:
        params = ReplayRequest.model_validate_json(await websocket.receive_text())
        # La inferencia es CPU-bound; no bloquear el event loop mientras se ejecuta.
        iterator = await asyncio.to_thread(lambda: list(run_replay(params)))
        for result in iterator:
            if not ws_allowed(websocket):
                await websocket.close(code=4401)
                return
            if not is_source_enabled("smd"):
                await websocket.close(code=1008, reason="Fuente deshabilitada")
                return
            await websocket.send_json({"type": "observation", "data": result})
            await asyncio.sleep(0)
        await websocket.send_json({"type": "completed", "count": len(iterator)})
    except WebSocketDisconnect:
        return
    except Exception as exc:
        if websocket.client_state.name == "CONNECTED" and websocket.application_state.name == "CONNECTED":
            try:
                await websocket.send_json({"type": "error", "message": str(exc)})
            except (RuntimeError, OSError):
                pass
    finally:
        # El cliente puede desconectarse durante el cálculo o el envío.
        # Un cierre redundante no debe generar una excepción ASGI.
        if websocket.client_state.name == "CONNECTED" and websocket.application_state.name == "CONNECTED":
            try:
                await websocket.close()
            except (WebSocketDisconnect, RuntimeError, OSError):
                pass


# ============================================================
# NASA SMAP/MSL - Reproduccion incremental con inferencia real
# ============================================================

class NASAReplayRequest(BaseModel):
    channel: str = Field(default="A-1", pattern=r"^[A-Z]-[0-9]{1,2}$")
    partition: str = Field(default="X_test", pattern=r"^(X_test|X_val)$")
    policy: str = Field(default="and", pattern=r"^(and|or|dual)$")
    limit: int = Field(default=100, ge=1, le=10000)
    start_index: int = Field(default=0, ge=0)


def load_real_nasa(
    channel: str,
    partition: str,
    start_index: int,
    limit: int,
    window_size: int,
):
    path = ROOT / "data" / "processed" / "nasa" / f"{channel}.npz"

    if not path.is_file():
        raise FileNotFoundError(f"Canal NASA no disponible: {channel}")

    with np.load(path, allow_pickle=False) as data:
        if partition not in data.files:
            raise ValueError(f"Particion NASA ausente: {partition}")

        total = len(data[partition])
        end_index = start_index + limit

        if start_index >= total or end_index > total:
            raise ValueError(
                f"Rango [{start_index}, {end_index}) fuera "
                f"de la particion de {total} observaciones"
            )

        warm_start = max(0, start_index - window_size + 1)

        values = np.asarray(
            data[partition][warm_start:end_index],
            dtype=np.float32,
        ).copy()

        labels = None

        if partition == "X_test" and "y_test" in data.files:
            labels = np.asarray(
                data["y_test"][start_index:end_index],
                dtype=np.int8,
            ).copy()

            if len(labels) != limit or not np.isin(labels, [0, 1]).all():
                raise ValueError("Etiquetas NASA invalidas")

    if (
        values.ndim != 2
        or values.shape[1] != 25
        or not np.isfinite(values).all()
    ):
        raise ValueError("NASA requiere 25 caracteristicas finitas")

    return warm_start, values, labels


def run_nasa_replay(request: NASAReplayRequest):
    from backend.app.adapters.nasa_inference import NASAInference
    from backend.app.streaming.telemetry_stream import TelemetryStream

    inference = NASAInference(machine=request.channel, device="cpu")

    stream = TelemetryStream(
        machine=request.channel,
        policy=request.policy,
        inference_service=inference,
    )

    warm_start, values, labels = load_real_nasa(
        channel=request.channel,
        partition=request.partition,
        start_index=request.start_index,
        limit=request.limit,
        window_size=inference.lstm_config.window_size,
    )

    for offset, observation in enumerate(values):
        index = warm_start + offset

        result = stream.process(
            observation,
            sequence_id=index,
        )

        if index < request.start_index:
            continue

        payload = asdict(result)

        payload["point_index"] = index
        payload["dataset"] = "nasa"
        payload["partition"] = request.partition
        payload["source_index"] = index
        payload["ground_truth"] = (
            int(labels[index - request.start_index])
            if labels is not None
            else None
        )

        yield payload


@app.post("/api/replay/nasa", dependencies=[Depends(require_csrf)])
def replay_nasa(request: NASAReplayRequest):
    require_source_enabled("nasa")
    try:
        return {
            "source": "nasa",
            "results": list(run_nasa_replay(request)),
        }
    except (FileNotFoundError, ValueError, KeyError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc


# ============================================================
# NASA SMAP/MSL - WebSocket de reproduccion
# ============================================================

@app.websocket("/ws/replay/nasa")
async def ws_replay_nasa(websocket: WebSocket):
    if not ws_allowed(websocket):
        await websocket.close(code=4401)
        return
    if not is_source_enabled("nasa"):
        await websocket.close(code=1008, reason="Fuente deshabilitada")
        return
    await websocket.accept()

    try:
        params = NASAReplayRequest.model_validate_json(
            await websocket.receive_text()
        )

        # La inferencia se ejecuta fuera del event loop.
        # Los resultados se calculan antes de transmitirlos.
        results = await asyncio.to_thread(
            lambda: list(run_nasa_replay(params))
        )

        for result in results:
            if not ws_allowed(websocket):
                await websocket.close(code=4401)
                return
            if not is_source_enabled("nasa"):
                await websocket.close(code=1008, reason="Fuente deshabilitada")
                return
            await websocket.send_json({
                "type": "observation",
                "data": result,
            })
            await asyncio.sleep(0)

        await websocket.send_json({
            "type": "completed",
            "count": len(results),
        })

    except WebSocketDisconnect:
        return

    except Exception as exc:
        if (
            websocket.client_state.name == "CONNECTED"
            and websocket.application_state.name == "CONNECTED"
        ):
            try:
                await websocket.send_json({
                    "type": "error",
                    "message": str(exc),
                })
            except (RuntimeError, OSError):
                pass

    finally:
        if (
            websocket.client_state.name == "CONNECTED"
            and websocket.application_state.name == "CONNECTED"
        ):
            try:
                await websocket.close()
            except (WebSocketDisconnect, RuntimeError, OSError):
                pass


# ============================================================
# NAB AWS - Reproduccion de predicciones historicas offline
# ============================================================

class NABReplayRequest(BaseModel):
    series: str = Field(
        default="ec2_cpu_utilization_24ae8d.csv",
        pattern=r"^[A-Za-z0-9_-]+\.csv$",
    )
    partition: str = Field(default="historical_test", pattern=r"^historical_test$")
    policy: str = Field(default="or", pattern=r"^(and|or|dual)$")
    limit: int = Field(default=100, ge=1, le=10000)
    start_index: int = Field(default=0, ge=0)


@app.post("/api/replay/nab", dependencies=[Depends(require_csrf)])
def replay_nab(request: NABReplayRequest):
    require_source_enabled("nab")
    from backend.app.adapters.nab_replay import NABReplay

    try:
        adapter = NABReplay()
        results = adapter.replay(
            series=request.series,
            start_index=request.start_index,
            limit=request.limit,
            policy=request.policy,
        )
        return {
            "source": "nab",
            "mode": "historical_offline_replay",
            "results": results,
        }
    except (FileNotFoundError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


# ============================================================
# NAB AWS - WebSocket de reproduccion historica offline
# ============================================================

@app.websocket("/ws/replay/nab")
async def ws_replay_nab(websocket: WebSocket):
    if not ws_allowed(websocket):
        await websocket.close(code=4401)
        return
    if not is_source_enabled("nab"):
        await websocket.close(code=1008, reason="Fuente deshabilitada")
        return
    await websocket.accept()

    try:
        params = NABReplayRequest.model_validate_json(
            await websocket.receive_text()
        )

        # Las predicciones son historicas; no se ejecuta inferencia en vivo.
        results = await asyncio.to_thread(
            lambda: replay_nab_results(params)
        )

        for result in results:
            if not ws_allowed(websocket):
                await websocket.close(code=4401)
                return
            if not is_source_enabled("nab"):
                await websocket.close(code=1008, reason="Fuente deshabilitada")
                return
            await websocket.send_json({
                "type": "observation",
                "data": result,
            })
            await asyncio.sleep(0)

        await websocket.send_json({
            "type": "completed",
            "count": len(results),
        })

    except WebSocketDisconnect:
        return

    except Exception as exc:
        if (
            websocket.client_state.name == "CONNECTED"
            and websocket.application_state.name == "CONNECTED"
        ):
            try:
                await websocket.send_json({
                    "type": "error",
                    "message": str(exc),
                })
            except (RuntimeError, OSError):
                pass

    finally:
        if (
            websocket.client_state.name == "CONNECTED"
            and websocket.application_state.name == "CONNECTED"
        ):
            try:
                await websocket.close()
            except (WebSocketDisconnect, RuntimeError, OSError):
                pass


def replay_nab_results(request: NABReplayRequest):
    from backend.app.adapters.nab_replay import NABReplay

    adapter = NABReplay()
    return adapter.replay(
        series=request.series,
        start_index=request.start_index,
        limit=request.limit,
        policy=request.policy,
    )

# ============================================================
# HDFS - Reproduccion historica de sesiones de logs
# ============================================================

class HDFSReplayRequest(BaseModel):
    partition: str = Field(default="historical_test", pattern=r"^historical_test$")
    policy: str = Field(default="lstm", pattern=r"^(if|lstm|and|or)$")
    limit: int = Field(default=100, ge=1, le=10000)
    start_index: int = Field(default=0, ge=0)


def replay_hdfs_results(request: HDFSReplayRequest):
    from backend.app.adapters.hdfs_replay import HDFSReplay
    return HDFSReplay().replay(
        start_index=request.start_index,
        limit=request.limit,
        policy=request.policy,
    )


@app.post("/api/replay/hdfs", dependencies=[Depends(require_csrf)])
def replay_hdfs(request: HDFSReplayRequest):
    require_source_enabled("hdfs")
    try:
        return {
            "source": "hdfs",
            "mode": "historical_offline_replay",
            "results": replay_hdfs_results(request),
        }
    except (FileNotFoundError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.websocket("/ws/replay/hdfs")
async def ws_replay_hdfs(websocket: WebSocket):
    if not ws_allowed(websocket):
        await websocket.close(code=4401)
        return
    if not is_source_enabled("hdfs"):
        await websocket.close(code=1008, reason="Fuente deshabilitada")
        return
    await websocket.accept()
    try:
        params = HDFSReplayRequest.model_validate_json(
            await websocket.receive_text()
        )
        results = await asyncio.to_thread(
            lambda: replay_hdfs_results(params)
        )
        for result in results:
            if not ws_allowed(websocket):
                await websocket.close(code=4401)
                return
            if not is_source_enabled("hdfs"):
                await websocket.close(code=1008, reason="Fuente deshabilitada")
                return
            await websocket.send_json({
                "type": "observation",
                "data": result,
            })
            await asyncio.sleep(0)
        await websocket.send_json({
            "type": "completed",
            "count": len(results),
        })
    except WebSocketDisconnect:
        return
    except Exception as exc:
        if (
            websocket.client_state.name == "CONNECTED"
            and websocket.application_state.name == "CONNECTED"
        ):
            try:
                await websocket.send_json({
                    "type": "error",
                    "message": str(exc),
                })
            except (RuntimeError, OSError):
                pass
    finally:
        if (
            websocket.client_state.name == "CONNECTED"
            and websocket.application_state.name == "CONNECTED"
        ):
            try:
                await websocket.close()
            except (WebSocketDisconnect, RuntimeError, OSError):
                pass
# BGL - Reproduccion historica offline por evento de log

class BGLReplayRequest(BaseModel):
    partition: str = Field(default="historical_test", pattern=r"^historical_test$")
    policy: str = Field(default="sgd", pattern=r"^(if|sgd|and|or)$")
    limit: int = Field(default=100, ge=1, le=10000)
    start_index: int = Field(default=0, ge=0)


def replay_bgl_results(request: BGLReplayRequest):
    from backend.app.adapters.bgl_replay import BGLReplay
    return BGLReplay().replay(
        start_index=request.start_index,
        limit=request.limit,
        policy=request.policy,
    )


@app.post("/api/replay/bgl", dependencies=[Depends(require_csrf)])
def replay_bgl(request: BGLReplayRequest):
    require_source_enabled("bgl")
    try:
        return {
            "source": "bgl",
            "mode": "historical_offline_replay",
            "results": replay_bgl_results(request),
        }
    except (FileNotFoundError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.websocket("/ws/replay/bgl")
async def ws_replay_bgl(websocket: WebSocket):
    if not ws_allowed(websocket):
        await websocket.close(code=4401)
        return
    if not is_source_enabled("bgl"):
        await websocket.close(code=1008, reason="Fuente deshabilitada")
        return
    await websocket.accept()
    try:
        params = BGLReplayRequest.model_validate_json(
            await websocket.receive_text()
        )
        results = await asyncio.to_thread(
            lambda: replay_bgl_results(params)
        )
        for result in results:
            if not ws_allowed(websocket):
                await websocket.close(code=4401)
                return
            if not is_source_enabled("bgl"):
                await websocket.close(code=1008, reason="Fuente deshabilitada")
                return
            await websocket.send_json({
                "type": "observation",
                "data": result,
            })
            await asyncio.sleep(0)
        await websocket.send_json({
            "type": "completed",
            "count": len(results),
        })
    except WebSocketDisconnect:
        return
    except Exception as exc:
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(exc),
            })
        except (RuntimeError, OSError, WebSocketDisconnect):
            pass
    finally:
        try:
            await websocket.close()
        except (RuntimeError, OSError, WebSocketDisconnect):
            pass


# Debe estar al final: los endpoints API y WebSocket tienen prioridad sobre "/".
app.mount("/", SafeFrontendFiles(directory=ROOT / "frontend", html=True), name="frontend")
