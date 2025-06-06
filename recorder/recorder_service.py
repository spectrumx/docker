import dataclasses
import json
import os
import pathlib
import shutil
import time
from typing import Optional

import aiomqtt
import anyio
import anyio.streams
from ruyaml import YAML

DRF_RECORDING_DIR = os.getenv("DRF_RECORDING_DIR", "/data/ringbuffer")
DRF_TMP_RINGBUFFER_DIR = os.getenv("DRF_TMP_RINGBUFFER_DIR", "/data/tmp-ringbuffer")
RECORDER_DEFAULT_CONFIG = os.getenv("RECORDER_DEFAULT_CONFIG", "sr1MHz")


@dataclasses.dataclass
class RecorderService:
    name: str = "recorder"
    recording_enabled: bool = False
    recording_scope: Optional[anyio.CancelScope] = None
    configs: dict[str, str] = dataclasses.field(default_factory=dict)
    active_config: Optional[str] = None


def load_configs(service, config_path=pathlib.Path("/app/configs")):
    config_paths = sorted(pathlib.Path(config_path).glob("*.yaml"))
    yaml = YAML(typ="safe")
    for p in config_paths:
        service.configs[p.stem] = yaml.load(p)
    if RECORDER_DEFAULT_CONFIG in service.configs:
        service.active_config = RECORDER_DEFAULT_CONFIG
    else:
        service.active_config = list(service.configs.keys())[0]


async def send_announce(client, service):
    payload = {
        "title": "Recorder",
        "description": f"Record data to {DRF_RECORDING_DIR}",
        "author": "Ryan Volz <rvolz@mit.edu>",
        "url": "ghcr.io/ryanvolz/mep-recorder:latest",
        "source": "https://github.com/spectrumx/docker/recorder",
        "output": {
            "output_name": {"type": "disk", "value": f"{DRF_RECORDING_DIR}"},
        },
        "version": "0.1",
        "type": "service",
        "time_started": time.time(),
    }
    await client.publish(f"announce/{service.name}", json.dumps(payload), retain=True)


async def send_status(client, service):
    payload = {
        "state": "recording" if service.recording_enabled else "waiting",
        "timestamp": time.time(),
    }
    await client.publish(f"{service.name}/status", json.dumps(payload), retain=True)


async def run_drf_mirror(service):
    command = [
        "drf",
        "mirror",
        "mv",
        "--verbose",
        ".",
        DRF_RECORDING_DIR,
    ]
    await anyio.run_process(command, stdout=None, stderr=None, check=False)


async def run_drf_mirror_tmp(service):
    command = [
        "drf",
        "mirror",
        "cp",
        "--link",
        DRF_RECORDING_DIR,
        DRF_TMP_RINGBUFFER_DIR,
    ]
    await anyio.run_process(command, stdout=None, stderr=None, check=False)


async def run_drf_ringbuffer_tmp(service):
    command = [
        "drf",
        "ringbuffer",
        "-l",
        "2",
        DRF_TMP_RINGBUFFER_DIR,
    ]
    with anyio.CancelScope() as scope:
        try:
            await anyio.run_process(command, stdout=None, stderr=None, check=False)
        finally:
            shutil.rmtree(DRF_TMP_RINGBUFFER_DIR, ignore_errors=True)


async def run_recorder(service):
    config = service.configs[service.active_config]
    command = [
        "python3",
        "/app/sdr_mep_recorder.py",
        "--config",
        json.dumps(config),
    ]
    with anyio.CancelScope() as scope:
        service.recording_scope = scope
        try:
            await anyio.run_process(command, stdout=None, stderr=None, check=False)
        finally:
            channel_dir = pathlib.Path(config["drf_sink"]["channel_dir"])
            shutil.rmtree(channel_dir, ignore_errors=True)


def disable_recording(service):
    if service.recording_enabled:
        service.recording_enabled = False
        service.recording_scope.cancel()
        service.recording_scope = None


def enable_recording(service, task_group):
    if not service.recording_enabled:
        service.recording_enabled = True
        task_group.start_soon(run_recorder, service)


async def process_commands(client, service, task_group):
    async for message in client.messages:
        payload = json.loads(message.payload.decode())
        if payload["task_name"] == "disable":
            disable_recording(service)
            await send_status(client, service)
        if payload["task_name"] == "enable":
            enable_recording(service, task_group)
            await send_status(client, service)
        if payload["task_name"] == "status":
            await send_status(client, service)


async def main():
    service = RecorderService()
    load_configs(service)
    will = aiomqtt.Will(
        f"{service.name}/status",
        payload=json.dumps({"state": "offline"}),
        qos=0,
        retain=True,
    )
    client = aiomqtt.Client(
        "localhost",
        1883,
        keepalive=60,
        will=will,
    )
    interval = 5  # seconds
    while True:
        try:
            async with client:
                await client.subscribe(f"{service.name}/command")
                await send_announce(client, service)
                await send_status(client, service)
                async with anyio.create_task_group() as tg:
                    tg.start_soon(run_drf_mirror, service)
                    tg.start_soon(run_drf_mirror_tmp, service)
                    tg.start_soon(run_drf_ringbuffer_tmp, service)
                    enable_recording(service, tg)
                    tg.start_soon(process_commands, client, service, tg)
        except aiomqtt.MqttError:
            print(f"Connection lost; Reconnecting in {interval} seconds ...")
            await anyio.sleep(interval)


if __name__ == "__main__":
    anyio.run(main)
