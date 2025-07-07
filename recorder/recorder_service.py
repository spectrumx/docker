import dataclasses
import json
import os
import pathlib
import shutil
import signal
import time
import traceback
from typing import Any, Optional

import aiomqtt
import anyio
import jsonargparse
from ruyaml import YAML

DRF_RECORDING_DIR = os.getenv("DRF_RECORDING_DIR", "/data/ringbuffer")
DRF_TMP_RINGBUFFER_DIR = os.getenv("DRF_TMP_RINGBUFFER_DIR", "/data/tmp-ringbuffer")
RECORDER_DEFAULT_CONFIG = os.getenv("RECORDER_DEFAULT_CONFIG", "sr1MHz")


@dataclasses.dataclass
class RecorderService:
    name: str = "recorder"
    recording_enabled: bool = False
    recording_scope: Optional[anyio.CancelScope] = None
    config: dict[str, Any] = dataclasses.field(default_factory=dict)
    loadable_configs: dict[str, dict[str, Any]] = dataclasses.field(
        default_factory=dict
    )


def load_configs(service, config_path=pathlib.Path("/app/configs")):
    config_paths = sorted(pathlib.Path(config_path).glob("*.yaml"))
    yaml = YAML(typ="safe")
    for p in config_paths:
        service.loadable_configs[p.stem] = jsonargparse.dict_to_namespace(yaml.load(p))
    if RECORDER_DEFAULT_CONFIG in service.loadable_configs:
        service.config = service.loadable_configs[RECORDER_DEFAULT_CONFIG]
    else:
        service.config = list(service.loadable_configs.keys())[0]


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


async def send_error(client, service, message, response_topic=None):
    if response_topic is None:
        response_topic = f"{service.name}/error"
    payload = {
        "message": message,
        "timestamp": time.time(),
    }
    await client.publish(response_topic, json.dumps(payload))


async def send_config(client, service, value, response_topic=None):
    if response_topic is None:
        response_topic = f"{service.name}/config/response"

    payload = {
        "value": value,
        "timestamp": time.time(),
    }
    await client.publish(response_topic, json.dumps(payload))


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
    shutil.rmtree(DRF_TMP_RINGBUFFER_DIR, ignore_errors=True)
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
        "1",
        DRF_TMP_RINGBUFFER_DIR,
    ]
    try:
        await anyio.run_process(command, stdout=None, stderr=None, check=False)
    finally:
        shutil.rmtree(DRF_TMP_RINGBUFFER_DIR, ignore_errors=True)


async def run_recorder(client, service):
    command = [
        "python3",
        "/app/sdr_mep_recorder.py",
        "--config",
        json.dumps(service.config.as_dict()),
    ]
    with anyio.CancelScope() as scope:
        service.recording_scope = scope
        await send_status(client, service)
        async with await anyio.open_process(
            command, stdout=None, stderr=None
        ) as process:
            try:
                await process.wait()
            finally:
                process.send_signal(signal.SIGINT)
                channel_dir = pathlib.Path(service.config["drf_sink.channel_dir"])
                shutil.rmtree(channel_dir, ignore_errors=True)
                service.recording_enabled = False
                service.recording_scope = None
                with anyio.CancelScope(shield=True):
                    await process.wait()
                    await send_status(client, service)


def disable_recording(service):
    if service.recording_enabled:
        service.recording_enabled = False
        service.recording_scope.cancel()
        service.recording_scope = None


def enable_recording(client, service, task_group):
    if not service.recording_enabled:
        service.recording_enabled = True
        task_group.start_soon(run_recorder, client, service)


async def process_config_command(client, service, payload):
    cmd = payload["task_name"].removeprefix("config.")
    args = payload.get("arguments", {})
    response_topic = payload.get("response_topic", None)
    try:
        if cmd == "get":
            key = args.get("key", "")
            try:
                if not key:
                    value = service.config
                else:
                    value = service.config[key]
                if isinstance(value, jsonargparse.Namespace):
                    value = value.as_dict()
            except KeyError:
                msg = f"ERROR config.get: key '{key}' not found."
                await send_error(client, service, msg, response_topic)
            else:
                await send_config(client, service, value, response_topic)
        if cmd == "set":
            key = args.get("key", "")
            val = args["value"]
            if isinstance(val, dict):
                val = jsonargparse.dict_to_namespace(val)
            service.config.update(val, key)
            await send_config(client, service, service.config.as_dict(), response_topic)
        if cmd == "list":
            available_config_names = sorted(list(service.loadable_configs.keys()))
            if response_topic is None:
                response_topic = f"{service.name}/config/response"
            payload = {
                "available_configs": available_config_names,
                "timestamp": time.time(),
            }
            await client.publish(response_topic, json.dumps(payload))
        if cmd == "load":
            config_name = args["name"]
            try:
                service.config = service.loadable_configs[config_name]
            except KeyError:
                msg = f"ERROR config.load: configuration '{config_name}' not found."
                await send_error(client, service, msg, response_topic)
            else:
                await send_config(
                    client, service, service.config.as_dict(), response_topic
                )
    except Exception:
        msg = f"ERROR config:\n{traceback.format_exc()}"
        await send_error(client, service, msg, response_topic)


async def process_commands(client, service, task_group):
    async for message in client.messages:
        payload = json.loads(message.payload.decode())
        if payload["task_name"] == "disable":
            disable_recording(service)
        if payload["task_name"] == "enable":
            enable_recording(client, service, task_group)
        if payload["task_name"] == "status":
            await send_status(client, service)
        if payload["task_name"].startswith("config."):
            await process_config_command(client, service, payload)


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
                    tg.start_soon(process_commands, client, service, tg)
        except aiomqtt.MqttError:
            msg = (
                "Connection to MQTT server lost;"
                f" Reconnecting in {interval} seconds ..."
            )
            print(msg)
            await anyio.sleep(interval)


if __name__ == "__main__":
    anyio.run(main)
