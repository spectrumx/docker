import dataclasses
import json
import pathlib
import shutil
import time
from typing import Optional

import aiomqtt
import anyio
import anyio.streams
from ruyaml import YAML


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
    if "sr1MHz" in service.configs:
        service.active_config = "sr1MHz"
    else:
        service.active_config = list(service.configs.keys())[0]


async def send_announce(client, service):
    payload = {
        "title": "Recorder",
        "description": "Record data to /data/ringbuffer/in",
        "author": "Ryan Volz <rvolz@mit.edu>",
        "url": "ghcr.io/ryanvolz/mep-recorder:latest",
        "source": "https://github.com/spectrumx/docker/recorder",
        "output": {
            "output_name": {"type": "disk", "value": "/data/ringbuffer/in"},
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
        ".",
        "/data/ringbuffer/in",
    ]
    async with await anyio.open_process(command) as process:
        async for text in anyio.streams.text.TextReceiveStream(process.stdout):
            print(text)


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
            async with await anyio.open_process(command) as process:
                async for text in anyio.streams.text.TextReceiveStream(process.stdout):
                    print(text)
        except anyio.get_cancelled_exc_class():
            channel_dir = pathlib.Path(config["drf_sink"]["channel_dir"])
            shutil.rmtree(channel_dir, ignore_errors=True)
            raise


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
        "mqtt",
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
                    enable_recording(service, tg)
                    tg.start_soon(process_commands, client, service, tg)
        except aiomqtt.MqttError:
            print(f"Connection lost; Reconnecting in {interval} seconds ...")
            await anyio.sleep(interval)


if __name__ == "__main__":
    anyio.run(main)
