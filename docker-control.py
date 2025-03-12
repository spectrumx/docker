import paho.mqtt.client as mqtt
import docker
import time
import json

docker = docker.from_env()

service_name = "docker-control"
announce_packet = {
  "title": "Docker Control Script",
  "description": "Manages Docker containers",
  "author": "Randy Herban <rherban@nd.edu>",
  "url": None,
  "source": "github.com/spectrumx/docker/docker-control.py",
  "output": {},
  "version": "0.1",
  "type": "service",       
  "time_started": time.time(),
}

# Define MQTT callbacks
def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")

def on_message(client, userdata, msg):
    payload = msg.payload.decode()
    print(f"Received message: {payload}")
    command, container_name = payload.split()

    if command == "start":
        start_container(container_name)
    elif command == "stop":
        stop_container(container_name)
    elif command == "pull":
        pull_container(container_name)
    else:
        print(f"Unknown command: {command}")

def send_status(client):
  global service_name
  payload = {"state": "oneline", "timestamp": time.time()}
  # Get Docker container status
  containers_info = []
  for container in client.containers.list():
    container_info = {
        'id': container.id,
        'name': container.name,
        'status': container.status,
        'image': container.image.tags
    }
    containers_info.append(container_info)
  payload["docker"] = containers_info
  client.publish(service_name + "/status", json.dumps(payload))

def start_container(container_name):
    try:
        container = docker.containers.get(container_name)
        container.start()
        print(f"Started container: {container_name}")
    except docker.errors.NotFound:
        print(f"Container not found: {container_name}")

def stop_container(container_name):
    try:
        container = docker.containers.get(container_name)
        container.stop()
        print(f"Stopped container: {container_name}")
    except docker.errors.NotFound:
        print(f"Container not found: {container_name}")

def pull_container(image_name):
    try:
        docker.images.pull(image_name)
        print(f"Pulled image: {image_name}")
    except docker.errors.APIError as e:
        print(f"Failed to pull image: {image_name}, error: {e}")

# Initialize MQTT client
mqtt_client = mqtt.Client()
mqtt_client.on_message = on_message
mqtt_client.on_connect = on_connect
#Set last will & testament which sends an offline status when this client disconnects.  Has to come before the connect() call.
mqtt_client.will_set(service_name + "/status", payload=json.dumps({"state":"offline"}), qos=0, retain=True)
mqtt_client.connect('localhost', 1883, 60)
mqtt_client.subscribe(service_name + "/command")

# Start MQTT loop in background, since there's no active tasks for us to run
mqtt_client.loop_forever()