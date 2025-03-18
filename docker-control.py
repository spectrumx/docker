import paho.mqtt.client as mqtt
import docker
import time
import json

docker_client = docker_client.from_env()

service_name = "docker"
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
    global service_name
    print(f"{service_name} connected to mqtt: {rc}")
    send_status(client)

def on_message(client, userdata, msg):
    payload = msg.payload.decode()
    print(f"Received message: {payload}")
    parts = payload.split()

    if parts[0] == "start":
        start_container(parts[1])
    elif parts[0] == "stop":
        stop_container(parts[1])
    elif parts[0] == "pull":
        pull_container(parts[1])
    elif parts[0] == "status":
        # List this as a valid command, but we send status with every command
        pass  
    else:
        print(f"Unknown command: {parts[0]}")
    
    send_status(client)

def send_status(client):
  global service_name
  # Get Docker container status
  containers_info = []
  for container in docker_client.containers.list():
    container_info = {
        'id': container.id,
        'name': container.name,
        'status': container.status,
        'image': container.image.tags
    }
    containers_info.append(container_info)
  #payload["docker"] = containers_info
  payload = {
      "state": "online", 
      "timestamp": time.time(),
      "task_name": "tasks.admin.save_docker",
      "arguments": {"containers": containers_info},
      }
  client.publish(service_name + "/status", json.dumps(payload), retain=True)

def start_container(container_name):
    try:
        container = docker_client.containers.get(container_name)
        container.start()
        print(f"Started container: {container_name}")
    except docker.errors.NotFound:
        print(f"Container not found: {container_name}")
    except docker.APIError as e:
        print(f"API error: {e}")

def stop_container(container_name):
    try:
        container = docker_client.containers.get(container_name)
        container.stop()
        print(f"Stopped container: {container_name}")
    except docker.errors.NotFound:
        print(f"Container not found: {container_name}")

def pull_container(image_name):
    try:
        docker_client.images.pull(image_name)
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