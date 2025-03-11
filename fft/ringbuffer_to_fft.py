# ----------------------------------------------------------------------------
# Copyright (c) 2017 Massachusetts Institute of Technology (MIT)
# All rights reserved.
#
# Distributed under the terms of the BSD 3-clause license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
from __future__ import absolute_import, division, print_function

import os
import numpy as np
import time
import datetime
import tempfile
import paho.mqtt.client as mqtt
import argparse
import base64
import json

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import digital_rf
from digital_rf.watchdog_drf import DirWatcher

service_name = "fft"
disabled = False

announce_packet = {
  "title": "FFT generator",
  "description": "Generate fft from last 1024 samples in /data/ringbuffer",
  "author": "Randy Herban <rherban@nd.edu>",
  "url": "https://hub.docker.com/u/randyherban/fft",
  "source": "https://github.com/spectrumx/docker/fft",
  "output":                               
    {
      "output_name":                      
      {  
        "type": "mqtt",                  
        "value": "fft/output_topic"
      },
    },
  "version": "0.1",
  "type": "algorithm",       
  "time_started": time.time(),
}

# MQTT functions
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        send_status(client)

def on_message(client, userdata, msg):
  global disabled
  payload = json.loads(msg.payload.decode())
  if payload['task_name'] == "disable":
    disabled = True
    send_status(client)
  if payload['task_name'] == "enable":
    disabled = False
    send_status(client)
  if payload['task_name'] == "status":
    send_status(client)

def send_status(client):
  global service_name
  payload = {"state": "disabled" if disabled else "online", "timestamp": time.time()}
  client.publish(service_name + "/status", json.dumps(payload))



# DigitalRF functions
class NewFileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            process_file(event.src_path, "new")

def read_iq_data(data_dir, channel):
    # Create a DigitalRFReader object
    reader = drf.DigitalRFReader(data_dir)
    
    # Get the bounds of the data for the specified channel
    start_sample, end_sample = reader.get_bounds(channel)
    
    # Read the data
    data = reader.read_vector(start_sample, end_sample - start_sample + 1, channel)
    
    return data

def process_file(data):
	fft_result_b64 = base64.b64encode(np.abs(np.fft.fft(data)).astype(np.float32).tobytes()).decode('utf-8') # Encode to base64 string

	print(f"{len(data)} samples read")
	payload = {
			"data": fft_result_b64,
			"type": "float32",
		"mac_address": "f4e11ea46780",
		"short_name": "WI-Lab V3.4-025 #6",
		"software_version": "v0.10b30",
		"latitude": 41.699584,
		"longitude": -86.237237,
		"altitude": 2,
		"batch": 0,
		"sample_rate": 24000000,
		"center_frequency": 2000000000,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f").replace(' ', 'T') + "+00:00",
		"gain": 1,
		"metadata": {
			"data_type": "periodogram",
			"fmin": 1988000000,
			"fmax": 2012000000,
			"nfft": 1024,
			"xcount": 1024,
			"gps_lock": False,
			"scan_time": 0.07766938209533691,
		},
		"hardware_version": "3.4",
		"hardware_board_id": "025"
	}

	client.publish('radiohound/clients/data/test', payload=json.dumps(payload))


client = mqtt.Client()
client.on_message = on_message
client.on_connect = on_connect
#Set last will & testament which sends an offline status when this client disconnects.  Has to come before the connect() call.
client.will_set(service_name + "/status", payload=json.dumps({"state":"offline"}), qos=0, retain=True)
client.connect('localhost', 1883, 60)
client.subscribe(service_name + "/command")
client.loop_start()

#Send announce packet
client.publish('announce/' + service_name, json.dumps(announce_packet), retain=True)

data_dir = '/data/ringbuffer'
channel = "ch0"
reader = digital_rf.DigitalRFReader(data_dir)

prev_end_sample = 0
while True:
    if not disabled:
        start_sample, end_sample = reader.get_bounds(channel)
        if end_sample - prev_end_sample > 0:
            try:
                process_file(reader.read_vector_1d(end_sample - 1024, 1024, channel))
            except:
                pass
        prev_end_sample = end_sample


    time.sleep(.1)
