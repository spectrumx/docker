# Docker images for the MEP Jetson
These are the services which enable the MEP Jetson to run and connect to the sensing platform.  

The docker compose file can be used to rebuid sub-containers like so:
```bash
cd /opt/radiohound/docker
docker-compose build                  #build all
docker-compose build radiohound_web   #build single
docker-compose build fft
```

To control services:
```bash
cd /opt/radiohound/docker
docker-compose up -d   
docker-compose down
```