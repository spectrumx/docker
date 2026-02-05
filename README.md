# Docker images for the MEP Jetson
These are the services which enable the MEP Jetson to run and connect to the sensing platform.  

To control services:
```bash
cd /opt/radiohound/docker
git pull
docker compose pull
docker compose up -d --force-recreate
docker compose down
```