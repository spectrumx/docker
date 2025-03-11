FROM randyherban/base-image
WORKDIR /opt/radiohound/icarus
COPY . .
RUN apt-get update -y
RUN apt-get install -y iputils-ping
RUN pip install --no-cache-dir -r requirements.txt
CMD ["python", "-u", "start.py"]