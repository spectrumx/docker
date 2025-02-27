FROM randyherban/base-image
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
ENV FLASK_ENV=development
EXPOSE 5001
CMD ["python", "-u", "flask/app.py"]