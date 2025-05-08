FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV REDIS_HOST=zada-ai_redis:6379
ENV REDIS_PASSWORD=d534e80625a9971e571e
ENV REDIS_PORT=6379

COPY . .

ENV PORT=5000
EXPOSE $PORT

CMD gunicorn --bind 0.0.0.0:${PORT} app:app