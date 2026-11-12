FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 DATA_DIR=/data
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY bot bot
COPY corpus/corpus.json corpus/corpus.json
COPY audio audio
# база, логи и кеш живут в томе, иначе пропадут при пересборке
VOLUME ["/data"]
CMD ["python", "-m", "bot.main"]
