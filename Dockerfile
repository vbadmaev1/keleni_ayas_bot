FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 DATA_DIR=/state
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY bot bot
COPY data/corpus.json data/corpus.json
COPY audio audio
# база, логи и кеш живут в томе, иначе пропадут при пересборке
VOLUME ["/state"]
CMD ["python", "-m", "bot.main"]
