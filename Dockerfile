FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gfortran libatlas-base-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py ./main.py
COPY static ./static

RUN useradd -m appuser
USER appuser

ENV PORT=5022
EXPOSE 5022
ENV MAL_TOKEN=""

CMD ["/bin/sh","-c","\
  if [ -n \"$MAL_TOKEN\" ] && [ ! -f token.json ]; then \
    echo \"{\\\"access_token\\\": \\\"$MAL_TOKEN\\\"}\" > token.json; \
  fi; \
  python main.py \
"]
