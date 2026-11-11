FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py recommender.py ./
COPY static ./static
RUN useradd --create-home appuser
USER appuser
ENV PORT=5022
EXPOSE 5022
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 180 main:app"]
