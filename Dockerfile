# Build Stage
FROM python:3.11-alpine AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Runtime Stage
FROM python:3.11-alpine AS runner

WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY app/ ./app/

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
