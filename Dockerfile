# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.10
FROM python:${PYTHON_VERSION}-slim

# ----------------------------------
# Environment
# ----------------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ----------------------------------
# Working directory
# ----------------------------------
WORKDIR /app

# ----------------------------------
# Install system deps (safe for TF)
# ----------------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------------
# Python dependencies (CACHED)
# ----------------------------------
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ----------------------------------
# App files
# ----------------------------------
COPY Dashboard.py .
COPY exoplanet_model.keras .

# ----------------------------------
# Non-root user (BEST PRACTICE)
# ----------------------------------
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser \
 && chown -R appuser:appuser /app

USER appuser

# ----------------------------------
# Streamlit
# ----------------------------------
EXPOSE 8501
CMD ["streamlit", "run", "Dashboard.py", "--server.address=0.0.0.0"]
