FROM python:3.10-slim@sha256:6ff506466f8b1e981719b468cc0023b8ef000b983cdec0eafe119d01f379e31d
RUN apt-get update && apt-get install -y --no-install-recommends libnuma1 libgomp1 git curl procps build-essential && rm -rf /var/lib/apt/lists/* && useradd -u 1000 -m -s /bin/bash worker
COPY docker/requirements.lock /opt/requirements.lock
RUN python -m venv /opt/fullbench && /opt/fullbench/bin/pip install --no-cache-dir pip==26.3 && /opt/fullbench/bin/pip install --no-cache-dir -r /opt/requirements.lock
COPY docker/task-shell.sh /opt/fullbench/task-shell.sh
ENV PATH=/opt/fullbench/bin:$PATH HOME=/home/worker PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
WORKDIR /workspace
CMD ["sleep", "infinity"]
