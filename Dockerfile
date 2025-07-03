FROM python:3.13

# install uv package manager
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

WORKDIR /app
COPY app.py pyproject.toml /app
RUN uv sync

# Currently otherwise not working
ENV PYTHONASYNCIODEBUG=1

ENTRYPOINT ["./.venv/bin/shiny", "run","-h", "0.0.0.0", "-p", "8000", "./app.py"]
