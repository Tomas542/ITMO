FROM python:3.12-slim-trixie
COPY --from=ghcr.io/astral-sh/uv:0.10.9 /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml uv.lock README.md /app/
COPY lab2/src /app/src
COPY lab2/train.py lab2/gui.py lab2/docker-entrypoint.sh /app/

ENV PYTHONUNBUFFERED=1
ENV UV_NO_DEV=1
ENV PYTHONPATH=/app/src

RUN uv sync --locked --no-cache && chmod +x /app/docker-entrypoint.sh

EXPOSE 7860
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["gui"]
