FROM python:3.12-slim-trixie
COPY --from=ghcr.io/astral-sh/uv:0.10.9 /uv /uvx /bin/

COPY . /app

ENV PYTHONUNBUFFERED=1
ENV UV_NO_DEV=1

WORKDIR /app
RUN uv sync --locked --no-cache && chmod +x /app/docker-entrypoint.sh

EXPOSE 7860
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["gui"]
