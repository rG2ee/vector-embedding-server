FROM python:3.10.5-slim as base

# Python settings
ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PYTHONDONTWRITEBYTECODE=1


WORKDIR /app


FROM base as builder

ENV \
    # Pip settings
    PIP_DEFAULT_TIMEOUT=100 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    # Poetry settings
    POETRY_NO_INTERACTION=1 \
    POETRY_VERSION=1.5.1

# Setup poetry
RUN pip install "poetry==$POETRY_VERSION"
RUN python -m venv /venv

# Copy relevant files
COPY pyproject.toml poetry.lock .
COPY vector_embedding_server ./vector_embedding_server
RUN  touch /app/README.md # required by poetry somehow
#CMD ["tail", "-f", "/dev/null"]




RUN . /venv/bin/activate && \
    # Install dependencies
    poetry install -n --only main --no-root && \
    # Install root package
    poetry build -f wheel -n && \
    pip install --no-deps dist/*.whl && \
    rm -rf dist *.egg-info


FROM base as final

ENV PATH="/venv/bin:$PATH"

COPY --from=builder /venv /venv
COPY ./vector_embedding_server/server.py ./server.py
COPY ./vector_embedding_server/templates ./templates
CMD uvicorn server:app --host 0.0.0.0 --port 8080