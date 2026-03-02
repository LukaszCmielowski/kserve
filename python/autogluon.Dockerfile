ARG PYTHON_VERSION=3.12
ARG BASE_IMAGE=registry.access.redhat.com/ubi9/python-312:latest
ARG VENV_PATH=/prod_venv

FROM ${BASE_IMAGE} AS builder
USER root
WORKDIR /

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    ln -s /root/.local/bin/uv /usr/local/bin/uv

# Create virtual environment
ARG VENV_PATH
ENV VIRTUAL_ENV=${VENV_PATH}
RUN uv venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# ========== Install kserve dependencies ==========
COPY kserve/pyproject.toml kserve/uv.lock kserve/
RUN cd kserve && uv sync --active --no-cache --index https://pypi.org/simple

COPY kserve kserve
RUN cd kserve && uv sync --active --no-cache --index https://pypi.org/simple

# ========== Install kserve storage dependencies ==========
COPY storage/pyproject.toml storage/uv.lock storage/
RUN cd storage && uv sync --active --no-cache --index https://pypi.org/simple

COPY storage storage
RUN cd storage && uv pip install . --no-cache --index https://pypi.org/simple

# ========== Install autogluonserver dependencies ==========
COPY autogluonserver autogluonserver
RUN cd autogluonserver && uv sync --active --no-cache --index https://pypi.org/simple
RUN python -c "print('Importing lighgbm');import lightgbm;print(lightgbm.__version__)"

# Generate third-party licenses
COPY pyproject.toml pyproject.toml
COPY third_party/pip-licenses.py pip-licenses.py
RUN mkdir -p third_party/library && python3 pip-licenses.py


# =================== Final stage ===================
FROM ${BASE_IMAGE} AS prod
USER root

# Note: BASE_IMAGE (e.g. aipcc/cpu) often has no dnf repos. Ensure it provides libgomp
# for OpenMP (LightGBM, XGBoost); otherwise use a prod image that includes it.

COPY third_party third_party

# Activate virtual env
ARG VENV_PATH
ENV VIRTUAL_ENV=${VENV_PATH}
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN useradd kserve -m -u 1000 -d /home/kserve

COPY --from=builder --chown=kserve:kserve /third_party third_party
COPY --from=builder --chown=kserve:kserve /$VIRTUAL_ENV $VIRTUAL_ENV
COPY --from=builder /kserve kserve
COPY --from=builder /storage storage
COPY --from=builder /autogluonserver autogluonserver

USER 1000
ENV PYTHONPATH=/autogluonserver
ENTRYPOINT ["python", "-m", "autogluonserver"]
