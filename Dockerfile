FROM python:3.8-slim
#
COPY poetry.lock pyproject.toml ./
#
ARG INSTALL_DEV=false
RUN pip install --upgrade pip \
    && pip install -U poetry \
    && poetry config virtualenvs.create false \
    && bash -c "if [ $INSTALL_DEV == 'true' ] ; then poetry install --no-interaction --no-root; else poetry install --no-interaction --no-ansi --no-dev ; fi"
#
COPY . /app
WORKDIR /app
#
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]