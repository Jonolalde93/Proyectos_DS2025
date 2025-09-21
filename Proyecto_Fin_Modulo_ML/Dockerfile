FROM python:3.12-slim

RUN mkdir app

COPY app/ app/

RUN mkdir modelo

COPY models/ models/

COPY pyproject.toml ./
COPY uv.lock .

RUN python -m pip install --upgrade pip setuptools wheel

RUN python -m pip install uv
RUN pip install pandas==2.2.2

RUN uv lock
RUN uv sync


EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "app/app.py"]
