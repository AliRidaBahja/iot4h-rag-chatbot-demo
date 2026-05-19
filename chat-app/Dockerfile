FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app .

CMD ["uvicorn", "main:app", "--reload", "--port", "8001", "--host", "0.0.0.0"]