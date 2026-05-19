FROM python:3.10

WORKDIR /app

COPY . /app

RUN pip install --default-timeout=100 --no-cache-dir -r requirements.txt

CMD ["python", "main.py"]