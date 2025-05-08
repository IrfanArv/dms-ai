FROM python:3.11.5-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY ./app /app/app
COPY ./upload /app/upload
COPY ./vectorstore /app/vectorstore

ENV UPLOAD_DIR=/app/upload
RUN mkdir -p $UPLOAD_DIR

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 