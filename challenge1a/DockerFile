
FROM --platform=linux/amd64 python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY pdfextract.py /app/pdfextract.py
COPY input /app/input
COPY output /app/output

RUN pip install --no-cache-dir pymupdf

ENV INPUT_DIR=/app/input
ENV OUTPUT_DIR=/app/output

CMD ["python", "pdfextract.py"]
