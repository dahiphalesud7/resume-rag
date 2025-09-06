# Use Python 3.10 slim as base
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

# Run Streamlit app from resume_rag folder
CMD ["streamlit", "run", "resume_rag/main.py", "--server.port=7860", "--server.address=0.0.0.0"]

