# Gunakan image Python resmi yang ringan
FROM python:3.11-slim

# Set working directory di dalam container
WORKDIR /code

# Copy requirements dulu agar caching efisien
COPY requirements.txt .

# Install dependencies (Gunakan versi CPU agar hemat space dan cepat)
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy seluruh kode projek ke dalam container
COPY . .

# Beri izin akses ke folder (penting untuk Hugging Face)
RUN chmod -R 777 /code

# Jalankan uvicorn. 
# PORT WAJIB 7860 untuk Hugging Face Spaces
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]