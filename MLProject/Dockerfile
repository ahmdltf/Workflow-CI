# 1. Gunakan base image Python resmi
FROM python:3.11-slim

# 2. Atur direktori kerja di dalam container
WORKDIR /app

# 3. Salin file requirements.txt terlebih dahulu agar bisa memanfaatkan cache Docker
COPY requirements.txt .

# 4. Install library yang dibutuhkan
RUN pip install --no-cache-dir -r requirements.txt

# 5. Salin seluruh isi folder MLProject ke dalam container
COPY . .

# 6. Jalankan script modeling kamu saat container dinyalakan
CMD ["python", "modelling.py"]