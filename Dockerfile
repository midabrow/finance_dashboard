
FROM python:3.10-slim

# Zmienne środowiskowe
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Aktualizacja pakietów systemowych i instalacja zależności systemowych
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    libxml2-dev \
    libxslt1-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Katalog roboczy
WORKDIR /app

# Kopiowanie plików wymaganych do instalacji zależności
COPY requirements.txt .

# Instalacja zależności
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Kopiowanie aplikacji do kontenera
COPY . .

# Otwarcie portu dla Streamlit
EXPOSE 8501

# Komenda startowa
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.runOnSave=true"]
