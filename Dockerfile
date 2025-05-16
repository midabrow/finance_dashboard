# Startujemy od lekkiego obrazu Pythona
FROM python:3.10-slim

# Ustawiamy zmienną środowiskową, żeby Streamlit nie pytał o konfig
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Aktualizacja pakietów systemowych i instalacja zależności systemowych
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Tworzymy katalog roboczy
WORKDIR /app

# Kopiujemy pliki wymagane do instalacji zależności
COPY requirements.txt .

# Instalujemy zależności
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Kopiujemy całą aplikację do kontenera
COPY . .

# Otwieramy port dla Streamlit
EXPOSE 8501

# Komenda startowa
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.runOnSave=true"]
