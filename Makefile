# Makefile — uruchamiaj, buduj, restartuj bez pisania docker-compose za każdym razem

up:
	docker compose up

build:
	docker compose build

down:
	docker compose down

reset:
	docker compose down -v && docker compose up --build

logs:make ui
	docker compose logs -f app

bash:
	docker exec -it finance_dashboard_app bash

db:
	docker exec -it finance_dashboard_db psql -U $$POSTGRES_USER -d $$POSTGRES_DB

ps:
	docker ps --filter name=finance_dashboard

reload:
	touch app/main.py  # force streamlit reload
