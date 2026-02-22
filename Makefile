.PHONY: run stop test clean

run:
	docker compose up --build

stop:
	docker compose down

test:
	cd backend && pip install -r requirements.txt && pytest tests/ -v

clean:
	docker compose down -v --rmi all
	rm -rf backend/data_cache
