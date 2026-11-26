.PHONY: up build down logs

s ?=

up:
	docker compose up -d $(s)

build:
	docker compose up -d --build  $(s)

down:
	docker compose down 

logs:
	docker compose logs $(s)