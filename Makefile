.PHONY: install test download eval feedback deception demo serve clean

install:
	python3 -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -e ".[dev]"

test:
	. .venv/bin/activate && pytest

download:
	. .venv/bin/activate && python -m scripts.download_off_subset --n 60

eval:
	. .venv/bin/activate && python -m scripts.run_eval

feedback:
	. .venv/bin/activate && opie feedback-eval --rounds 5 --db

deception:
	. .venv/bin/activate && opie deception-eval

demo:
	. .venv/bin/activate && python -m scripts.demo_one --code $(CODE)

serve:
	. .venv/bin/activate && uvicorn opie.api.app:app --reload --port 8000

clean:
	rm -rf .pytest_cache **/__pycache__ *.sqlite results/
