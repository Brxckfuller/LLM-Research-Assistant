PYTHON = .venv/bin/python

.PHONY: setup run test

setup:
	python3 -m venv .venv
	$(PYTHON) -m pip install -r requirements.txt

run:
	.venv/bin/streamlit run app/streamlit_app.py

test:
	PYTHONPATH=. $(PYTHON) -m pytest tests/ -q
