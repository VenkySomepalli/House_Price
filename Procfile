#web: sh setup.sh && streamlit run src/app.py
web: gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app
