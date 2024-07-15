FROM python:3.9-slim

WORKDIR /app

RUN pip install --root-user-action=ignore --upgrade pip

COPY . .
RUN pip install --root-user-action=ignore -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "Academic Generator.py", "--server.maxUploadSize", "30"]