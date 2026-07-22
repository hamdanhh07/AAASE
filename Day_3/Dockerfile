FROM python:3.12-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY skeleton_enterprise_multiagent.py app.py

ENV REPORTS_DIR=/reports
RUN mkdir /reports
VOLUME /reports

ENV LAB_STAGE=5
EXPOSE 8000
CMD ["python", "app.py", "serve"]
