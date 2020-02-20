FROM python:3.8.1-buster
WORKDIR /usr/src/app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
CMD python interact_mmi.py --no_cuda
