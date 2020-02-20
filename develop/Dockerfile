FROM python:3.8.1-buster
WORKDIR /usr/src/app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
CMD python interact_mmi.py --no_cuda
