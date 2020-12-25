FROM continuumio/miniconda3
COPY . usr/local/python/
EXPOSE 5000
WORKDIR /usr/local/python/
RUN pip install -r requirements.txt
CMD python runtime.py