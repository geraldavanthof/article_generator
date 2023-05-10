FROM python:3.10.11

WORKDIR /app

#COPY requirements.txt .

RUN pip install --no-deps bertopic
RUN pip install --upgrade numpy hdbscan umap-learn pandas scikit-learn tqdm plotly pyyaml
Run pip install flask


COPY ./model/topic_model ./model/topic_model
COPY server.py server.py
COPY clean_data_subset.csv clean_data_subset.csv

CMD [ "python", "server.py" ]