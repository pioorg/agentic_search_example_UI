# Agentic RAG Demo 

This repo contains a python script where the Elastic python client is integrated with a streamlit UI - To showcase an example of agentic search using LangChain. 

This project makes use of a GPT-4o-Mini deployed on Azure OpenAI, as well as the Google Custom Search API, and an Elastic Cloud deployment to hold data. 

To use this code directly, create a ```.env``` file and fill it with the following variables:

```bash
ELASTIC_ENDPOINT=<ELASTIC CLOUD ENDPOINT>
ELASTIC_API_KEY=<ELASTIC CLOUD API KEY>

# Enable custom search API
# https://developers.google.com/custom-search/v1/introduction/?apix=true
GCP_API_KEY=<GCP API KEY>
GCP_PSE_ID=<GCP PSE ID>


AZURE_OPENAI_SYSTEM_PROMPT="You are a helpful assistant. Be as concise and efficient as possible. Convey maximum meaning in fewest words possible."


AZURE_OPENAI_ENDPOINT='<AZURE ENDPOINT>'
AZURE_OPENAI_API_VERSION='<AZURE API VERSION>'
AZURE_OPENAI_API_KEY=<AZURE API KEY>
AZURE_OPENAI_MODEL='gpt-4o-mini'
```

This code requires a connection to the Google Custom Search API and to an Elastic Cloud deployment. The agent's tools are API calls to either the Google CSAPI or to specific Elasticsearch indices - The latter is a semantic_search over indices containing PDF files processed using the `semantic_text` datatype. 

```python
class ElasticSearcher:
    def __init__(self):
        self.client = Elasticsearch(
            os.environ.get("ELASTIC_ENDPOINT"),
            api_key=os.environ.get("ELASTIC_API_KEY")
        )
    
    def search(self, query, index="us_navy_dive_manual", size=10):
        response = self.client.search(
            index=index,
            body={
                "query": {
                    "semantic": {
                        "field": "semantic_content",
                        "query": query
                    }
                }     
            },
            size=size
        )
        return "\n".join([hit["_source"].get("body", "No Body") 
                            for hit in response["hits"]["hits"]])
```


This [tutorial](https://www.elastic.co/docs/solutions/search/semantic-search/semantic-search-semantic-text) contains all the necessary steps for creating a semantic search index - I highly recommend checking it out! Once the indices are created, simplify modify the `elastic.search` call in `tools` to call the specific index name. 

```
tools = [
    Tool(
        name="WebSearch",
        func=lambda q: googler.search(q, n=3),
        description="Search the web for information. Use for current events or general knowledge or to complement with additional information."
    ),
    Tool(
        name="NavyDiveManual",
        func=lambda q: elastic.search(q, index="us_navy_dive_manual"),
        description="Search the Operations Dive Manual. Use for diving procedures, advanced or technical operational planning, resourcing, and technical information."
    ),
    Tool(
        name="DivingSafetyManual",
        func=lambda q: elastic.search(q, index="diving_safety_manual"),
        description="Search the Diving Safety Manual. Use for generic diving safety protocols and best practices."
    )
]
```


To install dependencies, set up a [virtual env](https://docs.python.org/3/library/venv.html), then running `pip install -r requirements.txt` to install dependencies. 

Once set up, navigate to the project folder and use the command `streamlit run agentic_tooling.py` to start the UI. 


## Ingestion and Processing 

The data sources are in PDF format, so the next step was to ingest them into an Elastic Cloud deployment. I set-up this python script using Elastic's `bulk` API to upload documents to Elastic Cloud:

```python
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from elasticsearch import Elasticsearch, helpers # elasticsearch==8.14.0
from tqdm import tqdm # tqdm==4.66.4
from llama_index.core import SimpleDirectoryReader

def bulk_upload_to_elasticsearch(data, index_name, es, batch_size=500, max_workers=10):
    ''' 
    data: [ {document} ]
        document: {
                    "_id": str
                    ...
                  }
    index_name: str 
    es: Elasticsearch 
    batch_size: int 
    max_workers: int
    '''
    total_documents = len(data)
    success_bar = tqdm(total=total_documents, desc="Successful uploads", colour="green")
    failed_bar = tqdm(total=total_documents, desc="Failed uploads", colour="red")

    def create_action(doc):
        '''
        Define upload action from source documents
        '''
        return {
            "_index": index_name,
            "_id": doc["id_"],
            "body": doc["text"]
        }

    def read_and_create_batches(data):
        ''' 
        Yield document batches
        '''
        batch = []
        for doc in data:
            batch.append(create_action(doc))
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def upload_batch(batch):
        ''' 
        Make bulk call for batch
        '''
        try:
            success, failed = helpers.bulk(es, batch, raise_on_error=False, request_timeout=45)
            if isinstance(failed, list):
                failed = len(failed)
            return success, failed
        except Exception as e:
            print(f"Error during bulk upload: {str(e)}")
            return 0, len(batch)

    ''' 
    Parallel execution of batch upload
    '''
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {executor.submit(upload_batch, batch): batch for batch in read_and_create_batches(data)}
        for future in as_completed(future_to_batch):
            success, failed = future.result()
            success_bar.update(success)
            failed_bar.update(failed)

    ''' 
    Update progress bars
    '''
    total_uploaded = success_bar.n
    total_failed = failed_bar.n
    success_bar.close()
    failed_bar.close()

    return total_uploaded, total_failed

# This is connecting to ES Cloud via credentials stored in .env 
# May have to change this to suit your env. 
try:
    es_endpoint = os.environ.get("ELASTIC_ENDPOINT")
    es_client = Elasticsearch(
        es_endpoint,
        api_key=os.environ.get("ELASTIC_API_KEY")
    )
except Exception as e:
    es_client = None

print(es_client.ping())
```

After downloading the US Navy Dive Manual PDF and storing it in its own folder, I use [LlamaIndex's](https://www.llamaindex.ai/) `SimpleDirectoryReader` to load the PDF data, then trigger a bulk upload:

```python 
reader = SimpleDirectoryReader(input_dir="./data")
documents = reader.load_data()
bulk_upload_to_elasticsearch([i.to_dict() for i in list(documents)], 
                            "us_navy_dive_manual_raw", 
                            es_client, batch_size=16, max_workers=10)
```

## Semantic Data Embedding and Chunking 

On Elastic Cloud Serverless, the [ELSER v2](https://www.elastic.co/docs/explore-analyze/machine-learning/nlp/ml-nlp-elser) endpoint is pre-provisioned. You do not need to create a custom inference with your own id. Instead, use the built-in inference id ".elser-2-elasticsearch" (service "elasticsearch", model_id ".elser_model_2_linux-x86_64"). You can confirm it exists via DevTools:

```bash
GET _inference

GET _inference/sparse_embedding/.elser-2-elasticsearch
```
I then define a simple pipeline. Each document stores the text of a page from the dive manual in the `body` field, so I copy the contents of `body` to a field called `semantic_content`.

```bash 
PUT _ingest/pipeline/diving_pipeline
{
  "processors": [
    {
      "set": {
        "field": "semantic_content",
        "ignore_empty_value": true,
        "copy_from": "body"
      }
    }
  ]
}
```

I then create a new index called `us_navy_dive_manual`, and set `semantic_content` as a `semantic_text` field: 

```bash 
PUT us_navy_dive_manual
{
  "mappings": {
    "properties": {
      "semantic_content": {
        "type": "semantic_text",
        "inference_id": ".elser-2-elasticsearch"
      }
    }
  }
}
```

I then trigger a reindex job. Now the data will flow from `us_navy_dive_manual_raw`, be chunked and embedded using ELSER, and be reindexed into `us_navy_dive_manual` ready for use. 

```bash 
POST _reindex?slices=auto&wait_for_completion=false
{
  "source": {
    "index": "us_navy_dive_manual_raw",
    "size": 4
  },
  "dest": {
    "index": "us_navy_dive_manual",
    "pipeline": "diving_pipeline"
  },
  "conflicts": "proceed"
}
```

Tip: The helper script `setup_indices.py` automates the ingest pipeline and index creation for you and uses the preconfigured inference id `.elser-2-elasticsearch` by default:

```
python setup_indices.py
```

You can also use `seed_data.py` to load sample data into the indices:

```
python seed_data.py
```
