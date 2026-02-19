
# Ai2 Scholar QA

<img width="1050" alt="image" src="https://github.com/user-attachments/assets/2d7ccc15-2cd8-4316-bec6-ed2a1509f27b" />

<p align="center">
  <a href="https://qa.allen.ai/chat">
    <img alt="Live App" src="https://img.shields.io/badge/Ai2-qa.allen.ai-white?labelColor=teal&color=black">
  </a>  
  <a href="https://www.semanticscholar.org/paper/Ai2-Scholar-QA%3A-Organized-Literature-Synthesis-with-Singh-Chang/c815591e854afb83dc985fa3ff07506d6b25a1b4?utm_source=direct_link">
    <img alt="Paper URL" src="https://img.shields.io/badge/Semantic%20Scholar-white?logo=semanticscholar&labelColor=%231857B6&color=black">
  </a>
  <a href="https://pypi.org/project/ai2-scholar-qa/">
    <img alt="PyPI" src="https://img.shields.io/badge/PyPI-white?logo=PyPI&logoColor=white&labelColor=%233775A9&color=black">
  </a>
    <a href="https://huggingface.co/datasets/allenai/sqa_reranking_eval">
    <img alt="HuggingFace Dataset" src="https://img.shields.io/badge/Dataset-white?logo=Hugging%20Face&logoColor=white&labelColor=%23FFD21E&color=black">
  </a>
    </a>
    <a href="https://www.youtube.com/watch?v=augQU982aGQ&ab_channel=Ai2">
    <img alt="Demo walkthrough" src="https://img.shields.io/badge/Youtube-white?logo=YouTube&logoColor=white&labelColor=%23FF0000&color=black">
  </a>
</p>

#

This repo houses the code for the [live demo](https://scholarqa.allen.ai/) and can be run as local docker containers or embedded into another application as a [python package](https://pypi.org/project/ai2-scholar-qa).




- [Ai2 Scholar QA](#ai2-scholar-qa)
    + [Overview](#overview)
      - [Retrieval:](#retrieval)
      - [Multi-step Generation:](#multi-step-generation)
    + [Setup](#common-setup)
      - [Environment Variables](#environment-variables)
    + [Webapp](#web-app)
      - [Application Configuration](#application-configuration)
      - [docker-compose.yaml](#docker-composeyaml)
      - [Running the Webapp](#running-the-webapp)
      - [Startup](#startup)
      - [UI](#ui)
      - [Backend](#backend)
    + [Async API](#async-api)
    + [Python Package](#python-package)
    + [Custom Pipeline](#custom-pipeline)
      - [API end points](#api-end-points)
      - [ScholarQA class](#scholarqa-class)
      - [Pipeline Components](#pipeline-components)
    + [Citation](#citation)

- ### Overview

![image](https://github.com/user-attachments/assets/f5824b8e-8c9e-4c12-8a40-efb62b9e5e58)
  
Ai2 Scholar QA is a system for answering scientific queries and generating literature reviews by gathering evidence from multiple documents across our corpus (11M+ full text and 100M+ abstracts) and synthesizing an organized report with evidence for each claim. Based on the RAG architecture, Scholar QA has a retrieval component and a three step generator pipeline. 

* #### Retrieval: 
    The retrieval component consists of two sub-components:
  
    i. _Retriever_ - Based on the user query, relevant evidence passages are fetched using the Semantic Scholar public api's snippet/search end point which looks up an index of open source papers. Further, we also use the api's keyword search to suppliement the results from the index with paper abstracts. The user query is first pre-processed to extract metadata for filtering the candidate papers and re-phrasing the query as needed with the help of an LLM - [Prompt](https://github.com/allenai/ai2-scholarqa-lib/blob/345b101e16d1dd62517fbd2df5f2ad6d8065af93/api/scholarqa/llms/prompts.py#L221).

  ii. _Reranker_ - The results from the retriever are then reranked with [mixedbread-ai/mxbai-rerank-large-v1](https://huggingface.co/mixedbread-ai/mxbai-rerank-large-v1) and top k passages are retained and aggregated at the paper-level to combine all the passages from a single paper.
  
These components are encapsulated in the [PaperFinder](https://github.com/allenai/ai2-scholarqa-lib/blob/345b101e16d1dd62517fbd2df5f2ad6d8065af93/api/scholarqa/rag/retrieval.py#L21) class.

* #### Multi-step Generation:
  The generation pipeline uses an LLM (Claude Sonnet 3.7 default) and comprises of three steps:

  i. _Quote Extraction_ - The user query along with the aggregated passages from the retrieval component are used to extract the exact quotes relevant to answering the query - [Prompt](https://github.com/allenai/ai2-scholarqa-lib/blob/345b101e16d1dd62517fbd2df5f2ad6d8065af93/api/scholarqa/llms/prompts.py#L2) .

  ii. _Planning and Clustering_ - First, an organization outline is generated for the report with sections headings and the corresponding format of the section. The quotes from step (i) are clustered and assigned to each heading - [Prompt](https://github.com/allenai/ai2-scholarqa-lib/blob/345b101e16d1dd62517fbd2df5f2ad6d8065af93/api/scholarqa/llms/prompts.py#L52).

  iii. _Summary Generation_ -  Each section is generated based on the quotes assigned to that section and all the prior section text generated in the report - [Prompt](https://github.com/allenai/ai2-scholarqa-lib/blob/345b101e16d1dd62517fbd2df5f2ad6d8065af93/api/scholarqa/llms/prompts.py#L97).
  
  These steps are encapsulated in the [MultiStepQAPipeline](https://github.com/allenai/ai2-scholarqa-lib/blob/345b101e16d1dd62517fbd2df5f2ad6d8065af93/api/scholarqa/rag/multi_step_qa_pipeline.py#L50C7-L50C26) class. For sections that are determined to have a list format, we also generate literature review tables that compare and contrast all papers referenced in that section. We generate these tables using the pipeline proposed by the [ArxivDIGESTables paper](https://arxiv.org/pdf/2410.22360), which is available [here](https://github.com/bnewm0609/arxivDIGESTables/tree/main). 

Both PaperFinder and MultiStepQAPipeline are in turn wrapped inside [ScholarQA](https://github.com/allenai/ai2-scholarqa-lib/blob/41eb8374a88b5edfda7306519a8d61f6c225493f/api/scholarqa/scholar_qa.py#L27), which is the main class powering our system.

  For more info please refer to our [blogpost](allenai.org/blog/ai2-scholarqa).

The code is in this repo can be used as a Dockerized web app, an Async API or as a Python library. We start with the common configuration setup required for both the modes and then describe each mode separately below.

- ### Common Setup

* #### Environment Variables

Ai2 Scholar QA requires Semantic Scholar api and LLMs for its core functionality of retrieval and generation. So please ensure to create a ``.env``  file in the root directory with OR include in your runtime environment directly the following variables:

```
export S2_API_KEY=  
export ANTHROPIC_API_KEY=
export OPENAI_API_KEY=
```

``S2_API_KEY`` : Used to retrieve the relevant [paper passages](https://api.semanticscholar.org/api-docs/#tag/Snippet-Text/operation/get_snippet_search) , [keyword search results](https://api.semanticscholar.org/api-docs/#tag/Paper-Data/operation/get_graph_paper_relevance_search) and [associated metadata](https://api.semanticscholar.org/api-docs/#tag/Paper-Data/operation/post_graph_get_papers) via the Semantic Scholar public api.

``ANTHROPIC_API_KEY`` : Ai2 Scholar QA uses Anthropic's [Claude 3.7 Sonnet](https://www.anthropic.com/news/claude-3-7-sonnet) as the primary LLM for generation, but any model served by litellm from the providers listed [here](https://docs.litellm.ai/docs/completion/json_mode#pass-in-json_schema) will work. Please configure the corresponding api key here.
`OPENAI_API_KEY`: OpenAI's [GPT 4o](https://openai.com/index/gpt-4o-and-more-tools-to-chatgpt-free/) is configured as the fallback llm. 

**Note:** We also use OpenAI's [text moderation api](https://platform.openai.com/docs/guides/moderation/overview#:~:text=The%20moderation%20endpoint%20is%20free%20to%20use%20when%20monitoring%20the%20inputs%20and%20outputs%20of%20OpenAI%20APIs.%20We%20currently%20disallow%20other%20use%20cases.)  to validate and filter harmful queries. If you don't have access to an OpenAI api key, this feature will be disabled.

If you use [Modal](https://modal.com/) to serve your models, please configure `MODAL_TOKEN` and `MODAL_TOKEN_SECRET` here as well.

- ### Web App

* #### Application Configuration
The web app is initialized with a json config outlining the logging and pipeline attributes to be used at runtime. 
Please refer to [default.json](https://github.com/allenai/ai2-scholarqa-lib/blob/main/api/run_configs/default.json) for the default runtime config.
```json
{
  "logs": {
    "log_dir": "logs",
    "llm_cache_dir": "llm_cache",
    "event_trace_loc": "scholarqa_traces",
    "tracing_mode": "local"
  },
  "run_config": {
    "retrieval_service": "public_api",
    "retriever_args": {
      "n_retrieval": 256,
      "n_keyword_srch": 20
    },
    "reranker_service": "modal",
    "reranker_args": {
      "app_name": "ai2-scholar-qa",
      "api_name": "inference_api",
      "batch_size": 256,
      "gen_options": {}
    },
    "paper_finder_args": {
      "n_rerank": 50,
      "context_threshold": 0.0
    },
    "pipeline_args": {
      "validate": true,
      "llm": "anthropic/claude-3-5-sonnet-20241022",
      "decomposer_llm": "anthropic/claude-3-5-sonnet-20241022"
    }
  }
}
```

The config is used to populate the [AppConfig](https://github.com/allenai/ai2-scholarqa-lib/blob/c65c0917b64c501db397e01f34420c7167927da8/api/scholarqa/config/config_setup.py#L47) instance.
It wraps the logging and pipeline instances which are initialized with the config and are outlined below:

**Logging**
```python
class LogsConfig(BaseModel):
    log_dir: str = Field(default="logs", description="Directory to store logs, event traces and litellm cache")
    llm_cache_dir: str = Field(default="llm_cache", description="Sub directory to cache llm calls")
    event_trace_loc: str = Field(default="scholarqa_traces", description="Sub directory to store event traces"
                                                                         "OR the GCS bucket name")
    tracing_mode: Literal["local", "gcs"] = Field(default="local",
                                                  description="Mode to store event traces (local or gcs)")
```
**Note:**

  i. Event Traces are json documents containing a trace of the entire
  pipeline i.e. the results of retrieval, reranking, each step of the qa
  pipeline and associated costs, if any.
 
  ii. llm_cache_dir is used to initialize the local disk cache for caching llm calls via [litellm](https://docs.litellm.ai/docs/caching/all_caches).

  iii. The traces are stored locally in `{log_dir}/{event_trace_loc}` by
  default. They can also be persisted in a Google Cloud Storage (GCS)
  bucket. Please set the `tracing_mode="gcs"` and `event_trace_loc=<GCS
  bucket name>`  here and the `export
  GOOGLE_APPLICATION_CREDENTIALS=<Service Account Key json file path>`
  in .`env`.

  iv. By default, the working directory is `./api` , so the `log_dir` will be created inside it as a sub-directory unless the config is modified.

You can also activate Langsmith based log traces if you have an api key configured.
Please add the following environment variables:

```
LANGCHAIN_API_KEY
LANGCHAIN_TRACING_V2
LANGCHAIN_ENDPOINT
LANGCHAIN_PROJECT
```


**Pipeline**
```python
class RunConfig(BaseModel):  
    retrieval_service: str = Field(default="public_api", description="Service to use for paper retrieval")
    retriever_args: dict = Field(default=None, description="Arguments for the retrieval service")
    reranker_service: str = Field(default="modal", description="Service to use for paper reranking")
    reranker_args: dict = Field(default=None, description="Arguments for the reranker service")
    paper_finder_args: dict = Field(default=None, description="Arguments for the paper finder service")
    pipeline_args: dict = Field(default=None, description="Arguments for the Scholar QA pipeline service")
```

**Note:**

  i. `*(retrieval, reranker)_service` can be used to indicate the type
   of retrieval/reranker you want to instantiate. Ai2 Scholar QA uses the
   `FullTextRetriever` and `ModalReranker` respectively, which are chosen based on the
   default  `public_api` and `modal` config values respectively. To choose a
   SentenceTransformers reranker, replace `modal` with `crossencoder` or
   `biencoder` or define your own types.
 
  ii. `*(retriever, reranker, paper_finder, pipeline)_args` are used to
   initialize the corresponding instances of the pipeline components. eg.
   ``retriever = FullTextRetriever(**run_config.retriever_args)``. You
   can initialize multiple runs and customize your pipeline.

  iii.  If the `reranker_args` are not defined, the app resorts to using only the retrieval service.

* #### docker-compose.yaml
The web app initializes 4 docker containers - one each for the API, GUI, nginx proxy and sonar with their own Dockerfiles.
The api container config can also be used to declare environment variables - 
```yaml
api:  
build: ./api  
volumes:  
- ./api:/api  
- ./secret:/secret  
environment:  
# This ensures that errors are printed as they occur, which  
# makes debugging easier.  
- PYTHONUNBUFFERED=1  
- LOG_LEVEL=INFO
- CONFIG_PATH=run_configs/default.json  
ports:  
- 8000:8000  
env_file:  
- .env
```
`environment.CONFIG_PATH` indicates the path of the application configuration json file.
`env_file` indicates the path of the file with environment variables.

- ### Running the Webapp

Please refer to [DOCKER.md](https://github.com/allenai/ai2-scholarqa-lib/blob/main/docs/DOCKER.md)
for more info on setting up the docker app.

i. Clone the repo
```bash
git clone git@github.com:allenai/ai2-scholarqa-lib.git
cd ai2-scholarqa-lib
```
ii. Run docker-compose
```bash
docker compose up --build
```

The docker compose command takes a while to run the first time to install torch and related dependencies.
You can get the verbose output with the following command:

```bash
docker compose build --progress plain
```
Below we show videos of app startup, the UI and also backend logging while processing a user query.

#### Startup

https://github.com/user-attachments/assets/7d6761d6-1e95-4dac-9aeb-a5a898a89fbe

#### UI

https://github.com/user-attachments/assets/baed8710-2161-4fbf-b713-3a2dcf46ac61

#### Backend

https://github.com/user-attachments/assets/f9a1b39f-36c8-41c4-a0ac-10046ded0593

- ### Async API
The Ai2 Scholar QA UI is powered by an async api at the back end in [app.py](https://github.com/allenai/ai2-scholarqa-lib/blob/main/api/scholarqa/app.py) which is run from [dev.sh](https://github.com/allenai/ai2-scholarqa-lib/blob/main/api/dev.sh).

i. The `query_corpusqa` end point is first called with the `query`, and a uuid as the `user_id`, adn it returns a `task_id`.

<img width="1421" alt="image" src="https://github.com/user-attachments/assets/3b5792f0-04f9-4dbf-a704-d98beaf6e58b" />

<img width="964" alt="image" src="https://github.com/user-attachments/assets/6cbb4d38-f1f4-4444-9b2d-4139ca28c514" />

ii. Subsequently, the `query_corpusqa` is then polled to get the updated status of the async task until the task status is not `COMPLETED`

[Sample response](https://github.com/allenai/ai2-scholarqa-lib/blob/main/docs/sample_response) 


- ### Python Package

```python
conda create -n scholarqa python=3.11.3
conda activate scholarqa
pip install ai2-scholar-qa

#to use sentence transformer models as re-ranker
pip install 'ai2-scholar-qa[all]'
```

Both the webapp and the api are powered by the same pipeline represented by the [ScholarQA](https://github.com/allenai/ai2-scholarqa-lib/blob/main/api/scholarqa/scholar_qa.py) class. The pipeline consists of a retrieval component, `PaperFinder` which consists of a retriever and maybe a reranker and a 3 step generator component `MultiStepQAPipeline`. Each component is extensible and can be replaced by custom instances/classes as required.

**Sample usage**

```python
from scholarqa.rag.reranker.reranker_base import CrossEncoderScores
from scholarqa.rag.reranker.modal_engine import ModalReranker
from scholarqa.rag.retrieval import PaperFinderWithReranker
from scholarqa.rag.retriever_base import FullTextRetriever
from scholarqa import ScholarQA
from scholarqa.llms.constants import CLAUDE_37_SONNET

#Retrieval class/steps
retriever = FullTextRetriever(n_retrieval=256, n_keyword_srch=20) #full text and keyword search
reranker = CrossEncoderScores(model_name_or_path="mixedbread-ai/mxbai-rerank-large-v1") #sentence transformer


#Reranker if deployed on Modal, modal_app_name and modal_api_name are modal specific arguments.
#Please refer https://github.com/allenai/ai2-scholarqa-lib/blob/aps/readme_fixes/docs/MODAL.md for more info 
reranker = ModalReranker(app_name='<modal_app_name>', api_name='<modal_api_name>', batch_size=256, gen_options=dict())

#wraps around the retriever with `retrieve_passages()` and `retrieve_additional_papers()`, and reranker with rerank()
#any modifications to the retrieval output can be made here
paper_finder =  PaperFinderWithReranker(retriever, reranker, n_rerank=50, context_threshold=0.0)

#For wrapper class with MultiStepQAPipeline integrated
scholar_qa = ScholarQA(paper_finder=paper_finder, llm_model=CLAUDE_37_SONNET) #llm_model can be any litellm model
print(scholar_qa.answer_query("Which is the 9th planet in our solar system?"))
```

**Pipeline steps (Modular usage)**

Continuing from sample usage, below is a breakdown of the pipeline execution in the ScholarQA class.

```python
from scholarqa import ScholarQA
from scholarqa.rag.multi_step_qa_pipeline import MultiStepQAPipeline
from scholarqa.llms.constants import CLAUDE_37_SONNET
from scholarqa.llms.prompts import SYSTEM_PROMPT_QUOTE_PER_PAPER, SYSTEM_PROMPT_QUOTE_CLUSTER, PROMPT_ASSEMBLE_SUMMARY
from scholarqa.utils import NUMERIC_META_FIELDS, CATEGORICAL_META_FIELDS

# Custom MultiStepQAPipeline class/steps with llm_model asa any litellm supported model
mqa_pipeline = MultiStepQAPipeline(llm_model=CLAUDE_37_SONNET)

query = "Which is the 9th planet in our solar system?"

scholar_qa = ScholarQA(paper_finder=paper_finder, multi_step_pipeline=mqa_pipeline, llm_model=CLAUDE_37_SONNET)

# Decompose the query to get filters like year, venue, fos, citations, etc along with
# a re-written version of the query and a query suitable for keyword search.
llm_processed_query = scholar_qa.preprocess_query(query)

# Paper finder step - retrieve relevant paper passages from semantic scholar index and api
full_text_src, keyword_srch_res = scholar_qa.find_relevant_papers(llm_processed_query.result)
retrieved_candidates = full_text_src + keyword_srch_res

# Rerank the retrieved candidates based on the query with a cross encoder
# keyword search results are returned with associated metadata, metadata is retrieved separately for full text serach results
keyword_srch_metadata = [
    {k: v for k, v in paper.items() if k == "corpus_id" or k in NUMERIC_META_FIELDS or k in CATEGORICAL_META_FIELDS}
    for paper in keyword_srch_res]
reranked_df, paper_metadata = scholar_qa.rerank_and_aggregate(query, retrieved_candidates,
                                                              filter_paper_metadata={str(paper["corpus_id"]): paper for
                                                                                     paper in
                                                                                     keyword_srch_metadata})
# Step 1 - quote extraction
per_paper_quotes = scholar_qa.step_select_quotes(query, reranked_df, sys_prompt=SYSTEM_PROMPT_QUOTE_PER_PAPER)

# step 2: outline planning and clustering
cluster_json = scholar_qa.step_clustering(query, per_paper_quotes.result, sys_prompt=SYSTEM_PROMPT_QUOTE_CLUSTER)

# Changing to expected format in the summary generation prompt
plan_json = {f'{dim["name"]} ({dim["format"]})': dim["quotes"] for dim in cluster_json.result["dimensions"]}

# step 2.1: extend the clustered snippets in plan json with their inline citations
per_paper_summaries_extd = scholar_qa.extract_quote_citations(reranked_df, per_paper_quotes.result, plan_json,
                                                              paper_metadata)

# step 3: generating output as per the outline
answer = list(scholar_qa.step_gen_iterative_summary(query, per_paper_summaries_extd, plan_json,
                                                    sys_prompt=PROMPT_ASSEMBLE_SUMMARY))
```

- ### Custom Pipeline
* #### API end points
  The api end points in app.py can be extended with a fastapi APIRouter in another script.
  eg. `custom_app.py`
  
  ```python
  from fastapi import APIRouter, FastAPI
  from scholarqa.app import create_app as create_app_base
  from scholarqa.app import app_config
  from scholarqa.models import ToolRequest
  
  def create_app() -> FastAPI:
    app = create_app_base()
    custom_router = APIRouter()

    @custom_router.post("/retrieval")
    def retrieval(tool_request: ToolRequest, task_id: str):
      scholar_qa = app_config.load_scholarqa(task_id)
      #a re-written version of the query and a query suitable for keyword search.
      llm_processed_query = scholar_qa.preprocess_query(tool_request.query)
      full_text_src, keyword_srch_res = scholar_qa.find_relevant_papers(llm_processed_query.result)
      retrieved_candidates = full_text_src + keyword_srch_res
      return retrieved_candidates

    app.include_router(custom_router)
    return app.py
  ```

  To run `custom_app.py`, simply replace `scholarqa.app:create_app` in dev.sh with `<package>.custom_app:create_app`

* #### ScholarQA class
  To extend the existing ScholarQA functionality in a new class you can either create a sub class of ScholarQA or a new class altogether.
  Either way, `lazy_load_scholarqa` in app.py should be reimplemented in the new api script to ensure the correct class is initialized.

* #### Pipeline Components
  The components of the pipeline are individually extensible.
  We have the following abstract classes that can be extended to achieve desired customization for retrieval:
  
  - [AbstractRetriever](https://github.com/allenai/ai2-scholarqa-lib/blob/41eb8374a88b5edfda7306519a8d61f6c225493f/api/scholarqa/rag/retriever_base.py#L10)
  - [AbstractReranker](https://github.com/allenai/ai2-scholarqa-lib/blob/41eb8374a88b5edfda7306519a8d61f6c225493f/api/scholarqa/rag/reranker/reranker_base.py#L15)
  - [AbsPaperFinder](https://github.com/allenai/ai2-scholarqa-lib/blob/41eb8374a88b5edfda7306519a8d61f6c225493f/api/scholarqa/rag/retrieval.py#L14)

  and the MultiStepQAPipeline can be extended/modified as needed for generation.
  
* #### Modal deployment
  If you would prefer to serve your models via modal, please refer to [MODAL.md](https://github.com/allenai/ai2-scholarqa-lib/blob/main/docs/MODAL.md) for more info and sample code that we used to deploy the reranker model in the live demo.

- ### Citation
  Please cite the work as follows:
  ```bibtex
  @inproceedings{Singh2025Ai2SQ,
  title={Ai2 Scholar QA: Organized Literature Synthesis with Attribution},
  author={Amanpreet Singh and Joseph Chee Chang and Chloe Anastasiades and Dany Haddad and Aakanksha Naik and Amber Tanaka and Angele Zamarron and Cecile Nguyen and Jena D. Hwang and Jason Dunkleberger and Matt Latzke and Smita Rao and Jaron Lochner and Rob Evans and Rodney Kinney and Daniel S. Weld and Doug Downey and Sergey Feldman},
  year={2025},
  url={https://api.semanticscholar.org/CorpusID:277786810}
  ```

