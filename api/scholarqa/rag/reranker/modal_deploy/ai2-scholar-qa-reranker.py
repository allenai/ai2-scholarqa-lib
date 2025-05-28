# ## Setup

import os
import threading
import time
from typing import List
import modal

MODEL_NAME = "mixedbread-ai/mxbai-rerank-large-v1"
MODEL_DIR = f"/root/models/{MODEL_NAME}"
GPU_CONFIG = "L4:1"

APP_NAME = "<modal_app_name>"
APP_LABEL = APP_NAME.lower()


# ## Define a container image
#
# We want to create a Modal image which has the model weights pre-saved to a directory. The benefit of this
# is that the container no longer has to re-download the model from Huggingface - instead, it will take
# advantage of Modal's internal filesystem for faster cold starts.
#
# ### Download the weights
#
# We can download the model to a particular directory using the HuggingFace utility function `snapshot_download`.
#
# If you adapt this example to run another model,
# note that for this step to work on a [gated model](https://huggingface.co/docs/hub/en/models-gated)
# the `HF_TOKEN` environment variable must be set and provided as a [Modal Secret](https://modal.com/secrets).
#
# This can take some time -- at least a few minutes.

def download_model_to_image(model_dir, model_name):
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    os.makedirs(model_dir, exist_ok=True)

    snapshot_download(
        model_name,
        local_dir=model_dir,
    )
    move_cache()


# ### Image definition
#
# We’ll start from a basic Linux container image, install related libraries,
# and then use `run_function` to run the function defined above and ensure the weights of
# the model are saved within the container image.

reranker_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.3.1",
        "transformers==4.44.1",
        "sentencepiece",
        "hf-transfer==0.1.6",
        "huggingface_hub==0.23.4",
        "sentence-transformers==3.0.1",
        "peft==0.13.2"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        download_model_to_image,
        timeout=60 * 20,
        # secrets=[modal.Secret.from_name("chrisn-wildguard-hf-gated-read")],
        kwargs={
            "model_dir": MODEL_DIR,
            "model_name": MODEL_NAME,
        },
    )
    .add_local_python_source("custom_cross_encoder")
)

with reranker_image.imports():
    import torch
    from sentence_transformers import CrossEncoder
    from custom_cross_encoder import PaddedCrossEncoder
    from typing import List

app = modal.App(APP_NAME)


# ## The model class
#
# The inference function is best represented with Modal's [class syntax](/docs/guide/lifecycle-functions) and the `@enter` decorator.
# This enables us to load the model into memory just once every time a container starts up, and keep it cached
# on the GPU for each subsequent invocation of the function.
#
@app.cls(
    gpu=GPU_CONFIG,
    timeout=60 * 20,
    scaledown_window=60 * 20,
    min_containers=2,
    image=reranker_image
)
@modal.concurrent(max_inputs=1)
class Model:
    @modal.enter()
    def start_engine(self):
        print("🥶 cold starting inference")
        start = time.monotonic_ns()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.reranker_compiled = PaddedCrossEncoder(
            MODEL_DIR,
            automodel_args={"torch_dtype": "float16"},
            trust_remote_code=True,
            device=device,
        )
        self.compiled_model = torch.compile(self.reranker_compiled.model, dynamic=True)
        self.reranker_compiled.model = self.compiled_model

        self.reranker_torch = CrossEncoder(
            MODEL_DIR,
            automodel_args={"torch_dtype": "float16"},
            trust_remote_code=True,
            device=device,
        )

        self.compiled_flag = False
        self.compiling_thread = None
        duration_s = (time.monotonic_ns() - start) / 1e9
        print(f"🏎️ engine started in {duration_s:.0f}s")

    def compile_reranker(self, batch_size):
        print("compiling cross encoder model")
        sentence_pairs = [["This is the query entered by the user.", f"This is candidate # {i}"] for i in range(batch_size)]
        for _ in range(2):
            self.reranker_compiled.predict(sentence_pairs, convert_to_tensor=True, show_progress_bar=True, batch_size=batch_size)
        self.compiled_flag = True
        print("compilation done!")

    @modal.method()
    def get_scores(self, query: str, passages: List[str], batch_size: int) -> List[float]:
        sentence_pairs = [[query, passage] for passage in passages]
        try:
            if self.compiled_flag:
                print("reranking with compiled model")
                scores = self.reranker_compiled.predict(sentence_pairs, convert_to_tensor=True, show_progress_bar=True,
                                               batch_size=batch_size).tolist()
            else:
                print("reranking with torch model")
                scores = self.reranker_torch.predict(sentence_pairs, convert_to_tensor=True, show_progress_bar=True,
                                                     batch_size=16).tolist()
                if self.compiling_thread is None:
                    self.compiling_thread = threading.Thread(target=self.compile_reranker, args=(batch_size,))
                    self.compiling_thread.start()
        except Exception as e:
            print(e)
            scores = self.reranker_torch.predict(sentence_pairs, convert_to_tensor=True, show_progress_bar=True,
                                           batch_size=batch_size).tolist()
        return [float(s) for s in scores]


# ## Coupling a frontend web application
#
# We can stream inference from a FastAPI backend, also deployed on Modal.


api_image = (
    modal.Image.debian_slim(python_version="3.11")
)


@app.function(
    image=api_image,
    min_containers=1,
    timeout=60 * 10,
)
@modal.concurrent(max_inputs=20)
async def inference_api(query: str, passages: List[str], batch_size: int = 512) -> List[float]:
    model = Model()
    return model.get_scores.remote(query, passages, batch_size)


@app.local_entrypoint()
def main():
    start = time.time()
    s = inference_api.remote("What is the python package infinity_emb?",
                             ["This is a document not related to the python package infinity_emb, hence...",
                              "Paris is in France!",
                              "infinity_emb is a package for sentence embeddings and rerankings using transformer models in Python!"],
                             64)
    print(s)
    print("Time taken:", time.time() - start)
