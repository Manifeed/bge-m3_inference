# BGE-M3 Inference Service

GPU-backed standalone inference service for `bge-m3`. It exposes an OpenAI-like
API on `POST /v1/embeddings` and can return dense, sparse, and ColBERT vectors.

## Requirements

- Docker with NVIDIA GPU support
- `docker compose`
- NVIDIA driver loaded and visible from the host (`nvidia-smi` must work)

## Quick Start

```bash
cp .env.example .env
make up
```

The service listens on `http://localhost:8000` by default.

## Common Commands

```bash
make build
make up
make down
make clean
make test
make benchmark
```

`make benchmark` suppose que le service tourne deja localement. Par defaut, la
cible envoie un benchmark dense a `http://127.0.0.1:8000` en reutilisant
`BGE_M3_INFERENCE_API_KEY`. La cible recharge aussi automatiquement les
variables de `.env` si le fichier existe. Elle attend aussi que
`/internal/ready` reponde `200` avant de lancer les requetes de benchmark.

```bash
BGE_M3_INFERENCE_API_KEY=dev-secret make benchmark
BENCHMARK_EXTRA_ARGS="--dense --sparse --repeats 10" BGE_M3_INFERENCE_API_KEY=dev-secret make benchmark
BENCHMARK_EXTRA_ARGS="--payload-file /tmp/benchmark_payload.json" BGE_M3_INFERENCE_API_KEY=dev-secret make benchmark
```

Sur un shell interactif, `BGE_M3_INFERENCE_API_KEY=change-me` saisi seul ne
rend pas toujours la variable visible pour `make`. Utilise soit `export
BGE_M3_INFERENCE_API_KEY=change-me`, soit `.env`, soit la forme inline
`BGE_M3_INFERENCE_API_KEY=change-me make benchmark`.

## Configuration

Canonical environment variables:

- `BGE_M3_INFERENCE_API_KEY`: required Bearer token
- `BGE_M3_INFERENCE_PORT`: published local port
- `BGE_M3_INFERENCE_CONTAINER_NAME`: local container name
- `SOURCE_EMBEDDING_MODEL_PATH`: model path inside the container
- `BGE_M3_INFERENCE_BATCH_MAX_ITEMS`: max queued inputs per GPU batch
- `BGE_M3_INFERENCE_BATCH_MAX_TOKENS`: max estimated tokens per GPU batch
- `BGE_M3_INFERENCE_BATCH_MAX_WAIT_MS`: queue flush delay before dispatch
- `BGE_M3_INFERENCE_QUEUE_MAX_ITEMS`: max queued inputs before returning `503 overloaded`
- `BGE_M3_INFERENCE_MAX_LENGTH`: tokenizer truncation length
- `BGE_M3_INFERENCE_REQUEST_TIMEOUT_SECONDS`: max time a request can wait for batch completion
- `BGE_M3_INFERENCE_SHUTDOWN_GRACE_SECONDS`: max graceful shutdown wait for worker and warmup threads
- `BGE_M3_INFERENCE_USE_FP16`: enable FP16 model loading
- `NVIDIA_VISIBLE_DEVICES`: exposed GPU devices
- `NVIDIA_DRIVER_CAPABILITIES`: enabled NVIDIA capabilities
- `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE`: force offline runtime loading from the pre-baked image.
  They are intended for container runtime; the image build itself still downloads the model first.

Legacy compatibility aliases are still accepted for one transition cycle:

- `EMBEDDING_SERVICE_API_KEY`
- `EMBEDDING_SERVICE_PORT`
- `EMBEDDING_SERVICE_CONTAINER_NAME`

## API

### Request

All inference modes are opt-in. If `dense`, `sparse`, and `colbert` are all
omitted or `false`, the request is rejected. Empty input lists and blank
strings are also rejected with `422`. Leading and trailing whitespace is
trimmed, but internal spaces and line breaks are preserved.

```json
{
  "model": "bge-m3",
  "input": [
    "Docker compose",
    "Kubernetes ingress",
    "Fedora firewall"
  ],
  "dense": true,
  "sparse": true,
  "colbert": false
}
```

### Response

```json
{
  "data": [
    {
      "index": 0,
      "embedding": [0.1, 0.2],
      "sparse_embedding": {
        "indices": [12, 24],
        "values": [0.8, 0.4]
      },
      "colbert_embedding": null
    }
  ]
}
```

### ColBERT Example

```json
{
  "model": "bge-m3",
  "input": ["hybrid semantic search"],
  "dense": false,
  "sparse": false,
  "colbert": true
}
```

### `curl` Example

```bash
curl http://localhost:8000/v1/embeddings \
  -H "Authorization: Bearer ${BGE_M3_INFERENCE_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bge-m3",
    "input": ["Docker compose", "Kubernetes ingress"],
    "dense": true,
    "sparse": true,
    "colbert": false
  }'
```

## Internal Endpoints

- `GET /internal/health`: liveness probe
- `GET /internal/ready`: verifies API key config, runtime warmup, worker
  liveness, and current queue capacity

## Failure Modes

- `503 warming_up`: runtime is still loading the tokenizer/model
- `503 overloaded`: bounded queue is full, so new work is rejected
- `503 stopping`: graceful shutdown has started and new work is refused
- `503 request_timeout`: the request exceeded the internal serving timeout
- `503 embedding_model_unavailable`: runtime load or encode failed
- `503 embedding_service_misconfigured`: API key configuration is missing

## Performance Tuning

- Increase `BGE_M3_INFERENCE_BATCH_MAX_ITEMS` to improve throughput on short
  inputs when GPU memory allows it.
- Increase `BGE_M3_INFERENCE_BATCH_MAX_TOKENS` to batch more medium-length
  inputs together.
- Increase `BGE_M3_INFERENCE_QUEUE_MAX_ITEMS` only if your upstream services
  tolerate longer queueing under load.
- Lower `BGE_M3_INFERENCE_MAX_LENGTH` to improve tokens/sec when long-document
  recall is less important.
- Tune `BGE_M3_INFERENCE_REQUEST_TIMEOUT_SECONDS` to stay below or aligned with
  upstream HTTP client timeouts.
- ColBERT batches are automatically capped more aggressively because the output
  tensor is heavier than dense or sparse-only inference.
- If CUDA runs out of memory, the runtime downshifts the active batch size and
  retries once before returning `503`.

## Notes

- The image build downloads the model, so the first build is large.
- Startup warmup loads both the tokenizer and the BGE-M3 model once per
  process. Expect `/internal/ready` to stay in `503 warming_up` until that
  completes.
- The service refuses to serve inference without a CUDA-capable GPU.
- `/internal/ready` no longer runs a probe encode on every call. Warmup happens
  once per process lifecycle.
- `scripts/benchmark_inference.py` now reports `mean`, `p50`, `p95`, and
  estimated tokens/sec so you can compare dense+sparse and ColBERT profiles
  before and after tuning.
