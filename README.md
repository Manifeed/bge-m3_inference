# Embedding Service

Service autonome d'embeddings GPU pour `bge-m3`, expose une API compatible OpenAI sur `POST /v1/embeddings`.

## Prerequis

- Docker avec support GPU NVIDIA
- `docker compose`

## Demarrage

```bash
cp .env.example .env
make up
```

Le service ecoute par defaut sur `http://localhost:8000`.

## Commandes utiles

```bash
make build
make up
make down
make clean
make test
```

## Configuration

Variables principales:

- `EMBEDDING_SERVICE_API_KEY`: cle Bearer obligatoire
- `EMBEDDING_SERVICE_PORT`: port publie localement
- `SOURCE_EMBEDDING_MODEL_PATH`: chemin du modele dans le conteneur
- `NVIDIA_VISIBLE_DEVICES`: GPU exposes au conteneur
- `NVIDIA_DRIVER_CAPABILITIES`: capacites NVIDIA activees

## API

### Requete

```json
{
  "model": "bge-m3",
  "input": [
    "Docker compose",
    "Kubernetes ingress",
    "Fedora firewall"
  ],
  "dense": true,
  "sparse": true
}
```

### Reponse

```json
{
  "data": [
    {
      "index": 0,
      "embedding": [0.1, 0.2],
      "sparse_embedding": {
        "indices": [12, 24],
        "values": [0.8, 0.4]
      }
    }
  ]
}
```

### Exemple curl

```bash
curl http://localhost:8000/v1/embeddings \
  -H "Authorization: Bearer ${EMBEDDING_SERVICE_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bge-m3",
    "input": ["Docker compose", "Kubernetes ingress"],
    "dense": true,
    "sparse": true
  }'
```

## Endpoints internes

- `GET /internal/health`: liveness
- `GET /internal/ready`: verifie la config et le chargement du modele

## Notes

- Le build telecharge le modele dans l'image. Le premier build est donc lourd.
- Le service refuse de demarrer sans GPU CUDA detecte.
- Ce dossier est autonome et ne depend pas des fichiers du dossier `infra/`.
# embedding_service
