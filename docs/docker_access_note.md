# Docker Access Note

On the implementation host used for the initial commits, Docker CLI and Compose were installed, but Docker daemon access failed:

```text
permission denied while trying to connect to the docker API at unix:///var/run/docker.sock
```

`sudo docker info` also required an interactive password. Compose configuration validation can run without daemon access, but service startup, migrations against live PostgreSQL, Qdrant ingestion, and end-to-end retrieval tests require a Docker-enabled session.
