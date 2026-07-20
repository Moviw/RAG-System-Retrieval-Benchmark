import socket

import pytest


def port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) == 0


requires_postgres = pytest.mark.skipif(
    not port_open("127.0.0.1", 5432),
    reason="PostgreSQL is not running locally",
)
requires_qdrant = pytest.mark.skipif(
    not port_open("127.0.0.1", 6333),
    reason="Qdrant is not running locally",
)


@requires_postgres
def test_postgres_service_is_available() -> None:
    assert port_open("127.0.0.1", 5432)


@requires_qdrant
def test_qdrant_service_is_available() -> None:
    assert port_open("127.0.0.1", 6333)
