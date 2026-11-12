import httpx
import numpy as np


def _to_list(embedding: np.ndarray | list[float]) -> list[float]:
    if isinstance(embedding, np.ndarray):
        return embedding.tolist()
    return list(embedding)


class VoiceServiceClient:
    """Асинхронный HTTP-клиент. Один экземпляр на процесс (connection pool)."""

    def __init__(self, base_url: str, timeout: float = 60.0) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
        )

    async def identify(
        self,
        embedding: np.ndarray | list[float],
        threshold: float,
    ) -> dict:
        r = await self._client.post(
            "/api/v1/voices/identify",
            json={"embedding": _to_list(embedding)},
        )
        r.raise_for_status()
        return r.json()

    async def register_new_person(
        self,
        embedding: np.ndarray | list[float],
        first_name: str,
        last_name: str,
        middle_name: str | None = None,
    ) -> dict:
        payload: dict = {
            "embedding": _to_list(embedding),
            "first_name": first_name,
            "last_name": last_name,
        }
        if middle_name is not None:
            payload["middle_name"] = middle_name
        r = await self._client.post("/api/v1/voices", json=payload)
        r.raise_for_status()
        return r.json()

    async def close(self) -> None:
        await self._client.aclose()