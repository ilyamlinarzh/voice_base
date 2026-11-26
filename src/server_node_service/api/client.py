import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any

import httpx

from config import TrueConfSettings
from .schemas import CallDetails, CallRecord, CallSummary

logger = logging.getLogger(__name__)


class TrueConfAPIError(Exception):
    """Базовое исключение при работе с TrueConf Server API."""

    def __init__(
        self, message: str, status_code: int | None = None, payload: Any = None
    ):
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload


class TrueConfClient:
    def __init__(self, settings: TrueConfSettings):
        self._settings = settings
        self._client: httpx.AsyncClient | None = None

        self._access_token: str | None = settings.static_token or None
        self._token_expires_at: float = float("inf") if settings.static_token else 0.0
        self._token_lock = asyncio.Lock()

    async def __aenter__(self) -> "TrueConfClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    async def connect(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._settings.api_base_url,
                verify=self._settings.verify_ssl,
                timeout=httpx.Timeout(self._settings.request_timeout),
                headers={"Accept": "application/json"},
            )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError(
                "TrueConfClient не инициализирован. Используйте 'async with TrueConfClient(...)'."
            )
        return self._client

    async def _get_access_token(self, force_refresh: bool = False) -> str:
        """Возвращает валидный Bearer-токен, обновляя его по OAuth 2.0 при необходимости."""
        if self._settings.static_token:
            return self._settings.static_token

        now = time.monotonic()
        if not force_refresh and self._access_token and now < self._token_expires_at:
            return self._access_token

        async with self._token_lock:
            now = time.monotonic()
            if (
                not force_refresh
                and self._access_token
                and now < self._token_expires_at
            ):
                return self._access_token

            if not self._settings.client_id or not self._settings.client_secret:
                raise TrueConfAPIError(
                    "Не заданы TRUECONF_CLIENT_ID / TRUECONF_CLIENT_SECRET или TRUECONF_STATIC_TOKEN"
                )

            client = self._ensure_client()
            response = await client.post(
                self._settings.token_url,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self._settings.client_id,
                    "client_secret": self._settings.client_secret,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

            if response.status_code >= 400:
                raise TrueConfAPIError(
                    f"Ошибка OAuth2 авторизации: HTTP {response.status_code}",
                    status_code=response.status_code,
                    payload=response.text,
                )

            token_data = response.json()
            access_token = token_data.get("access_token")
            if not access_token:
                raise TrueConfAPIError(
                    "Ответ OAuth2 не содержит access_token", payload=token_data
                )

            expires_in = int(token_data.get("expires_in", 3600))
            self._access_token = access_token
            self._token_expires_at = time.monotonic() + max(
                expires_in - self._settings.token_refresh_margin, 10
            )
            logger.debug(
                "OAuth2 токен TrueConf успешно обновлён (expires_in=%ss)", expires_in
            )
            return self._access_token

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Выполняет HTTP-запрос с автообновлением токена при 401 и ретраями при сетевых/5xx ошибках."""
        client = self._ensure_client()
        token_refreshed = False

        for attempt in range(self._settings.max_retries + 1):
            token = await self._get_access_token(force_refresh=token_refreshed)
            headers = {"Authorization": f"Bearer {token}"}

            try:
                response = await client.request(
                    method=method,
                    url=path,
                    params=params,
                    json=json_body,
                    headers=headers,
                )
            except httpx.RequestError as exc:
                if attempt >= self._settings.max_retries:
                    raise TrueConfAPIError(
                        f"Сетевая ошибка при обращении к TrueConf: {exc}"
                    ) from exc
                await asyncio.sleep(self._settings.retry_backoff * (2**attempt))
                continue

            if (
                response.status_code == 401
                and not token_refreshed
                and not self._settings.static_token
            ):
                logger.warning(
                    "Получен HTTP 401 от TrueConf API, принудительно обновляем токен..."
                )
                token_refreshed = True
                continue

            if (
                response.status_code in (429, 500, 502, 503, 504)
                and attempt < self._settings.max_retries
            ):
                await asyncio.sleep(self._settings.retry_backoff * (2**attempt))
                continue

            if response.status_code >= 400:
                raise TrueConfAPIError(
                    f"Ошибка TrueConf API {method} {path}: HTTP {response.status_code}",
                    status_code=response.status_code,
                    payload=response.text,
                )

            return response.json() if response.content else {}

        raise TrueConfAPIError(f"Исчерпаны попытки запроса {method} {path}")

    async def get_completed_calls(
        self,
        *,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        page_size: int | None = None,
        max_pages: int = 10,
    ) -> list[CallSummary]:
        """
        1. Запрашивает список звонков (GET /logs/calls) с пагинацией
        и возвращает только завершённые сессии.
        """
        limit = page_size or self._settings.poll_page_size
        completed_calls: list[CallSummary] = []

        for page_id in range(1, max_pages + 1):
            params: dict[str, Any] = {
                "page_id": page_id,
                "page_size": limit,
            }
            if date_from is not None:
                params["date_from"] = date_from.astimezone(timezone.utc).isoformat()
            if date_to is not None:
                params["date_to"] = date_to.astimezone(timezone.utc).isoformat()

            data = await self._request("GET", "/logs/calls", params=params)

            # В API v4 список может возвращаться в ключе 'calls', 'list' или 'items'
            raw_items = data.get("calls") or data.get("list") or data.get("items") or []
            if not raw_items:
                break

            for item in raw_items:
                call = CallSummary.model_validate(item)
                if call.is_completed:
                    completed_calls.append(call)

            # Если получено меньше элементов, чем page_size, следующая страница пуста
            if len(raw_items) < limit:
                break

        return completed_calls

    async def get_call_details(self, session_id: str) -> CallDetails:
        """
        2. Получает детальную информацию о конкретном звонке (GET /logs/calls/{session_id}).
        """
        data = await self._request("GET", f"/logs/calls/{session_id}")
        call_payload = data.get("call") if isinstance(data.get("call"), dict) else data

        details = CallDetails.model_validate(call_payload)
        details.raw_payload = call_payload
        return details

    async def get_conference_records(self, conference_id: str) -> list[CallRecord]:
        """
        Вспомогательный метод: получает список готовых записей для конференции
        (GET /conferences/{conference_id}/records).
        """
        data = await self._request("GET", f"/conferences/{conference_id}/records")
        raw_records = data.get("records") or data.get("list") or data.get("items") or []
        return [CallRecord.model_validate(rec) for rec in raw_records]
