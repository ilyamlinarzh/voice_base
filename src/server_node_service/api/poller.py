import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .client import TrueConfAPIError, TrueConfClient
from config import TrueConfSettings
from .schemas import CallDetails, CallRecord

logger = logging.getLogger(__name__)

# Тип callback-функции, которая берёт готовый звонок в работу
CallHandler = Callable[[CallDetails, list[CallRecord]], Awaitable[None]]


class ProcessedCallsStore:
    """Простое персистентное хранилище ID обработанных звонков на диске (можно заменить на Redis/Postgres)."""

    def __init__(self, file_path: Path, max_items: int = 5000):
        self._file_path = file_path
        self._max_items = max_items
        self._ids: list[str] = []
        self._id_set: set[str] = set()
        self._load()

    def _load(self) -> None:
        if not self._file_path.exists():
            return
        try:
            data = json.loads(self._file_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                self._ids = [str(x) for x in data][-self._max_items :]
                self._id_set = set(self._ids)
        except Exception as exc:
            logger.warning(
                "Не удалось прочитать файл состояния %s: %s", self._file_path, exc
            )

    def contains(self, session_id: str) -> bool:
        return session_id in self._id_set

    def add(self, session_id: str) -> None:
        if session_id in self._id_set:
            return
        self._ids.append(session_id)
        self._id_set.add(session_id)
        if len(self._ids) > self._max_items:
            removed = self._ids.pop(0)
            self._id_set.discard(removed)
        self._save()

    def _save(self) -> None:
        try:
            self._file_path.parent.mkdir(parents=True, exist_ok=True)
            self._file_path.write_text(
                json.dumps(self._ids, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as exc:
            logger.warning(
                "Не удалось сохранить состояние пуллинга в %s: %s", self._file_path, exc
            )


class TrueConfCallPoller:
    """Асинхронный поллер завершённых звонков TrueConf Server."""

    def __init__(
        self,
        client: TrueConfClient,
        settings: TrueConfSettings,
        on_call_ready: CallHandler,
    ):
        self._client = client
        self._settings = settings
        self._on_call_ready = on_call_ready
        self._store = ProcessedCallsStore(settings.state_file)
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        """Запускает фоновую asyncio-задачу опроса."""
        if self._task is None or self._task.done():
            self._stop_event.clear()
            self._task = asyncio.create_task(
                self._run_loop(), name="trueconf-call-poller"
            )
            logger.info(
                "Пуллинг TrueConf запущен (интервал: %ss, require_recording=%s)",
                self._settings.poll_interval_seconds,
                self._settings.require_recording,
            )

    async def stop(self) -> None:
        """Корректно останавливает поллер, дожидаясь завершения текущего тика."""
        self._stop_event.set()
        if self._task is not None:
            await self._task
            self._task = None
            logger.info("Пуллинг TrueConf остановлен.")

    async def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                await self.poll_once()
            except TrueConfAPIError as exc:
                logger.error("Ошибка API при опросе звонков TrueConf: %s", exc)
            except Exception:
                logger.exception("Непредвиденная ошибка в цикле пуллинга TrueConf")

            # Ожидаем следующий интервал с мгновенной реакцией на сигнал остановки
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self._settings.poll_interval_seconds,
                )
            except asyncio.TimeoutError:
                continue

    async def poll_once(self) -> int:
        """
        Выполняет одну итерацию опроса.
        Возвращает количество новых звонков, взятых в работу.
        """
        date_from = datetime.now(timezone.utc) - timedelta(
            seconds=self._settings.poll_lookback_seconds
        )
        completed_calls = await self._client.get_completed_calls(date_from=date_from)

        # Сортируем от старых к новым, чтобы обрабатывать в хронологическом порядке
        completed_calls.sort(
            key=lambda c: c.end_time or datetime.min.replace(tzinfo=timezone.utc)
        )

        taken_count = 0
        for summary in completed_calls:
            if self._stop_event.is_set():
                break

            if self._store.contains(summary.id):
                continue

            # Запрашиваем полные данные о звонке
            details = await self._client.get_call_details(summary.id)
            if not details.is_completed:
                continue

            records: list[CallRecord] = []
            if details.conference_id:
                try:
                    records = await self._client.get_conference_records(
                        details.conference_id
                    )
                except TrueConfAPIError as exc:
                    logger.warning(
                        "Не удалось получить записи для conference_id=%s: %s",
                        details.conference_id,
                        exc,
                    )

            # Если для бизнес-логики обязательна готовая запись, ждём её появления в следующем тике
            if self._settings.require_recording and not records:
                logger.debug(
                    "Звонок %s завершён, но запись ещё не сформирована — отложен до следующего опроса",
                    details.id,
                )
                continue

            logger.info(
                "Найден новый завершённый звонок для обработки: session_id=%s",
                details.id,
            )
            try:
                await self._on_call_ready(details, records)
                # Помечаем звонок обработанным только после успешного выполнения callback
                self._store.add(details.id)
                taken_count += 1
            except Exception:
                logger.exception(
                    "Ошибка при обработке звонка session_id=%s — повторим на следующем тике",
                    details.id,
                )

        return taken_count
