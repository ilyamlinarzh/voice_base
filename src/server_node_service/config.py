from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TrueConfSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="TRUECONF_",
        extra="ignore",
    )

    # Базовый URL и пути API
    url_base: str = Field(
        ..., description="Базовый URL TrueConf Server, например http://trueconf-server"
    )
    api_prefix: str = Field(default="/api/v4", description="Префикс REST API v4")
    oauth_token_path: str = Field(
        default="/oauth2/v1/token", description="Эндпоинт получения OAuth2 токена"
    )

    # Креды OAuth 2.0
    client_id: str | None = Field(
        default=None, description="Application ID из раздела API -> OAuth2"
    )
    client_secret: str | None = Field(
        default=None, description="Secret из раздела API -> OAuth2"
    )
    static_token: str | None = Field(
        default=None, description="Тестовый статический токен (если не задан OAuth)"
    )

    # Параметры HTTP-клиента
    verify_ssl: bool = Field(
        default=False, description="Проверка TLS-сертификата между контейнерами"
    )
    request_timeout: float = Field(default=15.0, ge=1.0)
    max_retries: int = Field(default=3, ge=0)
    retry_backoff: float = Field(default=1.0, ge=0.1)
    token_refresh_margin: int = Field(
        default=60, ge=10, description="За сколько секунд до истечения обновлять токен"
    )

    # Параметры пуллинга
    poll_interval_seconds: float = Field(default=10.0, ge=1.0)
    poll_page_size: int = Field(default=50, ge=1, le=200)
    poll_lookback_seconds: int = Field(
        default=3600, ge=60, description="Глубина выборки при старте воркера"
    )
    require_recording: bool = Field(
        default=False,
        description="Брать в работу звонок только после появления готовой серверной записи",
    )
    state_file: Path = Field(
        default=Path("/tmp/trueconf_processed_calls.json"),
        description="Файл для сохранения ID уже обработанных сессий между перезапусками",
    )

    @property
    def api_base_url(self) -> str:
        return f"{self.url_base.rstrip('/')}/{self.api_prefix.strip('/')}"

    @property
    def token_url(self) -> str:
        return f"{self.url_base.rstrip('/')}/{self.oauth_token_path.lstrip('/')}"
