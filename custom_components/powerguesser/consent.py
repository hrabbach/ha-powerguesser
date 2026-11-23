"""Consent store and verification for PowerGuesser LLM learning."""

from collections.abc import Mapping
from typing import Any, TypedDict

from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store
import homeassistant.util.dt as dt_util

from .const import CONSENT_STORE_KEY, LEARNING_STORES_VERSION
from .llm_client import endpoint_origin


class ConsentRecord(TypedDict):
    """Stored record of explicit user consent for LLM payload sharing.

    Consent covers one origin, scheme + host + port with the default port filled in
    (https 443, http 80), and the entity-names flag. The model is not part of it (D-11,
    amended by WR-11): a transport downgrade or another port asks again.
    """

    scheme: str
    host: str
    port: int
    timestamp: str
    entity_names_flag: bool


def format_origin(scheme: str, host: str, port: int | None) -> str:
    """Render an origin as scheme://host:port (IPv6 hosts in brackets)."""
    shown_host = f"[{host}]" if ":" in host else host
    return f"{scheme}://{shown_host}:{port}"


def consent_valid(
    record: Mapping[str, Any] | None, base_url: str, names_flag: bool
) -> bool:
    """Return True if stored consent matches the endpoint's origin and the names flag.

    A record without scheme or port (the pre-WR-11 host-only shape) is never valid.
    """
    if record is None:
        return False
    if not record.get("scheme") or record.get("port") is None:
        return False
    scheme, host, port = endpoint_origin(base_url)
    return (
        record.get("scheme") == scheme
        and record.get("host") == host
        and record.get("port") == port
        and record.get("entity_names_flag") is names_flag
    )


class ConsentStore:
    """Wrapper around Home Assistant Store for consent records."""

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize the consent store."""
        self._store = Store[dict[str, Any]](
            hass, LEARNING_STORES_VERSION, CONSENT_STORE_KEY
        )
        self._record: ConsentRecord | None = None

    @property
    def record(self) -> ConsentRecord | None:
        """Return the in-memory cached consent record."""
        return self._record

    async def async_load(self) -> ConsentRecord | None:
        """Load consent record from disk into memory.

        Legacy host-only records are treated as no consent (no migration: phase 3 never
        shipped a release), so the user is asked again for the full origin.
        """
        data = await self._store.async_load()
        self._record = None
        if data and isinstance(data, dict):
            scheme = data.get("scheme")
            port = data.get("port")
            if isinstance(scheme, str) and scheme and isinstance(port, int):
                self._record = {
                    "scheme": scheme,
                    "host": str(data.get("host", "")),
                    "port": port,
                    "timestamp": str(data.get("timestamp", "")),
                    "entity_names_flag": bool(data.get("entity_names_flag", False)),
                }
        return self._record

    async def async_grant(self, base_url: str, names_flag: bool) -> None:
        """Grant consent for the endpoint's origin and the names flag, saving immediately."""
        scheme, host, port = endpoint_origin(base_url)
        self._record = {
            "scheme": scheme,
            "host": host,
            "port": int(port) if port is not None else 0,
            "timestamp": dt_util.utcnow().isoformat(),
            "entity_names_flag": names_flag,
        }
        await self._store.async_save(self._record)
