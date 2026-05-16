"""PDF download modules.

This package uses lazy attribute loading to avoid eager importing of heavy
submodules (e.g. playwright-based login helpers) during package import.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .arxiv_download import ArxivDownloader
    from .ezproxy import EZProxyClient
    from .inst_proxy import InstProxyClient
    from .legal_source_resolver import LegalPDFSourceResolver
    from .libgen import LibGenClient
    from .pdf_downloader import PDFDownloader
    from .school_db import SchoolEntry
    from .scihub import SciHubClient
    from .webvpn_login import PUBLISHER_LOGIN_URLS, WebVPNLoginClient


_LAZY_SYMBOLS: dict[str, tuple[str, str]] = {
    "ArxivDownloader": ("litbase_ai.download.arxiv_download", "ArxivDownloader"),
    "EZProxyClient": ("litbase_ai.download.ezproxy", "EZProxyClient"),
    "InstProxyClient": ("litbase_ai.download.inst_proxy", "InstProxyClient"),
    "LegalPDFSourceResolver": ("litbase_ai.download.legal_source_resolver", "LegalPDFSourceResolver"),
    "LibGenClient": ("litbase_ai.download.libgen", "LibGenClient"),
    "PDFDownloader": ("litbase_ai.download.pdf_downloader", "PDFDownloader"),
    "PUBLISHER_LOGIN_URLS": ("litbase_ai.download.webvpn_login", "PUBLISHER_LOGIN_URLS"),
    "SchoolEntry": ("litbase_ai.download.school_db", "SchoolEntry"),
    "SciHubClient": ("litbase_ai.download.scihub", "SciHubClient"),
    "WebVPNLoginClient": ("litbase_ai.download.webvpn_login", "WebVPNLoginClient"),
    "get_host_for": ("litbase_ai.download.school_db", "get_host_for"),
    "get_keys_for": ("litbase_ai.download.school_db", "get_keys_for"),
    "list_all": ("litbase_ai.download.school_db", "list_all"),
    "search": ("litbase_ai.download.school_db", "search"),
    "search_multi": ("litbase_ai.download.school_db", "search_multi"),
}


def __getattr__(name: str):
    target = _LAZY_SYMBOLS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(_LAZY_SYMBOLS.keys()))


__all__ = sorted(_LAZY_SYMBOLS.keys())
