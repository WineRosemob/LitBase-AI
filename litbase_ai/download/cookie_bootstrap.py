from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from litbase_ai.config import AppConfig
from litbase_ai.download.webvpn_login import COOKIE_DIR, PUBLISHER_LOGIN_URLS, WebVPNLoginClient
from litbase_ai.utils.logging import get_logger


logger = get_logger(__name__)

DEFAULT_POLICY_PATH = COOKIE_DIR / "cookie_bootstrap_policy.json"
DEFAULT_PUBLISHERS = ["elsevier", "springer", "wiley", "nature", "ieee"]


@dataclass
class CookieBootstrapPolicy:
    disable_inst_proxy: bool = False
    disable_ezproxy: bool = False
    disabled_publishers: list[str] = field(default_factory=list)
    updated_at_utc: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "disable_inst_proxy": bool(self.disable_inst_proxy),
            "disable_ezproxy": bool(self.disable_ezproxy),
            "disabled_publishers": sorted({p.strip().lower() for p in self.disabled_publishers if p and p.strip()}),
            "updated_at_utc": self.updated_at_utc or datetime.now(timezone.utc).isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CookieBootstrapPolicy":
        disabled_publishers = data.get("disabled_publishers")
        if not isinstance(disabled_publishers, list):
            disabled_publishers = []
        return cls(
            disable_inst_proxy=bool(data.get("disable_inst_proxy", False)),
            disable_ezproxy=bool(data.get("disable_ezproxy", False)),
            disabled_publishers=[str(p).strip().lower() for p in disabled_publishers if str(p).strip()],
            updated_at_utc=str(data.get("updated_at_utc") or ""),
        )


def load_cookie_bootstrap_policy(path: str | Path | None = None) -> CookieBootstrapPolicy:
    policy_path = Path(path).expanduser() if path else DEFAULT_POLICY_PATH
    if not policy_path.exists():
        return CookieBootstrapPolicy()
    try:
        payload = json.loads(policy_path.read_text(encoding="utf-8"))
    except Exception:
        return CookieBootstrapPolicy()
    if not isinstance(payload, dict):
        return CookieBootstrapPolicy()
    return CookieBootstrapPolicy.from_dict(payload)


def save_cookie_bootstrap_policy(policy: CookieBootstrapPolicy, path: str | Path | None = None) -> Path:
    policy_path = Path(path).expanduser() if path else DEFAULT_POLICY_PATH
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_text(json.dumps(policy.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return policy_path


def publisher_host_keywords(publishers: list[str]) -> list[str]:
    hosts: list[str] = []
    for publisher in publishers:
        key = str(publisher).strip().lower()
        if not key:
            continue
        url = PUBLISHER_LOGIN_URLS.get(key)
        if not url:
            continue
        parsed = urlparse(url)
        host = parsed.netloc.strip().lower()
        if host.startswith("www."):
            host = host[4:]
        if host:
            hosts.append(host)
            if "." in host:
                parts = host.split(".")
                if len(parts) >= 2:
                    hosts.append(".".join(parts[-2:]))
    dedup: list[str] = []
    seen: set[str] = set()
    for h in hosts:
        if h in seen:
            continue
        seen.add(h)
        dedup.append(h)
    return dedup


def parse_publishers_arg(raw: str | None) -> list[str]:
    if raw is None:
        return list(DEFAULT_PUBLISHERS)
    parsed = [seg.strip().lower() for seg in str(raw).split(",")]
    return [p for p in parsed if p]


class CookieBootstrapper:
    """Interactive/non-interactive cookie bootstrap wizard.

    Scope:
    - WebVPN cookie (for institutional proxy flow)
    - EZProxy cookie file presence
    - Optional per-publisher cookie availability
    - Persist disable policy so later pipeline runs can skip related flows
    """

    def __init__(
        self,
        *,
        config: AppConfig,
        publishers: list[str] | None = None,
        cookie_dir: str | Path | None = None,
        policy_path: str | Path | None = None,
        headless: bool = False,
    ) -> None:
        self.config = config
        self.cookie_dir = Path(cookie_dir).expanduser() if cookie_dir else COOKIE_DIR
        self.cookie_dir.mkdir(parents=True, exist_ok=True)
        self.policy_path = Path(policy_path).expanduser() if policy_path else DEFAULT_POLICY_PATH
        self.publishers = publishers if publishers is not None else list(DEFAULT_PUBLISHERS)
        self.headless = bool(headless)

    def run(self, *, interactive: bool = True) -> dict[str, Any]:
        policy = load_cookie_bootstrap_policy(self.policy_path)
        report: dict[str, Any] = {
            "policy_path": str(self.policy_path),
            "interactive": bool(interactive),
            "webvpn": {},
            "ezproxy": {},
            "publishers": {},
        }

        # 1) WebVPN cookie check
        webvpn_result = self._ensure_webvpn_cookie(interactive=interactive, policy=policy)
        report["webvpn"] = webvpn_result

        # 2) EZProxy cookie check (if EZProxy configured)
        ezproxy_result = self._ensure_ezproxy_cookie(interactive=interactive, policy=policy)
        report["ezproxy"] = ezproxy_result

        # 3) Publisher cookies
        publisher_results = self._ensure_publishers(interactive=interactive, policy=policy)
        report["publishers"] = publisher_results

        policy.updated_at_utc = datetime.now(timezone.utc).isoformat()
        save_cookie_bootstrap_policy(policy, self.policy_path)
        report["policy"] = policy.to_dict()
        return report

    def _ensure_webvpn_cookie(self, *, interactive: bool, policy: CookieBootstrapPolicy) -> dict[str, Any]:
        vpn_url = (self.config.webvpn_url or "").strip()
        needs_inst_proxy = bool(
            vpn_url
            or self.config.enable_inst_proxy
            or self.config.inst_proxy_school
            or self.config.inst_proxy_url
            or self.config.webvpn_auto_login
        )
        if not needs_inst_proxy:
            return {
                "checked": False,
                "reason": "inst_proxy_not_configured",
                "disable_inst_proxy": policy.disable_inst_proxy,
            }

        if not vpn_url:
            policy.disable_inst_proxy = True
            return {
                "checked": False,
                "reason": "webvpn_url_missing",
                "disable_inst_proxy": policy.disable_inst_proxy,
            }

        client = WebVPNLoginClient(vpn_url=vpn_url, cookie_dir=self.cookie_dir, headless=self.headless)
        ok, msg = client.test_cookies()
        if ok:
            policy.disable_inst_proxy = False
            return {
                "checked": True,
                "cookie_ok": True,
                "message": msg,
                "cookie_file": str(client.cookie_file),
                "disable_inst_proxy": policy.disable_inst_proxy,
            }

        attempted_login = False
        if interactive and self._confirm(
            prompt=(
                f"[Init] WebVPN cookie 无效 ({msg})。是否现在打开 Firefox 手动登录并保存 cookie?"
            ),
            default=True,
        ):
            attempted_login = True
            ok = client.login_manual(username_hint=self.config.webvpn_username or "")
            if ok:
                ok, msg = client.test_cookies()

        if not ok:
            disable = True
            if interactive:
                disable = self._confirm(
                    prompt="[Init] 没有可用 WebVPN cookie。是否禁用 InstProxy/WebVPN 下载流程（后续不再调用）?",
                    default=True,
                )
            policy.disable_inst_proxy = bool(disable)

        return {
            "checked": True,
            "cookie_ok": bool(ok),
            "message": msg,
            "cookie_file": str(client.cookie_file),
            "attempted_login": attempted_login,
            "disable_inst_proxy": policy.disable_inst_proxy,
        }

    def _ensure_ezproxy_cookie(self, *, interactive: bool, policy: CookieBootstrapPolicy) -> dict[str, Any]:
        needs_ezproxy = bool(self.config.enable_ezproxy or self.config.ezproxy_template)
        if not needs_ezproxy:
            return {
                "checked": False,
                "reason": "ezproxy_not_configured",
                "disable_ezproxy": policy.disable_ezproxy,
            }

        cookie_file_raw = (self.config.ezproxy_cookie_file or "").strip()
        if not cookie_file_raw:
            policy.disable_ezproxy = True
            return {
                "checked": True,
                "cookie_ok": False,
                "message": "EZPROXY_COOKIE_FILE missing",
                "disable_ezproxy": policy.disable_ezproxy,
            }

        cookie_file = Path(cookie_file_raw).expanduser()
        exists = cookie_file.exists() and cookie_file.is_file() and cookie_file.stat().st_size > 0
        if exists:
            policy.disable_ezproxy = False
            return {
                "checked": True,
                "cookie_ok": True,
                "cookie_file": str(cookie_file),
                "disable_ezproxy": policy.disable_ezproxy,
            }

        disable = True
        if interactive:
            disable = self._confirm(
                prompt=(
                    f"[Init] 未找到 EZProxy cookie 文件: {cookie_file}。"
                    "是否禁用 EZProxy 流程（后续不再调用）?"
                ),
                default=True,
            )
        policy.disable_ezproxy = bool(disable)
        return {
            "checked": True,
            "cookie_ok": False,
            "cookie_file": str(cookie_file),
            "disable_ezproxy": policy.disable_ezproxy,
        }

    def _ensure_publishers(self, *, interactive: bool, policy: CookieBootstrapPolicy) -> dict[str, Any]:
        results: dict[str, Any] = {}
        disabled = {p.strip().lower() for p in policy.disabled_publishers if p and p.strip()}
        for publisher in self.publishers:
            key = publisher.strip().lower()
            if not key:
                continue
            if key not in PUBLISHER_LOGIN_URLS:
                results[key] = {"supported": False, "message": "unknown_publisher"}
                continue

            has_cookie = self._has_cookie_for_publisher(key)
            attempted_login = False
            if not has_cookie and interactive and self._confirm(
                prompt=f"[Init] 未检测到 {key} 的 cookie。是否现在登录 {key}?",
                default=False,
            ):
                attempted_login = True
                client = WebVPNLoginClient(vpn_url=self.config.webvpn_url or PUBLISHER_LOGIN_URLS[key], cookie_dir=self.cookie_dir, headless=self.headless)
                ok = client.login_publisher(key)
                has_cookie = bool(ok) and self._has_cookie_for_publisher(key)

            if not has_cookie:
                should_disable = True
                if interactive:
                    should_disable = self._confirm(
                        prompt=f"[Init] 没有 {key} cookie。是否禁用 {key} 相关代理尝试（后续不再调用）?",
                        default=True,
                    )
                if should_disable:
                    disabled.add(key)
                else:
                    disabled.discard(key)
            else:
                disabled.discard(key)

            results[key] = {
                "supported": True,
                "cookie_ok": has_cookie,
                "attempted_login": attempted_login,
                "disabled": key in disabled,
            }

        policy.disabled_publishers = sorted(disabled)
        return results

    def _has_cookie_for_publisher(self, publisher: str) -> bool:
        url = PUBLISHER_LOGIN_URLS.get(publisher)
        if not url:
            return False
        host = urlparse(url).netloc.lower()
        if host.startswith("www."):
            host = host[4:]

        # Check host-specific cookie file first.
        host_cookie_file = self.cookie_dir / f"{host.replace(':', '_').replace('.', '_')}_cookies.json"
        if self._cookie_file_has_host(host_cookie_file, host):
            return True

        # Then check merged publisher cookie bundle.
        merged_cookie_file = self.cookie_dir / "publisher_cookies.json"
        return self._cookie_file_has_host(merged_cookie_file, host)

    @staticmethod
    def _cookie_file_has_host(path: Path, host: str) -> bool:
        if not path.exists():
            return False
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return False
        if not isinstance(loaded, list):
            return False
        for item in loaded:
            if not isinstance(item, dict):
                continue
            domain = str(item.get("domain") or "").strip().lstrip(".").lower()
            if not domain:
                continue
            if domain == host or host.endswith(domain) or domain.endswith(host):
                return True
        return False

    @staticmethod
    def _confirm(*, prompt: str, default: bool) -> bool:
        suffix = "[Y/n]" if default else "[y/N]"
        while True:
            try:
                raw = input(f"{prompt} {suffix} ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return default
            if not raw:
                return default
            if raw in {"y", "yes", "1", "true"}:
                return True
            if raw in {"n", "no", "0", "false"}:
                return False
            print("Please answer y or n.")


__all__ = [
    "CookieBootstrapPolicy",
    "CookieBootstrapper",
    "DEFAULT_POLICY_PATH",
    "DEFAULT_PUBLISHERS",
    "load_cookie_bootstrap_policy",
    "parse_publishers_arg",
    "publisher_host_keywords",
    "save_cookie_bootstrap_policy",
]
