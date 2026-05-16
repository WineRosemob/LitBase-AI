from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from litbase_ai.config import AppConfig
from litbase_ai.download.cookie_bootstrap import (
    CookieBootstrapPolicy,
    CookieBootstrapper,
    load_cookie_bootstrap_policy,
    parse_publishers_arg,
    publisher_host_keywords,
    save_cookie_bootstrap_policy,
)


class _FakeWebVPNClientNoCookie:
    def __init__(self, vpn_url: str, cookie_dir=None, headless=False):
        self.vpn_url = vpn_url
        base = Path(cookie_dir or tempfile.gettempdir())
        self.cookie_file = base / "fake_webvpn_cookie.json"

    def test_cookies(self) -> tuple[bool, str]:
        return False, "no saved cookies"

    def login_manual(self, username_hint: str = "") -> bool:
        return False


class CookieBootstrapTest(unittest.TestCase):
    def test_policy_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            policy_path = Path(tmp_dir) / "policy.json"
            policy = CookieBootstrapPolicy(
                disable_inst_proxy=True,
                disable_ezproxy=False,
                disabled_publishers=["nature", "elsevier"],
            )
            save_cookie_bootstrap_policy(policy, path=policy_path)
            loaded = load_cookie_bootstrap_policy(policy_path)

        self.assertTrue(loaded.disable_inst_proxy)
        self.assertFalse(loaded.disable_ezproxy)
        self.assertEqual(sorted(loaded.disabled_publishers), ["elsevier", "nature"])

    def test_parse_publishers(self) -> None:
        self.assertEqual(parse_publishers_arg("elsevier, springer ,nature"), ["elsevier", "springer", "nature"])
        self.assertTrue(parse_publishers_arg(None))

    def test_publisher_host_keywords(self) -> None:
        keys = publisher_host_keywords(["elsevier", "wiley"])
        self.assertIn("sciencedirect.com", keys)
        self.assertIn("onlinelibrary.wiley.com", keys)

    def test_bootstrap_non_interactive_disables_missing_flows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env.test"
            env_path.write_text(
                "\n".join(
                    [
                        "WEBVPN_URL=https://webvpn.example.edu",
                        "WEBVPN_AUTO_LOGIN=true",
                        "ENABLE_INST_PROXY=true",
                        "ENABLE_EZPROXY=true",
                        "EZPROXY_TEMPLATE=https://ezproxy.example/login?url={url}",
                        f"EZPROXY_COOKIE_FILE={Path(tmp_dir) / 'missing_ezproxy_cookie.json'}",
                    ]
                ),
                encoding="utf-8",
            )
            config = AppConfig.load(env_file=env_path)
            policy_path = Path(tmp_dir) / "cookie_policy.json"

            with patch("litbase_ai.download.cookie_bootstrap.WebVPNLoginClient", _FakeWebVPNClientNoCookie):
                bootstrapper = CookieBootstrapper(
                    config=config,
                    publishers=["elsevier", "nature"],
                    cookie_dir=tmp_dir,
                    policy_path=policy_path,
                )
                report = bootstrapper.run(interactive=False)

            policy = load_cookie_bootstrap_policy(policy_path)

        self.assertTrue(report["webvpn"]["checked"])
        self.assertTrue(policy.disable_inst_proxy)
        self.assertTrue(policy.disable_ezproxy)
        self.assertIn("elsevier", policy.disabled_publishers)
        self.assertIn("nature", policy.disabled_publishers)


if __name__ == "__main__":
    unittest.main()
