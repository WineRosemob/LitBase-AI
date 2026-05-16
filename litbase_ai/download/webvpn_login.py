"""WebVPN / institutional proxy login — local browser + cookie capture.

Opens a visible Firefox browser window. You complete the WebVPN login
manually (including CAPTCHA). When you press Enter in the terminal,
cookies are captured and saved. Subsequent pipeline runs reuse them.

Also supports importing cookies from a JSON file or Netscape cookies.txt.
Supports publisher SSO login for direct publisher access without WebVPN.
"""

from __future__ import annotations
import asyncio, json, sys, os
from pathlib import Path

try:
    from playwright.async_api import async_playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

import httpx
from litbase_ai.utils.logging import get_logger

logger = get_logger(__name__)
COOKIE_DIR = Path.home() / ".litbase-ai" / "webvpn_cookies"

# Known publisher login URLs for quick SSO access
PUBLISHER_LOGIN_URLS = {
    "elsevier": "https://www.sciencedirect.com/",
    "springer": "https://link.springer.com/",
    "wiley": "https://onlinelibrary.wiley.com/",
    "nature": "https://www.nature.com/",
    "science": "https://www.science.org/",
    "ieee": "https://ieeexplore.ieee.org/",
    "taylor": "https://www.tandfonline.com/",
    "pnas": "https://www.pnas.org/",
    "acs": "https://pubs.acs.org/",
    "rsc": "https://pubs.rsc.org/",
    "iop": "https://iopscience.iop.org/",
    "oxford": "https://academic.oup.com/",
    "acm": "https://dl.acm.org/",
}

UA = "Mozilla/5.0 (X11; Linux x86_64; rv:126.0) Gecko/20100101 Firefox/126.0"


class WebVPNLoginError(Exception):
    pass


class WebVPNLoginClient:
    """Local browser login + cookie capture for WebVPN / publisher SSO."""

    def __init__(self, vpn_url="https://webvpn.example.edu", cookie_dir=None, headless=False):
        self.vpn_url = vpn_url.rstrip("/")
        self.cookie_dir = Path(cookie_dir or COOKIE_DIR)
        self.cookie_dir.mkdir(parents=True, exist_ok=True)
        self.headless = headless

    @property
    def cookie_file(self) -> Path:
        host = self.vpn_url.replace("https://","").replace("http://","").split("/")[0]
        return self.cookie_dir / f"{host.replace(':','_').replace('.','_')}_cookies.json"

    # ── Main API ──────────────────────────────────────────────────────

    def login_manual(self, username_hint: str = "") -> bool:
        """Open browser, let user login manually, capture cookies on Enter.

        Args:
            username_hint: Displayed in the terminal prompt for reference.
        """
        if not HAS_PLAYWRIGHT:
            raise WebVPNLoginError("pip install playwright && playwright install firefox")
        return asyncio.run(self._login_manual_async(username_hint))

    def login_publisher(self, publisher: str) -> bool:
        """Open browser at publisher SSO page, let user login, capture cookies.

        Args:
            publisher: Publisher key (see PUBLISHER_LOGIN_URLS).
        """
        url = PUBLISHER_LOGIN_URLS.get(publisher.lower(), self.vpn_url)
        saved = self.vpn_url
        self.vpn_url = url
        try:
            return self.login_manual(f"({publisher})")
        finally:
            self.vpn_url = saved

    def test_cookies(self) -> tuple[bool, str]:
        """Test if saved cookies are still valid."""
        if not self.cookie_file.exists():
            return False, "no saved cookies"
        try:
            cookies = json.loads(self.cookie_file.read_text(encoding="utf-8"))
        except Exception as e:
            return False, str(e)
        if not cookies:
            return False, "empty"

        try:
            c = httpx.Client(timeout=15, follow_redirects=True)
            for ck in cookies:
                c.cookies.set(ck["name"], ck["value"], domain=ck.get("domain",""), path=ck.get("path","/"))
            r = c.get(self.vpn_url, headers={"User-Agent": UA}, follow_redirects=True)
            ok = r.status_code < 400 and "login" not in str(r.url).lower()
            return ok, f"status={r.status_code}" if ok else "redirected to login"
        except Exception as e:
            return False, str(e)

    def load_cookies(self) -> list[dict]:
        """Load saved cookies for use in httpx/requests sessions."""
        if not self.cookie_file.exists():
            return []
        try:
            return json.loads(self.cookie_file.read_text(encoding="utf-8"))
        except Exception:
            return []

    def import_cookies_file(self, path: str | Path) -> bool:
        """Import cookies from a JSON file or Netscape cookies.txt format."""
        path = Path(path)
        if not path.exists():
            logger.error("File not found: %s", path)
            return False

        try:
            content = path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error("Cannot read %s: %s", path, e)
            return False

        cookies = self._parse_cookies(content)
        if not cookies:
            logger.error("No cookies found in %s", path)
            return False

        self.cookie_file.write_text(json.dumps(cookies, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Imported %d cookies from %s → %s", len(cookies), path, self.cookie_file)
        return True

    def import_cookies_clipboard(self, text: str) -> bool:
        """Import cookies from clipboard JSON text."""
        cookies = self._parse_cookies(text)
        if not cookies:
            logger.error("No valid cookies found in clipboard text")
            return False
        self.cookie_file.write_text(json.dumps(cookies, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Imported %d cookies from clipboard → %s", len(cookies), self.cookie_file)
        return True

    def export_cookies_netscape(self) -> str | None:
        """Export cookies in Netscape format for use with curl/wget."""
        cookies = self.load_cookies()
        if not cookies:
            return None
        lines = ["# Netscape HTTP Cookie File\n"]
        for c in cookies:
            domain = c.get("domain", "")
            flag = "TRUE" if domain.startswith(".") else "FALSE"
            path = c.get("path", "/")
            secure = "TRUE" if c.get("secure") else "FALSE"
            expires = "0"
            name = c.get("name", "")
            value = c.get("value", "")
            lines.append(f"{domain}\t{flag}\t{path}\t{secure}\t{expires}\t{name}\t{value}\n")
        return "".join(lines)

    # ── Async login flow ──────────────────────────────────────────────

    async def _login_manual_async(self, hint: str):
        async with async_playwright() as pw:
            browser = await pw.firefox.launch(headless=self.headless)
            ctx = await browser.new_context(
                viewport={"width": 1280, "height": 800},
                user_agent=UA,
            )
            try:
                page = await ctx.new_page()
                await page.goto(self.vpn_url, wait_until="networkidle", timeout=30000)

                print(f"""
╔══════════════════════════════════════════════════════════════╗
║  {('WebVPN' if 'webvpn' in self.vpn_url.lower() else 'Publisher')} Login{' - ' + hint if hint else ''} {' ' * (26 - len(hint))}║
╠══════════════════════════════════════════════════════════════╣
║  A Firefox window has opened at:                           ║
║  {self.vpn_url[:53]}║
║                                                              ║
║  1. Complete the login in the browser window                ║
║     (enter username, password, CAPTCHA, etc.)               ║
║  2. Wait for the portal homepage to load                    ║
║  3. Return to THIS terminal and press Enter                 ║
║                                                              ║
║  Cookies will be captured and reused automatically.         ║
╚══════════════════════════════════════════════════════════════╝
""")
                try:
                    input("Press ENTER when login is complete...")
                except (EOFError, KeyboardInterrupt):
                    print("\nAborted.")
                    return False

                # Capture all cookies
                cookies = await ctx.cookies()
                if not cookies:
                    logger.warning("No cookies captured — login may not have completed")
                    print("No cookies found. Make sure you completed the login.")
                    return False

                # Build cookie list
                cl = []
                domain_set = set()
                for c in cookies:
                    cl.append({
                        "name": c["name"], "value": c["value"],
                        "domain": c.get("domain", ""), "path": c.get("path", "/"),
                        "httpOnly": c.get("httpOnly", False),
                        "secure": c.get("secure", False),
                    })
                    domain_set.add(c.get("domain", "").lstrip("."))

                # Save
                self.cookie_file.write_text(
                    json.dumps(cl, ensure_ascii=False, indent=2), encoding="utf-8")

                # Also save Netscape format
                netscape_path = self.cookie_file.with_suffix(".txt")
                netscape_path.write_text(self._to_netscape(cl), encoding="utf-8")

                # Also save/merge publisher cookies so multi-publisher logins
                # can accumulate instead of overwriting previous sessions.
                publisher_path = self.cookie_dir / "publisher_cookies.json"
                existing: list[dict] = []
                if publisher_path.exists():
                    try:
                        loaded_existing = json.loads(publisher_path.read_text(encoding="utf-8"))
                        if isinstance(loaded_existing, list):
                            existing = [x for x in loaded_existing if isinstance(x, dict)]
                    except Exception:
                        existing = []

                merged: list[dict] = []
                seen: set[tuple[str, str, str]] = set()
                for item in [*existing, *cl]:
                    name = str(item.get("name") or "").strip()
                    if not name:
                        continue
                    domain = str(item.get("domain") or "").strip().lower()
                    path_key = str(item.get("path") or "/")
                    key = (name, domain, path_key)
                    if key in seen:
                        continue
                    seen.add(key)
                    merged.append(item)

                publisher_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")

                print(f"""
╔══════════════════════════════════════════════════════════════╗
║  ✓ Login successful!                                        ║
║                                                              ║
║  Cookies saved: {len(cl):>3}                                 ║
║  Domains: {len(domain_set):>3}                               ║
║                                                              ║
║  JSON: {str(self.cookie_file):>48}║
║  Netscape: {str(netscape_path):>43}║
║                                                              ║
║  You can now run the pipeline:                              ║
║    litbase-ai search --env-file examples/example.env ...    ║
╚══════════════════════════════════════════════════════════════╝
""")
                return True

            finally:
                await ctx.close()
                await browser.close()

    # ── Cookie parsing ────────────────────────────────────────────────

    @staticmethod
    def _parse_cookies(text: str) -> list[dict]:
        """Parse cookies from JSON array or Netscape format."""
        text = text.strip()

        # Try JSON first
        if text.startswith("["):
            try:
                data = json.loads(text)
                return [
                    {"name": c.get("name",""), "value": c.get("value",""),
                     "domain": c.get("domain",""), "path": c.get("path","/")}
                    for c in data if isinstance(c, dict) and c.get("name")
                ]
            except json.JSONDecodeError:
                pass
        if text.startswith("{"):
            try:
                data = json.loads(text)
                if isinstance(data, dict) and "cookies" in data:
                    return WebVPNLoginClient._parse_cookies(json.dumps(data["cookies"]))
            except json.JSONDecodeError:
                pass

        # Try Netscape format
        if "# Netscape" in text or "# HTTP Cookie" in text:
            cookies = []
            for line in text.split("\n"):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) >= 7:
                    cookies.append({
                        "name": parts[5], "value": parts[6],
                        "domain": parts[0], "path": parts[2] if len(parts) > 2 else "/",
                    })
            if cookies:
                return cookies

        return []

    @staticmethod
    def _to_netscape(cookies: list[dict]) -> str:
        lines = ["# Netscape HTTP Cookie File\n"]
        for c in cookies:
            domain = c.get("domain", "")
            flag = "TRUE" if domain.startswith(".") else "FALSE"
            path = c.get("path", "/")
            secure = "TRUE" if c.get("secure") else "FALSE"
            lines.append(f"{domain}\t{flag}\t{path}\t{secure}\t0\t{c['name']}\t{c['value']}\n")
        return "".join(lines)


# ── CLI ────────────────────────────────────────────────────────────────

def login_cli():
    import argparse
    p = argparse.ArgumentParser(
        description="WebVPN / publisher SSO login — opens browser, captures cookies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Login to WebVPN
  python -m litbase_ai.download.webvpn_login --vpn-url https://webvpn.example.edu

  # Login to a publisher directly
  python -m litbase_ai.download.webvpn_login --publisher elsevier

  # Import cookies from file
  python -m litbase_ai.download.webvpn_login --import-file cookies.json

  # Test saved cookies
  python -m litbase_ai.download.webvpn_login --test

  # Export to Netscape format
  python -m litbase_ai.download.webvpn_login --export-netscape
        """,
    )
    p.add_argument("--vpn-url", default=os.environ.get("WEBVPN_URL", "https://webvpn.example.edu"))
    p.add_argument("--publisher", help="Publisher key for SSO (elsevier, springer, wiley, nature, ieee, ...)")
    p.add_argument("--cookie-dir")
    p.add_argument("--test", action="store_true", help="Test if saved cookies are still valid")
    p.add_argument("--import-file", help="Import cookies from JSON or Netscape cookies.txt file")
    p.add_argument("--import-clipboard", help="Import cookies from clipboard JSON text")
    p.add_argument("--export-netscape", action="store_true", help="Export cookies in Netscape format")
    p.add_argument("--headless", action="store_true", default=False)
    args = p.parse_args()

    client = WebVPNLoginClient(
        vpn_url=args.vpn_url,
        cookie_dir=args.cookie_dir,
        headless=args.headless,
    )

    # Cookie test
    if args.test:
        ok, msg = client.test_cookies()
        print(f"{'OK' if ok else 'FAILED'} — {msg}")
        sys.exit(0 if ok else 1)

    # Import
    if args.import_file:
        ok = client.import_cookies_file(args.import_file)
        sys.exit(0 if ok else 1)

    if args.import_clipboard:
        ok = client.import_cookies_clipboard(args.import_clipboard)
        sys.exit(0 if ok else 1)

    # Export
    if args.export_netscape:
        netscape = client.export_cookies_netscape()
        if netscape:
            print(netscape)
        else:
            print("No cookies available. Run login first.")
            sys.exit(1)
        return

    # Login
    if args.publisher:
        print(f"Logging into publisher: {args.publisher}...")
        ok = client.login_publisher(args.publisher)
    else:
        ok = client.login_manual()

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    login_cli()
