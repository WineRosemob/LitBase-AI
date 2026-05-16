# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in LitBase-AI, **please do not open a public issue**.

Instead, please report it privately by sending an email to:

**security@example.com** *(replace with your security contact)*

We will respond as quickly as possible and keep you informed of our progress.

### What to Include

- A clear description of the vulnerability
- Steps to reproduce (if possible)
- The affected version(s)
- Any potential mitigations you've identified

### Disclosure Policy

1. The vulnerability is reported privately
2. We will acknowledge receipt within 48 hours
3. We will investigate and provide an initial assessment within 1 week
4. Once a fix is ready, we will release a patch and publicly disclose the vulnerability
5. Credit will be given to the reporter (unless anonymity is requested)

## Security Best Practices for Users

- **Never commit `.env` files** — use `.env.example` as a template
- Store API keys and credentials only in local `.env` files (git-ignored)
- Keep your `OPENALEX_MAILTO` and `UNPAYWALL_EMAIL` up to date for fair-use rate limits
- Review `WEBVPN_USERNAME` and `WEBVPN_PASSWORD` before committing config changes

---

> This security policy is adapted from industry best practices and is subject to change.
