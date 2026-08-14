# Changelog

## 0.2.0 — 2026-08-14

- Injectable `backend_instance` and `auto_init` on `SpinRAG` so the algorithm can be tested without a network.
- Exported `DEFAULT_LLAMACPP_MODEL`, `DEFAULT_OPENROUTER_BASE_URL`, `parse_spin`, `clean_llm_output`, `cosine_similarity` (fixes the Dash demo imports).
- Offline pytest suite (unit, integration, contract) with an 85% coverage gate.
- Multi-stage `Dockerfile` (`base` / `test` / `runtime`) and `docker-compose.yml`.
- GitHub Actions CI (Ruff, Python 3.11/3.12, Docker) and a tag-driven GHCR release workflow.
- `demo.py` honors `HOST` / `PORT` / `DASH_DEBUG` for container use.

## 0.1.0 — 2026-06-06

First official release: OpenRouter + llama.cpp backends via the OpenAI SDK.
