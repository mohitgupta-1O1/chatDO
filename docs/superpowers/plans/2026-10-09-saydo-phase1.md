# SayDo Phase 1 (M0–M4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship Phase 1 of SayDo: record or upload a meeting, then get a speaker-labelled, timestamped transcript. Overlapping speech is represented explicitly, every segment has a confidence score, review flags mark doubtful segments, and users can rename speakers, reassign segments, merge speakers and acknowledge reviews. A self-hosted Python API and an embeddable React package deliver all of this, with an evaluation harness that measures it.

**Architecture:**
- **API:** a FastAPI service (`api/`). It owns storage (SQLite through SQLAlchemy and Alembic, plus files under `data_dir`), a database-backed job queue drained by one in-process worker thread, and a pipeline of swappable plugins: transcriber, diarizer and embedder.
- **Pipeline:** pure functions (overlap detection, word-to-speaker merge, confidence) sit between the plugins and the database, so each one can be unit-tested without models.
- **Front end:** a React 19 package (`packages/react`) talks to the API through a typed client generated from OpenAPI. A Next.js 16 reference app (`web/`) only composes that package.

**Tech Stack:**
- **API:** Python 3.12, uv, FastAPI, SQLAlchemy 2, Alembic, pydantic-settings, numpy, soundfile, ffmpeg.
- **Models:** faster-whisper and pyannote.audio 4.x.
- **Front end:** pnpm workspaces, TypeScript 5.9, React 19, tsup, Vitest, Testing Library, fake-indexeddb, Next.js 16, Playwright.
- **Evaluation:** pyannote.metrics, rapidfuzz, scipy.

**Spec:** `docs/superpowers/specs/2026-10-09-saydo-design.md`. Read §5–§10, §12, §13 and §16–§18 before starting.

## Global Constraints

- **Python and Node versions:**
  - Python `>=3.12`, managed with `uv`. The system Python on the dev machine is 3.9; never use it.
  - Node `>=22` and pnpm `>=10`.
- **Front-end library versions:**
  - TypeScript `^5.9.3`, not 7.x, because tsup's `.d.ts` build needs the TypeScript JS API.
  - Next.js `^16.4.0`, React and React DOM `^19.3.0`.
- **License:** the project is Apache-2.0. No GPL or AGPL runtime dependencies in `api/` core or `packages/react`.
- **Configuration:**
  - Precedence: defaults < YAML file (`$SAYDO_CONFIG`, default `./saydo.yaml`) < env vars `SAYDO__SECTION__KEY` (for example `SAYDO__API__TOKEN`, `SAYDO__PIPELINE__TRANSCRIBER__NAME`).
- **HTTP conventions:**
  - Every route lives under `/api/v1`.
  - Errors always use the body `{"error": {"code": str, "message": str}}`.
- **Review and confidence defaults:**
  - threshold `0.6`;
  - weights asr `0.4`, margin `0.4`, length `0.2`;
  - penalties: overlap `0.6`, separated `0.8`, orphan `0.7`;
  - margin temperature `0.05`;
  - length saturates at `1.5` s;
  - segments shorter than `0.5` s get margin `0.5`.
- **Merge constants:**
  - an overlap region must last at least `0.2` s; regions closer than `0.1` s merge;
  - a new segment starts after a gap over `1.0` s;
  - the orphan window is `0.5` s.
- **Hallucination guard:** drop an ASR segment when `no_speech_prob > 0.6` **and** `avg_logprob < -1.0`.
- **Speaker labels:** `"Speaker N"`, numbered by first appearance in time. Display name is `display_name ?? label`.
- **Privacy:**
  - No audio and no transcript text in logs, ever. Logs contain IDs, stage names, counts and durations.
  - All recording files live under `data_dir/recordings/<uuid>/`. Every file-system path built from an ID is validated to resolve inside that directory.
  - SayDo sets `PYANNOTE_METRICS_ENABLED=0` (pyannote's opt-out telemetry) before importing pyannote.
- **Datetimes:** stored as naive UTC and serialized as ISO-8601 with a `Z` suffix.
- **IDs:** UUIDv7 strings.
- **One API process:** run the API as a single process. One worker thread owns the models and GPU.

## Review Focus

Five ways real use could break the program even though the spec never mentions them, most likely first. Each one has a test in the task that owns the code.

1. **A recording is deleted while its job is processing.** Expected: no rows remain, no files reappear, and the worker keeps running. Tests: Task 9 `test_process_raises_recording_gone_when_deleted_mid_run`, Task 10 `test_recording_deleted_mid_job_cleans_up`.
2. **A corrupt or non-audio file is uploaded with an audio MIME type.** Expected: the recording becomes `failed` at stage `normalize` with a readable message, and the worker goes on to the next job. Test: Task 10 `test_corrupt_upload_fails_cleanly_and_worker_continues`.
3. **A silent recording, or one with no detectable speech.** Expected: `ready` with zero segments, and the UI says "No speech detected" instead of an empty page or placeholder rows. Tests: Task 9 `test_silent_audio_is_ready_with_no_segments`, Task 17 `shows "No speech detected" for an empty transcript`.
4. **A malformed or traversal recording ID in a URL** (`..`, or anything that isn't a UUID). Expected: 404 without touching the file system. Tests: Task 3 `test_dir_rejects_non_uuid_and_traversal`, Task 4 `test_traversal_id_is_404`.
5. **Tab sharing goes wrong in the browser.** If the user clicks the browser's "Stop sharing" mid-meeting, or shares a tab without ticking "Share tab audio", the recorder stops and keeps the captured audio, or shows a clear error, instead of hanging. Tests: Task 15 `auto-stops and keeps audio when a captured track ends`, `openCapture throws no_tab_audio when the shared tab has no audio`.

---

## Spec deviations (decided while planning; recorded in spec §22)

1. **Audio playback with a token.** `<audio>` elements can't send an `Authorization` header. So `GET /recordings/{id}/audio-url` returns a short-lived URL signed with HMAC-SHA256, valid for one hour, which the audio endpoint accepts in place of the header.
2. **`POST /recordings/{id}/retry`** implements the "Retry" button from spec §15.
3. **Embedder as its own plugin kind.** `pipeline.embedder` is a third plugin kind (default `pyannote`, model `pyannote/wespeaker-voxceleb-resnet34-LM`). SayDo computes speaker centroids itself, from non-overlapping segments, so confidence margins are computed in one embedding space whichever diarizer is in use. Diarizers therefore return only turns.
4. **Plugins receive in-memory audio.** Plugin methods take an `Audio` object (samples plus sample rate), not a `Path`, so the audio is decoded once.
5. **One Dockerfile for the API**, with a `TORCH_FLAVOR` build argument (`cpu` or `cu128`) and `docker-compose.gpu.yml` as an override, instead of two Dockerfiles.
6. **Evaluation code location.** The evaluation library lives in `api/src/saydo/evaluation/` so it is typed and unit-tested. `eval/` keeps only the CLI scripts, data and reports.
7. **cpWER is computed in SayDo** (Hungarian speaker mapping plus rapidfuzz word-level Levenshtein distance) instead of with meeteval, so we don't depend on an API we can't pin.
8. **LibriCSS is deferred to M5.** The manifest format accepts it, but no download script is written now. Phase 1 overlap results are broken down by overlap ratio across AMI meetings and in-house recordings.

---

## File Structure

```
.gitignore                                  # Task 1
saydo.example.yaml                          # Task 1 (extended in Task 11)
LICENSE, README.md, THIRD_PARTY_LICENSES.md # Task 13
docker/api.Dockerfile, docker/web.Dockerfile, docker-compose.yml, docker-compose.gpu.yml  # Task 13 (web Dockerfile in Task 19)
.github/workflows/ci.yml                    # Task 13 (web + e2e jobs added in Task 19)
api/
  pyproject.toml, .python-version, alembic.ini                       # Tasks 1, 2
  scripts/export_openapi.py                                          # Task 14
  src/saydo/
    __init__.py                 version
    config.py                   Settings (YAML + env)                # Task 1
    ids.py                      uuid7(), is_uuid()                   # Task 2
    main.py                     create_app(), run()                  # Tasks 1, 4, 10
    services.py                 delete_recording, load_detail, bump_version, editable_recording   # Tasks 4, 12
    db/models.py, db/session.py, db/migrations/{env.py,script.py.mako,versions/}                  # Task 2
    storage/files.py            RecordingFiles (paths + deletion)    # Task 3
    api/errors.py, api/auth.py, api/deps.py, api/schemas.py, api/health.py, api/recordings.py     # Task 4
    api/speakers.py, api/segments.py                                  # Task 12
    pipeline/errors.py          PipelineError, AudioDecodeError, PluginLoadError, RecordingGone   # Task 3
    pipeline/normalize.py       ffmpeg normalization                 # Task 3
    pipeline/types.py           Word, AsrSegment, Turn, Diarization, Audio, load_audio            # Task 5
    pipeline/interfaces.py      Transcriber, Diarizer, Embedder protocols                         # Task 5
    pipeline/registry.py        plugin lookup, device resolution    # Task 5
    pipeline/overlap.py         Region, find_overlap_regions, region_index_at                     # Task 6
    pipeline/merge.py           DraftSegment, build_segments        # Task 7
    pipeline/confidence.py      scoring + needs_review              # Task 8
    pipeline/speakers.py        embeddings, centroids, margins, numbering                         # Task 9
    pipeline/run.py             Plugins, analyze(), PipelineRunner  # Task 9
    jobs/worker.py, jobs/sweeper.py                                  # Task 10
    plugins/fake.py                                                  # Task 5
    plugins/faster_whisper_transcriber.py, plugins/pyannote_diarizer.py,
    plugins/pyannote_embedder.py, plugins/multichannel_diarizer.py   # Task 11
    evaluation/rttm.py, evaluation/metrics.py, evaluation/manifest.py, evaluation/runner.py      # Task 20
  tests/conftest.py, tests/audio_helpers.py, tests/unit/*, tests/api/*, tests/contract/*          # all tasks
package.json, pnpm-workspace.yaml, tsconfig.base.json                                             # Task 14
packages/react/
  package.json, tsconfig.json, tsup.config.ts, vitest.config.ts, vitest.setup.ts, openapi.json   # Task 14
  src/index.ts, src/styles.css                                                                    # Tasks 14–18
  src/client/schema.d.ts (generated), src/client/client.ts, src/context.tsx                       # Task 14
  src/recorder/capture.ts, src/recorder/chunkStore.ts, src/recorder/useRecorder.ts                # Task 15
  src/components/MeetingRecorder.tsx                                                              # Task 16
  src/transcript/model.ts, src/hooks/useRecording.ts, src/components/{TranscriptView,SegmentRow,OverlapGroup}.tsx  # Task 17
  src/components/SpeakerChip.tsx                                                                  # Task 18
web/                                                                                              # Task 19
eval/{prepare_ami.py,run_eval.py,README.md,reports/.gitkeep}                                      # Task 20
docs/acceptance/phase1.md                                                                         # Task 21
```

---

### Task 1: Python project, configuration and health endpoint (M0)

**Files:**
- Create: `.gitignore`, `saydo.example.yaml`
- Create: `api/.python-version`, `api/pyproject.toml`
- Create: `api/src/saydo/__init__.py`, `api/src/saydo/config.py`, `api/src/saydo/main.py`
- Test: `api/tests/conftest.py`, `api/tests/unit/test_config.py`, `api/tests/api/test_health.py`

**Interfaces:**
- Produces:
  - `saydo.__version__: str`.
  - `saydo.config.Settings` with fields:
    - `data_dir: Path`, `database_url: str | None`;
    - `api: ApiSettings(token, cors_origins)`;
    - `upload: UploadSettings(max_bytes, allowed_mime_prefixes)`;
    - `pipeline: PipelineSettings(device, transcriber, diarizer, embedder)`, where each plugin entry is a `PluginSettings(name, options: dict)`;
    - `review: ReviewSettings(threshold, weights: ReviewWeights(asr, margin, length), overlap_penalty, separated_penalty, orphan_penalty)`;
    - `retention: RetentionSettings(delete_raw_audio_after_processing, keep_playback_audio, days)`.

    Property `resolved_database_url: str`.
  - `saydo.main.create_app(settings: Settings | None = None) -> FastAPI` and `saydo.main.run() -> None`.
  - Test fixtures: `settings` (fake plugins, `data_dir` under `tmp_path`).

- [ ] **Step 1: Install prerequisites (one time)**

```bash
brew install uv ffmpeg          # Linux: curl -LsSf https://astral.sh/uv/install.sh | sh && sudo apt-get install -y ffmpeg
uv --version && ffmpeg -version | head -1
```
Expected: both print versions.

- [ ] **Step 2: Create the project files**

`.gitignore`:
```gitignore
# Python
__pycache__/
*.py[cod]
.venv/
.pytest_cache/
.mypy_cache/
.ruff_cache/
# Node
node_modules/
dist/
.next/
playwright-report/
test-results/
# SayDo runtime data: never commit meeting audio or transcripts
data/
/saydo.yaml
api/saydo.yaml
eval/data/
.env
```

`api/.python-version`:
```
3.12
```

`api/pyproject.toml`:
```toml
[project]
name = "saydo"
version = "0.1.0"
description = "Speaker-labelled meeting transcripts with honest overlap handling"
requires-python = ">=3.12"
license = "Apache-2.0"
dependencies = [
  "fastapi>=0.115",
  "uvicorn[standard]>=0.30",
  "python-multipart>=0.0.9",
  "sqlalchemy>=2.0.30",
  "alembic>=1.13",
  "pydantic>=2.7",
  "pydantic-settings[yaml]>=2.4",
  "numpy>=2.0",
  "soundfile>=0.12",
]

[project.optional-dependencies]
ml = ["faster-whisper>=1.1", "pyannote.audio>=4.0", "torch>=2.4"]
eval = ["pyannote.metrics>=3.2", "rapidfuzz>=3.9", "scipy>=1.13"]

[dependency-groups]
dev = ["pytest>=8", "httpx>=0.27", "ruff>=0.6", "mypy>=1.11"]

[project.scripts]
saydo-api = "saydo.main:run"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/saydo"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["tests"]
addopts = "-m 'not slow' --import-mode=importlib"
markers = ["slow: needs real models and network (run with: uv run --extra ml --extra eval pytest -m slow)"]

[tool.ruff]
line-length = 100
target-version = "py312"
extend-exclude = ["src/saydo/db/migrations/versions"]

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "S"]
ignore = ["S101", "S603", "S607"]

[tool.ruff.lint.per-file-ignores]
"tests/**" = ["S"]

[tool.mypy]
python_version = "3.12"
ignore_missing_imports = true
check_untyped_defs = true
```

`api/src/saydo/__init__.py`:
```python
__version__ = "0.1.0"
```

`saydo.example.yaml`:
```yaml
# Copy to saydo.yaml (or point SAYDO_CONFIG at it). Env vars override: SAYDO__SECTION__KEY.
data_dir: ./data
# database_url: postgresql+psycopg://saydo:saydo@localhost/saydo   # default: SQLite in data_dir
api:
  token: null                       # set to require "Authorization: Bearer <token>"
  cors_origins: ["http://localhost:3000"]
upload:
  max_bytes: 1073741824             # 1 GiB
pipeline:
  device: auto                      # auto | cuda | mps | cpu
  transcriber: { name: faster_whisper, options: { model: large-v3-turbo, compute_type: auto } }
  diarizer:    { name: pyannote, options: { model: pyannote/speaker-diarization-community-1 } }
  embedder:    { name: pyannote, options: { model: pyannote/wespeaker-voxceleb-resnet34-LM } }
review:
  threshold: 0.6                    # segments below this confidence are flagged "needs review"
  weights: { asr: 0.4, margin: 0.4, length: 0.2 }
  overlap_penalty: 0.6
retention:
  delete_raw_audio_after_processing: false
  keep_playback_audio: true
  days: null                        # e.g. 30 to hard-delete recordings after 30 days
```

- [ ] **Step 3: Write the failing tests**

`api/tests/conftest.py`:
```python
import os

import pytest

from saydo.config import Settings

FAKE_PIPELINE = {
    "device": "cpu",
    "transcriber": {"name": "fake"},
    "diarizer": {"name": "fake"},
    "embedder": {"name": "fake"},
}


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch, tmp_path):
    """Keep the developer's shell config and saydo.yaml out of tests."""
    for key in list(os.environ):
        if key.startswith("SAYDO__"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("SAYDO_CONFIG", str(tmp_path / "no-config.yaml"))


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(data_dir=tmp_path / "data", pipeline=FAKE_PIPELINE)
```

`api/tests/unit/test_config.py`:
```python
from saydo.config import Settings


def test_defaults_match_spec(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    s = Settings()
    assert s.pipeline.device == "auto"
    assert s.pipeline.transcriber.name == "faster_whisper"
    assert s.pipeline.diarizer.name == "pyannote"
    assert s.pipeline.embedder.name == "pyannote"
    assert s.review.threshold == 0.6
    assert (s.review.weights.asr, s.review.weights.margin, s.review.weights.length) == (0.4, 0.4, 0.2)
    assert s.review.overlap_penalty == 0.6
    assert s.retention.days is None
    assert s.resolved_database_url.startswith("sqlite:///")
    assert s.resolved_database_url.endswith("data/saydo.db")


def test_yaml_file_then_env_override(tmp_path, monkeypatch):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "review:\n  threshold: 0.5\n"
        "pipeline:\n  transcriber:\n    name: fake\n    options: {model: tiny}\n"
    )
    monkeypatch.setenv("SAYDO_CONFIG", str(cfg))
    monkeypatch.setenv("SAYDO__REVIEW__THRESHOLD", "0.7")
    s = Settings()
    assert s.review.threshold == 0.7
    assert s.pipeline.transcriber.name == "fake"
    assert s.pipeline.transcriber.options == {"model": "tiny"}


def test_env_can_set_plugin_options_without_name(monkeypatch):
    monkeypatch.setenv("SAYDO__PIPELINE__TRANSCRIBER__OPTIONS__MODEL", "small")
    s = Settings()
    assert s.pipeline.transcriber.name == "faster_whisper"
    assert s.pipeline.transcriber.options == {"model": "small"}


def test_env_parses_json_lists(monkeypatch):
    monkeypatch.setenv("SAYDO__API__CORS_ORIGINS", '["https://a.example","https://b.example"]')
    monkeypatch.setenv("SAYDO__API__TOKEN", "s3cret")
    s = Settings()
    assert s.api.cors_origins == ["https://a.example", "https://b.example"]
    assert s.api.token == "s3cret"
```

`api/tests/api/test_health.py`:
```python
from fastapi.testclient import TestClient

from saydo import __version__
from saydo.main import create_app


def test_health(settings):
    client = TestClient(create_app(settings))
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}


def test_openapi_is_served_under_api_v1(settings):
    client = TestClient(create_app(settings))
    assert client.get("/api/v1/openapi.json").status_code == 200
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `cd api && uv sync && uv run pytest -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'saydo.config'`.

- [ ] **Step 5: Implement**

`api/src/saydo/config.py`:
```python
"""SayDo configuration.

Precedence (lowest to highest): defaults < YAML file ($SAYDO_CONFIG, default ./saydo.yaml)
< environment variables named SAYDO__SECTION__KEY.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


class ApiSettings(BaseModel):
    token: str | None = None
    cors_origins: list[str] = ["http://localhost:3000"]


class UploadSettings(BaseModel):
    max_bytes: int = 1024 * 1024 * 1024
    allowed_mime_prefixes: list[str] = ["audio/", "video/webm", "video/mp4"]


class PluginSettings(BaseModel):
    name: str
    options: dict[str, Any] = {}


# Each plugin kind has its own default name so that env vars can override
# options alone (SAYDO__PIPELINE__TRANSCRIBER__OPTIONS__MODEL=small).
class TranscriberSettings(PluginSettings):
    name: str = "faster_whisper"


class DiarizerSettings(PluginSettings):
    name: str = "pyannote"


class EmbedderSettings(PluginSettings):
    name: str = "pyannote"


class PipelineSettings(BaseModel):
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    transcriber: TranscriberSettings = TranscriberSettings()
    diarizer: DiarizerSettings = DiarizerSettings()
    embedder: EmbedderSettings = EmbedderSettings()


class ReviewWeights(BaseModel):
    asr: float = 0.4
    margin: float = 0.4
    length: float = 0.2


class ReviewSettings(BaseModel):
    threshold: float = Field(0.6, ge=0.0, le=1.0)
    weights: ReviewWeights = ReviewWeights()
    overlap_penalty: float = 0.6
    separated_penalty: float = 0.8
    orphan_penalty: float = 0.7


class RetentionSettings(BaseModel):
    delete_raw_audio_after_processing: bool = False
    keep_playback_audio: bool = True
    days: int | None = Field(None, ge=1)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SAYDO__", env_nested_delimiter="__", extra="ignore"
    )

    data_dir: Path = Path("./data")
    database_url: str | None = None
    api: ApiSettings = ApiSettings()
    upload: UploadSettings = UploadSettings()
    pipeline: PipelineSettings = PipelineSettings()
    review: ReviewSettings = ReviewSettings()
    retention: RetentionSettings = RetentionSettings()

    @property
    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        return f"sqlite:///{(self.data_dir / 'saydo.db').resolve()}"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        sources: list[PydanticBaseSettingsSource] = [init_settings, env_settings]
        config_path = Path(os.environ.get("SAYDO_CONFIG", "saydo.yaml"))
        if config_path.is_file():
            sources.append(YamlConfigSettingsSource(settings_cls, yaml_file=config_path))
        return tuple(sources)
```

`api/src/saydo/main.py`:
```python
"""FastAPI application factory and the `saydo-api` CLI entry point."""

from __future__ import annotations

import os

from fastapi import FastAPI

from saydo import __version__
from saydo.config import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    app = FastAPI(
        title="SayDo API",
        version=__version__,
        openapi_url="/api/v1/openapi.json",
        docs_url="/api/v1/docs",
        redoc_url=None,
    )
    app.state.settings = settings

    @app.get("/api/v1/health", tags=["health"])
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    return app


def run() -> None:
    import uvicorn

    uvicorn.run(
        "saydo.main:create_app",
        factory=True,
        host=os.environ.get("SAYDO_HOST", "127.0.0.1"),
        port=int(os.environ.get("SAYDO_PORT", "8000")),
    )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd api && uv run pytest -q && uv run ruff check . && uv run ruff format --check .`
Expected: `6 passed`; ruff reports no issues. If the format check fails, run `uv run ruff format .` and re-run.

- [ ] **Step 7: Commit**

```bash
git add .gitignore saydo.example.yaml api/
git commit -m "feat(api): project scaffold, layered config and health endpoint"
```

---

### Task 2: Database schema, sessions and migrations (M0)

**Files:**
- Create: `api/src/saydo/ids.py`
- Create: `api/src/saydo/db/__init__.py` (empty), `api/src/saydo/db/models.py`, `api/src/saydo/db/session.py`
- Create: `api/src/saydo/db/migrations/env.py`, `api/src/saydo/db/migrations/script.py.mako`, `api/alembic.ini`
- Generate: `api/src/saydo/db/migrations/versions/0001_initial_schema.py`
- Modify: `api/tests/conftest.py` (append the `engine` and `session_factory` fixtures)
- Test: `api/tests/unit/test_ids.py`, `api/tests/unit/test_db.py`

**Interfaces:**
- Consumes: `Settings.resolved_database_url` (Task 1).
- Produces:
  - `saydo.ids.uuid7() -> str`, `saydo.ids.is_uuid(value: str) -> bool`.
  - `saydo.db.models`: `Base`, `utcnow() -> datetime` (naive UTC), and the ORM classes:
    - `Recording`: id, title, created_at, source, original_filename, mime_type, channels, duration_s, status, stage, error, min_speakers, max_speakers, language, pipeline_config, transcript_version, raw_audio_deleted_at, retain_until.
    - `Speaker`: id, recording_id, label, display_name, color_index, embedding, merged_into_id.
    - `OverlapRegion`: id, recording_id, start_s, end_s, n_speakers.
    - `Segment`: id, recording_id, speaker_id, original_speaker_id, start_s, end_s, text, words, origin, is_overlap, overlap_region_id, confidence, confidence_parts, needs_review, user_corrected, reviewed.
    - `Job`: id, recording_id, kind, status, attempts, created_at, started_at, finished_at, error.
  - `saydo.db.session`: `make_engine(url: str) -> Engine`, `make_session_factory(engine) -> sessionmaker[Session]`, `migrate(engine) -> None`.
  - Test fixtures: `engine`, `session_factory`.

- [ ] **Step 1: Write the failing tests**

`api/tests/unit/test_ids.py`:
```python
import time
import uuid

from saydo.ids import is_uuid, uuid7


def test_uuid7_has_version_7_and_rfc_variant():
    value = uuid.UUID(uuid7())
    assert value.version == 7
    assert value.variant == uuid.RFC_4122


def test_uuid7_sorts_by_creation_time():
    first = uuid7()
    time.sleep(0.002)
    assert uuid7() > first


def test_is_uuid():
    assert is_uuid(uuid7())
    assert not is_uuid("../etc/passwd")
    assert not is_uuid("not-a-uuid")
    assert not is_uuid(uuid7().upper())
```

Append to `api/tests/conftest.py`:
```python
from saydo.db.session import make_engine, make_session_factory, migrate  # noqa: E402


@pytest.fixture
def engine(settings):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    eng = make_engine(settings.resolved_database_url)
    migrate(eng)
    yield eng
    eng.dispose()


@pytest.fixture
def session_factory(engine):
    return make_session_factory(engine)
```

`api/tests/unit/test_db.py`:
```python
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import delete, func, select

from saydo.db.models import Base, Job, OverlapRegion, Recording, Segment, Speaker


def test_migrations_match_models(engine):
    with engine.connect() as conn:
        context = MigrationContext.configure(conn, opts={"compare_type": True})
        assert compare_metadata(context, Base.metadata) == []


def test_deleting_recording_cascades_to_children(session_factory):
    with session_factory() as s:
        rec = Recording(source="upload", status="ready")
        s.add(rec)
        s.flush()
        spk = Speaker(recording_id=rec.id, label="Speaker 1", color_index=0)
        region = OverlapRegion(recording_id=rec.id, start_s=0.0, end_s=1.0, n_speakers=2)
        s.add_all([spk, region])
        s.flush()
        s.add(
            Segment(
                recording_id=rec.id, speaker_id=spk.id, original_speaker_id=spk.id,
                start_s=0.0, end_s=1.0, text="hi", words=[], origin="mixture",
                is_overlap=True, overlap_region_id=region.id, confidence=0.5,
                confidence_parts={}, needs_review=True,
            )
        )
        s.add(Job(recording_id=rec.id, kind="transcribe", status="queued"))
        s.commit()

        s.execute(delete(Recording).where(Recording.id == rec.id))
        s.commit()
        for model in (Speaker, Segment, Job, OverlapRegion):
            assert s.scalar(select(func.count()).select_from(model)) == 0


def test_defaults(session_factory):
    with session_factory() as s:
        rec = Recording(source="mic")
        s.add(rec)
        s.commit()
        assert rec.status == "uploaded"
        assert rec.transcript_version == 0
        assert rec.title == "Untitled meeting"
        assert rec.created_at.tzinfo is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd api && uv run pytest -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'saydo.ids'` (and `saydo.db`).

- [ ] **Step 3: Implement IDs and models**

`api/src/saydo/ids.py`:
```python
"""UUIDv7 identifiers (time-ordered), so IDs sort by creation time."""

import os
import re
import time
import uuid

_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def uuid7() -> str:
    millis = time.time_ns() // 1_000_000
    rand = int.from_bytes(os.urandom(10), "big")  # 80 random bits
    value = (
        (millis & ((1 << 48) - 1)) << 80
        | 0x7 << 76  # version
        | ((rand >> 62) & 0xFFF) << 64  # rand_a (12 bits)
        | 0b10 << 62  # RFC 4122 variant
        | rand & ((1 << 62) - 1)  # rand_b (62 bits)
    )
    return str(uuid.UUID(int=value))


def is_uuid(value: str) -> bool:
    return bool(_UUID_RE.match(value))
```

`api/src/saydo/db/__init__.py`: empty file.

`api/src/saydo/db/models.py`:
```python
"""ORM models. Datetimes are naive UTC; IDs are UUIDv7 strings.

Children reference `recording.id` with ON DELETE CASCADE, so deleting a recording
row removes everything derived from it (spec section 12).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from saydo.ids import uuid7


def utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


def _id() -> Mapped[str]:
    return mapped_column(String(36), primary_key=True, default=uuid7)


def _recording_fk() -> Mapped[str]:
    return mapped_column(
        String(36), ForeignKey("recording.id", ondelete="CASCADE"), index=True
    )


class Recording(Base):
    __tablename__ = "recording"

    id: Mapped[str] = _id()
    title: Mapped[str] = mapped_column(String(200), default="Untitled meeting")
    created_at: Mapped[datetime] = mapped_column(DateTime(), default=utcnow)
    source: Mapped[str] = mapped_column(String(16))  # mic | tab | mic+tab | upload
    original_filename: Mapped[str | None] = mapped_column(String(255))
    mime_type: Mapped[str | None] = mapped_column(String(100))
    channels: Mapped[int | None] = mapped_column(Integer)
    duration_s: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(16), default="uploaded")
    stage: Mapped[str | None] = mapped_column(String(32))
    error: Mapped[str | None] = mapped_column(Text)
    min_speakers: Mapped[int | None] = mapped_column(Integer)
    max_speakers: Mapped[int | None] = mapped_column(Integer)
    language: Mapped[str | None] = mapped_column(String(16))
    pipeline_config: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    transcript_version: Mapped[int] = mapped_column(Integer, default=0)
    raw_audio_deleted_at: Mapped[datetime | None] = mapped_column(DateTime())
    retain_until: Mapped[datetime | None] = mapped_column(DateTime(), index=True)


class Speaker(Base):
    __tablename__ = "speaker"

    id: Mapped[str] = _id()
    recording_id: Mapped[str] = _recording_fk()
    label: Mapped[str] = mapped_column(String(32))
    display_name: Mapped[str | None] = mapped_column(String(80))
    color_index: Mapped[int] = mapped_column(Integer, default=0)
    embedding: Mapped[bytes | None] = mapped_column(LargeBinary)
    merged_into_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("speaker.id", ondelete="SET NULL")
    )


class OverlapRegion(Base):
    __tablename__ = "overlap_region"

    id: Mapped[str] = _id()
    recording_id: Mapped[str] = _recording_fk()
    start_s: Mapped[float] = mapped_column(Float)
    end_s: Mapped[float] = mapped_column(Float)
    n_speakers: Mapped[int] = mapped_column(Integer)


class Segment(Base):
    __tablename__ = "segment"
    __table_args__ = (Index("ix_segment_recording_start", "recording_id", "start_s"),)

    id: Mapped[str] = _id()
    recording_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("recording.id", ondelete="CASCADE")
    )
    speaker_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("speaker.id", ondelete="CASCADE"), index=True
    )
    original_speaker_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("speaker.id", ondelete="CASCADE")
    )
    start_s: Mapped[float] = mapped_column(Float)
    end_s: Mapped[float] = mapped_column(Float)
    text: Mapped[str] = mapped_column(Text, default="")
    words: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    origin: Mapped[str] = mapped_column(String(16))  # mixture | separated | placeholder
    is_overlap: Mapped[bool] = mapped_column(Boolean, default=False)
    overlap_region_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("overlap_region.id", ondelete="SET NULL")
    )
    confidence: Mapped[float] = mapped_column(Float)
    confidence_parts: Mapped[dict[str, float]] = mapped_column(JSON, default=dict)
    needs_review: Mapped[bool] = mapped_column(Boolean, default=False)
    user_corrected: Mapped[bool] = mapped_column(Boolean, default=False)
    reviewed: Mapped[bool] = mapped_column(Boolean, default=False)


class Job(Base):
    __tablename__ = "job"

    id: Mapped[str] = _id()
    recording_id: Mapped[str] = _recording_fk()
    kind: Mapped[str] = mapped_column(String(16))  # transcribe (extract arrives in Phase 2)
    status: Mapped[str] = mapped_column(String(16), default="queued", index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime())
    error: Mapped[str | None] = mapped_column(Text)
```

`api/src/saydo/db/session.py`:
```python
"""Engine/session construction and programmatic Alembic migrations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def make_engine(url: str) -> Engine:
    kwargs: dict[str, Any] = {}
    if url.startswith("sqlite"):
        # The worker thread and request threads share the engine.
        kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
    engine = create_engine(url, **kwargs)
    if engine.dialect.name == "sqlite":

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_connection: Any, _record: Any) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()

    return engine


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(engine, expire_on_commit=False)


def migrate(engine: Engine) -> None:
    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS_DIR))
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, "head")
```

`api/src/saydo/db/migrations/env.py`:
```python
from logging.config import fileConfig

from alembic import context

from saydo.db.models import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _run(connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        render_as_batch=True,  # SQLite needs batch mode for ALTER TABLE
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connection = config.attributes.get("connection")
    if connection is not None:
        _run(connection)
        return
    # CLI usage (alembic revision --autogenerate): build the engine from Settings.
    from saydo.config import Settings
    from saydo.db.session import make_engine

    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    engine = make_engine(settings.resolved_database_url)
    with engine.begin() as conn:
        _run(conn)


if context.is_offline_mode():
    raise SystemExit("Offline migrations are not supported")
run_migrations_online()
```

`api/src/saydo/db/migrations/script.py.mako`:
```mako
"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

revision: str = ${repr(up_revision)}
down_revision: Union[str, None] = ${repr(down_revision)}
branch_labels: Union[str, Sequence[str], None] = ${repr(branch_labels)}
depends_on: Union[str, Sequence[str], None] = ${repr(depends_on)}


def upgrade() -> None:
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    ${downgrades if downgrades else "pass"}
```

`api/alembic.ini`:
```ini
[alembic]
script_location = src/saydo/db/migrations

[loggers]
keys = root

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARNING
handlers = console

[handler_console]
class = StreamHandler
args = (sys.stderr,)
formatter = generic

[formatter_generic]
format = %(levelname)s %(name)s %(message)s
```

- [ ] **Step 4: Generate the initial migration**

```bash
cd api
mkdir -p src/saydo/db/migrations/versions
rm -rf /tmp/saydo-gen && mkdir -p /tmp/saydo-gen
SAYDO__DATA_DIR=/tmp/saydo-gen uv run alembic revision --autogenerate --rev-id 0001 -m "initial schema"
```
Expected: `Generating .../versions/0001_initial_schema.py ... done`. Open the file: it must contain `op.create_table` for `recording`, `job`, `overlap_region`, `speaker` and `segment`, plus the indexes. If any table is missing, the model import in `env.py` is wrong. Fix that; don't edit the migration by hand.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd api && uv run pytest -q`
Expected: all pass, including `test_migrations_match_models` (empty diff) and the cascade test.

- [ ] **Step 6: Commit**

```bash
git add api/
git commit -m "feat(api): database models, uuid7 ids and alembic migrations"
```

---

### Task 3: File storage and audio normalization (M1)

**Files:**
- Create: `api/src/saydo/storage/__init__.py` (empty), `api/src/saydo/storage/files.py`
- Create: `api/src/saydo/pipeline/__init__.py` (empty), `api/src/saydo/pipeline/errors.py`, `api/src/saydo/pipeline/normalize.py`
- Create: `api/tests/audio_helpers.py`
- Test: `api/tests/unit/test_files.py`, `api/tests/unit/test_normalize.py`

**Interfaces:**
- Consumes: `saydo.ids.is_uuid` (Task 2).
- Produces:
  - `saydo.storage.files.InvalidRecordingId(ValueError)`.
  - `saydo.storage.files.RecordingFiles(data_dir: Path)` with methods:
    - `dir(rid) -> Path`
    - `original_path(rid, ext) -> Path`
    - `find_original(rid) -> Path | None`
    - `normalized_path(rid) -> Path`
    - `stems_dir(rid) -> Path`
    - `delete_all(rid) -> None`
    - `delete_raw(rid) -> None`
    - `delete_playback(rid) -> None`
  - `saydo.pipeline.errors`: `PipelineError(RuntimeError)`, `AudioDecodeError(PipelineError)`, `PluginLoadError(PipelineError)`, `RecordingGone(Exception)`.
  - `saydo.pipeline.normalize`:
    - `check_ffmpeg() -> None` (raises `RuntimeError`);
    - `probe_channels(src: Path) -> int`;
    - `normalize(src: Path, dst: Path, *, keep_channels: bool) -> NormalizedAudio`, where `NormalizedAudio(path: Path, duration_s: float, channels: int)`.
  - Test helpers `audio_helpers.write_tone(path, seconds, *, sample_rate=16000, channels=1, amplitude=0.3, silent=False) -> Path` and `audio_helpers.make_webm(path, seconds) -> Path`.

- [ ] **Step 1: Write the test helpers and failing tests**

`api/tests/audio_helpers.py`:
```python
"""Synthetic audio for tests. Requires ffmpeg on PATH for make_webm."""

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf


def write_tone(
    path: Path,
    seconds: float,
    *,
    sample_rate: int = 16000,
    channels: int = 1,
    amplitude: float = 0.3,
    silent: bool = False,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    mono = np.zeros_like(t) if silent else amplitude * np.sin(2 * np.pi * 220 * t)
    data = np.stack([mono] * channels, axis=1).astype(np.float32)
    sf.write(path, data, sample_rate, subtype="PCM_16")
    return path


def make_webm(path: Path, seconds: float = 2.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
            "-c:a", "libopus", str(path),
        ],
        check=True,
    )
    return path
```

`api/tests/unit/test_files.py`:
```python
import pytest

from saydo.ids import uuid7
from saydo.storage.files import InvalidRecordingId, RecordingFiles


def test_paths_live_under_data_dir(tmp_path):
    files = RecordingFiles(tmp_path)
    rid = uuid7()
    assert files.dir(rid) == (tmp_path / "recordings" / rid).resolve()
    assert files.original_path(rid, "webm").name == "original.webm"
    assert files.normalized_path(rid).name == "audio.wav"
    assert files.stems_dir(rid).name == "stems"


@pytest.mark.parametrize("bad", ["..", "../etc", "not-a-uuid", "", "a/b"])
def test_dir_rejects_non_uuid_and_traversal(tmp_path, bad):
    with pytest.raises(InvalidRecordingId):
        RecordingFiles(tmp_path).dir(bad)


def test_original_extension_is_sanitized(tmp_path):
    files = RecordingFiles(tmp_path)
    assert files.original_path(uuid7(), "../../x").name == "original.bin"
    assert files.original_path(uuid7(), "M4A").name == "original.m4a"


def test_find_original_and_delete(tmp_path):
    files = RecordingFiles(tmp_path)
    rid = uuid7()
    original = files.original_path(rid, "webm")
    original.parent.mkdir(parents=True)
    original.write_bytes(b"x")
    files.normalized_path(rid).write_bytes(b"y")
    files.stems_dir(rid).mkdir()
    (files.stems_dir(rid) / "s0.wav").write_bytes(b"z")
    assert files.find_original(rid) == original

    files.delete_raw(rid)
    assert files.find_original(rid) is None
    assert not files.stems_dir(rid).exists()
    assert files.normalized_path(rid).exists()

    files.delete_playback(rid)
    assert not files.normalized_path(rid).exists()

    files.delete_all(rid)
    assert not files.dir(rid).exists()
    files.delete_all(rid)  # idempotent
```

`api/tests/unit/test_normalize.py`:
```python
import pytest
import soundfile as sf
from audio_helpers import make_webm, write_tone

from saydo.pipeline.errors import AudioDecodeError
from saydo.pipeline.normalize import check_ffmpeg, normalize, probe_channels


def test_check_ffmpeg_passes_when_installed():
    check_ffmpeg()


def test_webm_becomes_16k_mono_wav(tmp_path):
    src = make_webm(tmp_path / "in.webm", seconds=2.0)
    out = normalize(src, tmp_path / "audio.wav", keep_channels=False)
    info = sf.info(out.path)
    assert (info.samplerate, info.channels, info.subtype) == (16000, 1, "PCM_16")
    assert out.channels == 1
    assert out.duration_s == pytest.approx(2.0, abs=0.1)


def test_keep_channels_preserves_stereo(tmp_path):
    src = write_tone(tmp_path / "in.wav", 1.0, sample_rate=44100, channels=2)
    assert probe_channels(src) == 2
    out = normalize(src, tmp_path / "audio.wav", keep_channels=True)
    assert out.channels == 2
    assert sf.info(out.path).samplerate == 16000


def test_garbage_file_raises_audio_decode_error(tmp_path):
    src = tmp_path / "in.webm"
    src.write_bytes(b"definitely not audio" * 100)
    with pytest.raises(AudioDecodeError):
        normalize(src, tmp_path / "audio.wav", keep_channels=False)
    assert not (tmp_path / "audio.wav").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd api && uv run pytest tests/unit/test_files.py tests/unit/test_normalize.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'saydo.storage'`.

- [ ] **Step 3: Implement**

`api/src/saydo/pipeline/errors.py`:
```python
"""Exceptions shared by the pipeline, plugins and worker.

Messages of PipelineError subclasses are shown to users, so they must never contain
transcript text or audio content.
"""


class PipelineError(RuntimeError):
    """A user-facing processing failure."""


class AudioDecodeError(PipelineError):
    """ffmpeg could not read the uploaded file."""


class PluginLoadError(PipelineError):
    """A model plugin could not be loaded (missing package, gated model, bad options)."""


class RecordingGone(Exception):
    """The recording was deleted while it was being processed."""
```

`api/src/saydo/storage/files.py`:
```python
"""On-disk layout: <data_dir>/recordings/<recording_id>/{original.<ext>, audio.wav, stems/}.

Every path is derived from a validated UUID and checked to resolve inside the
recordings root, so a crafted ID can never reach other files.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from saydo.ids import is_uuid

_EXT_RE = re.compile(r"^[a-z0-9]{1,5}$")


class InvalidRecordingId(ValueError):
    pass


class RecordingFiles:
    def __init__(self, data_dir: Path) -> None:
        self.root = (data_dir / "recordings").resolve()

    def dir(self, recording_id: str) -> Path:
        if not is_uuid(recording_id):
            raise InvalidRecordingId(recording_id)
        path = (self.root / recording_id).resolve()
        if path.parent != self.root:
            raise InvalidRecordingId(recording_id)
        return path

    def original_path(self, recording_id: str, ext: str) -> Path:
        ext = ext.lower()
        if not _EXT_RE.match(ext):
            ext = "bin"
        return self.dir(recording_id) / f"original.{ext}"

    def find_original(self, recording_id: str) -> Path | None:
        directory = self.dir(recording_id)
        if not directory.exists():
            return None
        return next(iter(sorted(directory.glob("original.*"))), None)

    def normalized_path(self, recording_id: str) -> Path:
        return self.dir(recording_id) / "audio.wav"

    def stems_dir(self, recording_id: str) -> Path:
        return self.dir(recording_id) / "stems"

    def delete_all(self, recording_id: str) -> None:
        directory = self.dir(recording_id)
        if directory.exists():
            shutil.rmtree(directory)

    def delete_raw(self, recording_id: str) -> None:
        original = self.find_original(recording_id)
        if original is not None:
            original.unlink(missing_ok=True)
        stems = self.stems_dir(recording_id)
        if stems.exists():
            shutil.rmtree(stems)

    def delete_playback(self, recording_id: str) -> None:
        self.normalized_path(recording_id).unlink(missing_ok=True)
```

`api/src/saydo/pipeline/normalize.py`:
```python
"""Convert any uploaded audio/video into 16 kHz 16-bit PCM WAV with ffmpeg."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf

from saydo.pipeline.errors import AudioDecodeError

TARGET_SAMPLE_RATE = 16000


@dataclass(frozen=True)
class NormalizedAudio:
    path: Path
    duration_s: float
    channels: int


def check_ffmpeg() -> None:
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        raise RuntimeError(
            f"{' and '.join(missing)} not found on PATH. Install ffmpeg "
            "(macOS: brew install ffmpeg; Debian/Ubuntu: apt-get install ffmpeg)."
        )


def _last_line(stderr: str) -> str:
    lines = [line for line in stderr.strip().splitlines() if line.strip()]
    return lines[-1][:200] if lines else "unknown error"


def probe_channels(src: Path) -> int:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=channels", "-of", "csv=p=0", str(src),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise AudioDecodeError(f"Could not read audio: {_last_line(result.stderr)}")
    value = result.stdout.strip().split(",")[0]
    if not value.isdigit():
        raise AudioDecodeError("The uploaded file has no audio stream")
    return int(value)


def normalize(src: Path, dst: Path, *, keep_channels: bool) -> NormalizedAudio:
    probe_channels(src)  # fail fast with a clear message on non-audio input
    tmp = dst.with_name(dst.stem + ".tmp.wav")
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src), "-vn", "-ar", str(TARGET_SAMPLE_RATE), "-c:a", "pcm_s16le",
    ]
    if not keep_channels:
        cmd += ["-ac", "1"]
    cmd.append(str(tmp))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if result.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise AudioDecodeError(f"Could not decode audio: {_last_line(result.stderr)}")
    tmp.replace(dst)
    info = sf.info(dst)
    return NormalizedAudio(path=dst, duration_s=float(info.duration), channels=int(info.channels))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd api && uv run pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add api/
git commit -m "feat(api): safe recording file layout and ffmpeg normalization"
```

---

### Task 4: Recordings REST API: upload, list, detail, status, audio, delete (M1)

**Files:**
- Create: `api/src/saydo/api/__init__.py` (empty), `api/src/saydo/api/errors.py`, `api/src/saydo/api/auth.py`, `api/src/saydo/api/deps.py`, `api/src/saydo/api/schemas.py`, `api/src/saydo/api/health.py`, `api/src/saydo/api/recordings.py`
- Create: `api/src/saydo/services.py`
- Modify: `api/src/saydo/main.py` (replace the whole file)
- Modify: `api/tests/conftest.py` (append the `app`, `client` and `wav_bytes` fixtures)
- Test: `api/tests/api/test_recordings.py`

**Interfaces:**
- Consumes: Tasks 1–3.
- Produces:
  - `create_app(settings=None, *, check_environment: bool = True) -> FastAPI`. It sets `app.state.settings`, `app.state.session_factory` and `app.state.files`.
  - `saydo.api.errors.ApiError(status: int, code: str, message: str)` and `install_error_handlers(app)`.
  - `saydo.api.auth`: `require_token`, `require_audio_access`, `sign_audio(token, recording_id, expires) -> str`.
  - `saydo.api.deps`: `SessionDep`, `FilesDep`, `SettingsDep`.
  - `saydo.api.schemas`: `RecordingCreated`, `RecordingSummary`, `RecordingOut`, `StatusOut`, `SpeakerOut`, `WordOut`, `SegmentOut`, `OverlapRegionOut`, `RecordingDetail`, `AudioUrlOut`.
  - `saydo.services`:
    - `delete_recording(session, files, recording_id) -> None`;
    - `load_detail(session, files, recording_id) -> RecordingDetail | None`;
    - `get_recording_or_404(session, recording_id) -> Recording`.
  - Endpoints:
    - `POST /api/v1/recordings`
    - `GET /api/v1/recordings`
    - `GET /api/v1/recordings/{id}`
    - `GET /api/v1/recordings/{id}/status`
    - `GET /api/v1/recordings/{id}/audio-url`
    - `GET /api/v1/recordings/{id}/audio`
    - `DELETE /api/v1/recordings/{id}`

- [ ] **Step 1: Write the failing tests**

Append to `api/tests/conftest.py`:
```python
import io  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from saydo.main import create_app  # noqa: E402


@pytest.fixture
def app(settings):
    return create_app(settings)


@pytest.fixture
def client(app):
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def wav_bytes() -> bytes:
    t = np.arange(16000 * 2) / 16000
    buffer = io.BytesIO()
    sf.write(buffer, (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), 16000, format="WAV")
    return buffer.getvalue()
```

`api/tests/api/test_recordings.py`:
```python
from audio_helpers import write_tone
from sqlalchemy import func, select

from saydo.db.models import Job, Recording


def upload(client, data: bytes, *, mime="audio/wav", **form):
    return client.post(
        "/api/v1/recordings",
        files={"file": ("meeting.wav", data, mime)},
        data={"source": "upload", **form},
    )


def make_ready(app, recording_id):
    write_tone(app.state.files.normalized_path(recording_id), 1.0)
    with app.state.session_factory() as s:
        s.get(Recording, recording_id).status = "ready"
        s.commit()


def test_upload_stores_file_and_queues_job(client, app, wav_bytes):
    response = upload(client, wav_bytes, title="Sprint planning", min_speakers="2", max_speakers="5")
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    rid = body["id"]
    assert app.state.files.find_original(rid).name == "original.wav"

    detail = client.get(f"/api/v1/recordings/{rid}").json()
    assert detail["recording"]["title"] == "Sprint planning"
    assert detail["recording"]["status"] == "queued"
    assert detail["recording"]["min_speakers"] == 2
    assert detail["recording"]["created_at"].endswith("Z")
    assert detail["speakers"] == [] and detail["segments"] == []
    with app.state.session_factory() as s:
        job = s.scalars(select(Job).where(Job.recording_id == rid)).one()
        assert (job.kind, job.status) == ("transcribe", "queued")


def test_codec_parameters_in_mime_are_accepted(client, wav_bytes):
    assert upload(client, wav_bytes, mime="audio/webm;codecs=opus").status_code == 202


def test_unsupported_type_is_415_and_leaves_nothing(client, app, wav_bytes):
    response = upload(client, wav_bytes, mime="text/plain")
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_media_type"
    assert not app.state.files.root.exists() or not any(app.state.files.root.iterdir())


def test_too_large_is_413_and_cleans_up(client, app, settings, wav_bytes):
    settings.upload.max_bytes = 1000
    response = upload(client, wav_bytes)
    assert response.status_code == 413
    assert not any(app.state.files.root.iterdir())
    with app.state.session_factory() as s:
        assert s.scalar(select(func.count()).select_from(Recording)) == 0


def test_empty_file_is_422(client):
    response = upload(client, b"")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "empty_file"


def test_min_speakers_above_max_is_422(client, wav_bytes):
    response = upload(client, wav_bytes, min_speakers="5", max_speakers="2")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_unknown_recording_is_404_with_error_envelope(client):
    response = client.get("/api/v1/recordings/0192f2a0-0000-7000-8000-000000000000")
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found", "message": "Recording not found"}}


def test_traversal_id_is_404(client):
    assert client.get("/api/v1/recordings/..%2F..%2Fetc").status_code == 404
    assert client.delete("/api/v1/recordings/not-a-uuid").status_code == 404
    assert client.get("/api/v1/recordings/not-a-uuid/audio").status_code == 404


def test_list_is_newest_first(client, wav_bytes):
    first = upload(client, wav_bytes, title="first").json()["id"]
    second = upload(client, wav_bytes, title="second").json()["id"]
    ids = [r["id"] for r in client.get("/api/v1/recordings").json()]
    assert ids[:2] == [second, first]


def test_status_endpoint(client, wav_bytes):
    rid = upload(client, wav_bytes).json()["id"]
    assert client.get(f"/api/v1/recordings/{rid}/status").json() == {
        "status": "queued", "stage": None, "error": None,
    }


def test_delete_removes_files_and_rows_and_is_idempotent(client, app, wav_bytes):
    rid = upload(client, wav_bytes).json()["id"]
    assert client.delete(f"/api/v1/recordings/{rid}").status_code == 204
    assert not app.state.files.dir(rid).exists()
    with app.state.session_factory() as s:
        assert s.get(Recording, rid) is None
        assert s.scalar(select(func.count()).select_from(Job)) == 0
    assert client.delete(f"/api/v1/recordings/{rid}").status_code == 204


def test_retention_days_sets_retain_until(client, settings, wav_bytes):
    settings.retention.days = 7
    rid = upload(client, wav_bytes).json()["id"]
    recording = client.get(f"/api/v1/recordings/{rid}").json()["recording"]
    assert recording["retain_until"] is not None


def test_audio_supports_range_requests(client, app, wav_bytes):
    rid = upload(client, wav_bytes).json()["id"]
    assert client.get(f"/api/v1/recordings/{rid}/audio").status_code == 409
    make_ready(app, rid)
    url = client.get(f"/api/v1/recordings/{rid}/audio-url").json()["url"]
    response = client.get(url, headers={"Range": "bytes=0-99"})
    assert response.status_code == 206
    assert len(response.content) == 100


def test_audio_deleted_by_retention_is_410(client, app, wav_bytes):
    rid = upload(client, wav_bytes).json()["id"]
    make_ready(app, rid)
    app.state.files.delete_playback(rid)
    assert client.get(f"/api/v1/recordings/{rid}/audio").status_code == 410
    assert client.get(f"/api/v1/recordings/{rid}").json()["recording"]["playback_available"] is False


def test_token_protects_api_but_not_health(client, settings, wav_bytes):
    settings.api.token = "s3cret"
    assert client.get("/api/v1/health").status_code == 200
    assert client.get("/api/v1/recordings").status_code == 401
    assert client.get("/api/v1/recordings", headers={"Authorization": "Bearer wrong"}).status_code == 401
    ok = client.get("/api/v1/recordings", headers={"Authorization": "Bearer s3cret"})
    assert ok.status_code == 200


def test_signed_audio_url_works_without_header(client, app, settings, wav_bytes):
    rid = upload(client, wav_bytes).json()["id"]
    make_ready(app, rid)
    settings.api.token = "s3cret"
    auth = {"Authorization": "Bearer s3cret"}
    url = client.get(f"/api/v1/recordings/{rid}/audio-url", headers=auth).json()["url"]
    assert "sig=" in url
    assert client.get(url).status_code == 200
    assert client.get(url.replace("sig=", "sig=0")).status_code == 401
    assert client.get(f"/api/v1/recordings/{rid}/audio").status_code == 401
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd api && uv run pytest tests/api -q`
Expected: FAIL. The recordings tests get 404 on `/api/v1/recordings` (route missing), and the import of `saydo.api` fails.

- [ ] **Step 3: Implement errors, auth, deps and schemas**

`api/src/saydo/api/errors.py`:
```python
"""Uniform error envelope: {"error": {"code": ..., "message": ...}}."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from saydo.storage.files import InvalidRecordingId


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def _envelope(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        return _envelope(exc.status, exc.code, exc.message)

    @app.exception_handler(InvalidRecordingId)
    async def _invalid_id(_: Request, __: InvalidRecordingId) -> JSONResponse:
        return _envelope(404, "not_found", "Recording not found")

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        message = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        return _envelope(422, "validation_error", message)

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = "not_found" if exc.status_code == 404 else "http_error"
        return _envelope(exc.status_code, code, str(exc.detail))
```

`api/src/saydo/api/auth.py`:
```python
"""Optional static bearer token, plus HMAC-signed audio URLs for <audio> elements
(which cannot send an Authorization header)."""

from __future__ import annotations

import hashlib
import hmac
import time

from fastapi import Request

from saydo.api.errors import ApiError


def _bearer_ok(request: Request, expected: str) -> bool:
    scheme, _, value = request.headers.get("authorization", "").partition(" ")
    return scheme.lower() == "bearer" and hmac.compare_digest(value.encode(), expected.encode())


def require_token(request: Request) -> None:
    expected = request.app.state.settings.api.token
    if expected and not _bearer_ok(request, expected):
        raise ApiError(401, "unauthorized", "Missing or invalid bearer token")


def sign_audio(token: str, recording_id: str, expires: int) -> str:
    message = f"{recording_id}:{expires}".encode()
    return hmac.new(token.encode(), message, hashlib.sha256).hexdigest()


def require_audio_access(
    request: Request, recording_id: str, expires: int | None = None, sig: str | None = None
) -> None:
    expected = request.app.state.settings.api.token
    if not expected or _bearer_ok(request, expected):
        return
    if (
        expires is not None
        and sig
        and expires >= int(time.time())
        and hmac.compare_digest(sig, sign_audio(expected, recording_id, expires))
    ):
        return
    raise ApiError(401, "unauthorized", "Missing or invalid audio signature")
```

`api/src/saydo/api/deps.py`:
```python
from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from saydo.config import Settings
from saydo.storage.files import RecordingFiles


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_files(request: Request) -> RecordingFiles:
    return request.app.state.files


def get_session(request: Request) -> Iterator[Session]:
    with request.app.state.session_factory() as session:
        yield session


SettingsDep = Annotated[Settings, Depends(get_settings)]
FilesDep = Annotated[RecordingFiles, Depends(get_files)]
SessionDep = Annotated[Session, Depends(get_session)]
```

`api/src/saydo/api/schemas.py`:
```python
"""Response/request models. These names become the TypeScript types in @saydo/react."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, PlainSerializer


def _iso_utc(value: datetime) -> str:
    return value.replace(tzinfo=UTC).isoformat().replace("+00:00", "Z")


UtcDatetime = Annotated[datetime, PlainSerializer(_iso_utc, return_type=str)]

RecordingStatus = Literal["uploaded", "queued", "processing", "ready", "failed"]
RecordingSource = Literal["mic", "tab", "mic+tab", "upload"]
SegmentOrigin = Literal["mixture", "separated", "placeholder"]


class RecordingCreated(BaseModel):
    id: str
    status: RecordingStatus


class RecordingSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    status: RecordingStatus
    source: RecordingSource
    created_at: UtcDatetime
    duration_s: float | None


class RecordingOut(RecordingSummary):
    stage: str | None
    error: str | None
    channels: int | None
    language: str | None
    min_speakers: int | None
    max_speakers: int | None
    transcript_version: int
    raw_audio_deleted_at: UtcDatetime | None
    retain_until: UtcDatetime | None
    playback_available: bool


class StatusOut(BaseModel):
    status: RecordingStatus
    stage: str | None
    error: str | None


class SpeakerOut(BaseModel):
    id: str
    label: str
    display_name: str | None
    name: str
    color_index: int


class WordOut(BaseModel):
    start_s: float
    end_s: float
    text: str
    prob: float


class SegmentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    speaker_id: str
    original_speaker_id: str
    start_s: float
    end_s: float
    text: str
    words: list[WordOut]
    origin: SegmentOrigin
    is_overlap: bool
    overlap_region_id: str | None
    confidence: float
    confidence_parts: dict[str, float]
    needs_review: bool
    user_corrected: bool
    reviewed: bool


class OverlapRegionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    start_s: float
    end_s: float
    n_speakers: int


class RecordingDetail(BaseModel):
    recording: RecordingOut
    speakers: list[SpeakerOut]
    overlap_regions: list[OverlapRegionOut]
    segments: list[SegmentOut]


class AudioUrlOut(BaseModel):
    url: str
    expires_at: int
```

`api/src/saydo/api/health.py`:
```python
from fastapi import APIRouter

from saydo import __version__

router = APIRouter(prefix="/api/v1", tags=["health"])


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}
```

- [ ] **Step 4: Implement services and routes**

`api/src/saydo/services.py`:
```python
"""Use cases shared by API routes, the worker and the retention sweeper."""

from __future__ import annotations

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from saydo.api.errors import ApiError
from saydo.api.schemas import (
    OverlapRegionOut,
    RecordingDetail,
    RecordingOut,
    SegmentOut,
    SpeakerOut,
)
from saydo.db.models import Job, OverlapRegion, Recording, Segment, Speaker
from saydo.storage.files import RecordingFiles


def get_recording_or_404(session: Session, recording_id: str) -> Recording:
    recording = session.get(Recording, recording_id)
    if recording is None:
        raise ApiError(404, "not_found", "Recording not found")
    return recording


def delete_recording(session: Session, files: RecordingFiles, recording_id: str) -> None:
    """Hard delete, in the order of spec section 12: pending jobs, files, rows."""
    files.dir(recording_id)  # validates the id before touching anything
    session.execute(
        delete(Job).where(Job.recording_id == recording_id, Job.status == "queued")
    )
    session.commit()
    files.delete_all(recording_id)
    session.execute(delete(Recording).where(Recording.id == recording_id))
    session.commit()


def _speaker_number(speaker: Speaker) -> int:
    return int(speaker.label.rsplit(" ", 1)[-1])


def speaker_out(speaker: Speaker) -> SpeakerOut:
    return SpeakerOut(
        id=speaker.id,
        label=speaker.label,
        display_name=speaker.display_name,
        name=speaker.display_name or speaker.label,
        color_index=speaker.color_index,
    )


def load_detail(session: Session, files: RecordingFiles, recording_id: str) -> RecordingDetail | None:
    recording = session.get(Recording, recording_id)
    if recording is None:
        return None
    speakers = session.scalars(
        select(Speaker).where(
            Speaker.recording_id == recording_id, Speaker.merged_into_id.is_(None)
        )
    ).all()
    regions = session.scalars(
        select(OverlapRegion)
        .where(OverlapRegion.recording_id == recording_id)
        .order_by(OverlapRegion.start_s)
    ).all()
    segments = session.scalars(
        select(Segment)
        .where(Segment.recording_id == recording_id)
        .order_by(Segment.start_s, Segment.end_s)
    ).all()
    playback = recording.status == "ready" and files.normalized_path(recording_id).exists()
    return RecordingDetail(
        recording=RecordingOut.model_validate(
            {**_columns(recording), "playback_available": playback}
        ),
        speakers=[speaker_out(s) for s in sorted(speakers, key=_speaker_number)],
        overlap_regions=[OverlapRegionOut.model_validate(r) for r in regions],
        segments=[SegmentOut.model_validate(s) for s in segments],
    )


def _columns(recording: Recording) -> dict[str, object]:
    return {c.key: getattr(recording, c.key) for c in Recording.__table__.columns}
```

`api/src/saydo/api/recordings.py`:
```python
"""Recording lifecycle endpoints: upload, list, detail, status, audio, delete."""

from __future__ import annotations

import time
from datetime import timedelta
from pathlib import PurePath
from typing import Annotated

from fastapi import APIRouter, Depends, Form, Response, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import select

from saydo.api.auth import require_audio_access, require_token, sign_audio
from saydo.api.deps import FilesDep, SessionDep, SettingsDep
from saydo.api.errors import ApiError
from saydo.api.schemas import (
    AudioUrlOut,
    RecordingCreated,
    RecordingDetail,
    RecordingSource,
    RecordingSummary,
    StatusOut,
)
from saydo.db.models import Job, Recording, utcnow
from saydo.ids import uuid7
from saydo.services import delete_recording, get_recording_or_404, load_detail

router = APIRouter(prefix="/api/v1", tags=["recordings"], dependencies=[Depends(require_token)])
audio_router = APIRouter(prefix="/api/v1", tags=["recordings"])

EXT_BY_MIME = {
    "audio/webm": "webm", "video/webm": "webm", "audio/ogg": "ogg",
    "audio/mp4": "m4a", "audio/x-m4a": "m4a", "video/mp4": "mp4", "audio/aac": "aac",
    "audio/mpeg": "mp3", "audio/wav": "wav", "audio/x-wav": "wav", "audio/wave": "wav",
    "audio/flac": "flac", "audio/x-flac": "flac",
}
CHUNK = 1024 * 1024


def _extension(mime: str, filename: str | None) -> str:
    if mime in EXT_BY_MIME:
        return EXT_BY_MIME[mime]
    suffix = PurePath(filename or "").suffix.lstrip(".").lower()
    return suffix or "bin"


@router.post("/recordings", status_code=202, response_model=RecordingCreated)
def create_recording(
    file: UploadFile,
    session: SessionDep,
    files: FilesDep,
    settings: SettingsDep,
    source: Annotated[RecordingSource, Form()] = "upload",
    title: Annotated[str | None, Form(max_length=200)] = None,
    min_speakers: Annotated[int | None, Form(ge=1, le=20)] = None,
    max_speakers: Annotated[int | None, Form(ge=1, le=20)] = None,
    language: Annotated[str | None, Form(pattern=r"^[a-z]{2,3}$")] = None,
) -> RecordingCreated:
    if min_speakers and max_speakers and min_speakers > max_speakers:
        raise ApiError(422, "validation_error", "min_speakers must be <= max_speakers")
    mime = (file.content_type or "").split(";")[0].strip().lower()
    if not any(mime.startswith(p) for p in settings.upload.allowed_mime_prefixes):
        raise ApiError(415, "unsupported_media_type", f"Unsupported content type: {mime or 'unknown'}")

    recording_id = uuid7()
    destination = files.original_path(recording_id, _extension(mime, file.filename))
    destination.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    try:
        with destination.open("wb") as out:
            while chunk := file.file.read(CHUNK):
                size += len(chunk)
                if size > settings.upload.max_bytes:
                    raise ApiError(413, "payload_too_large", "The file exceeds the upload size limit")
                out.write(chunk)
        if size == 0:
            raise ApiError(422, "empty_file", "Uploaded file is empty")
    except BaseException:
        files.delete_all(recording_id)
        raise

    now = utcnow()
    days = settings.retention.days
    session.add(
        Recording(
            id=recording_id,
            title=(title or "").strip() or f"Meeting {now:%Y-%m-%d %H:%M} UTC",
            created_at=now,
            source=source,
            original_filename=(file.filename or "")[:255] or None,
            mime_type=mime,
            status="queued",
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            language=language,
            retain_until=now + timedelta(days=days) if days else None,
        )
    )
    session.flush()
    session.add(Job(recording_id=recording_id, kind="transcribe", status="queued"))
    session.commit()
    return RecordingCreated(id=recording_id, status="queued")


@router.get("/recordings", response_model=list[RecordingSummary])
def list_recordings(session: SessionDep) -> list[Recording]:
    return list(
        session.scalars(
            select(Recording).order_by(Recording.created_at.desc(), Recording.id.desc()).limit(200)
        )
    )


@router.get("/recordings/{recording_id}", response_model=RecordingDetail)
def get_recording(recording_id: str, session: SessionDep, files: FilesDep) -> RecordingDetail:
    detail = load_detail(session, files, recording_id)
    if detail is None:
        raise ApiError(404, "not_found", "Recording not found")
    return detail


@router.get("/recordings/{recording_id}/status", response_model=StatusOut)
def get_status(recording_id: str, session: SessionDep) -> StatusOut:
    recording = get_recording_or_404(session, recording_id)
    return StatusOut(status=recording.status, stage=recording.stage, error=recording.error)


@router.get("/recordings/{recording_id}/audio-url", response_model=AudioUrlOut)
def get_audio_url(recording_id: str, session: SessionDep, settings: SettingsDep) -> AudioUrlOut:
    get_recording_or_404(session, recording_id)
    expires = int(time.time()) + 3600
    url = f"/api/v1/recordings/{recording_id}/audio"
    if settings.api.token:
        url += f"?expires={expires}&sig={sign_audio(settings.api.token, recording_id, expires)}"
    return AudioUrlOut(url=url, expires_at=expires)


@audio_router.get(
    "/recordings/{recording_id}/audio",
    dependencies=[Depends(require_audio_access)],
    response_class=FileResponse,
)
def get_audio(recording_id: str, session: SessionDep, files: FilesDep) -> FileResponse:
    path = files.normalized_path(recording_id)  # validates the id first
    recording = get_recording_or_404(session, recording_id)
    if recording.status != "ready":
        raise ApiError(409, "not_ready", "Audio is available once processing has finished")
    if not path.exists():
        raise ApiError(410, "audio_deleted", "Playback audio was deleted by the retention policy")
    return FileResponse(path, media_type="audio/wav")


@router.delete("/recordings/{recording_id}", status_code=204)
def remove_recording(recording_id: str, session: SessionDep, files: FilesDep) -> Response:
    delete_recording(session, files, recording_id)
    return Response(status_code=204)
```

Replace `api/src/saydo/main.py`:
```python
"""FastAPI application factory and the `saydo-api` CLI entry point."""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from saydo import __version__
from saydo.api import health, recordings
from saydo.api.errors import install_error_handlers
from saydo.config import Settings
from saydo.db.session import make_engine, make_session_factory, migrate
from saydo.pipeline.normalize import check_ffmpeg
from saydo.storage.files import RecordingFiles


def create_app(settings: Settings | None = None, *, check_environment: bool = True) -> FastAPI:
    settings = settings or Settings()
    if check_environment:
        check_ffmpeg()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    engine = make_engine(settings.resolved_database_url)
    migrate(engine)

    app = FastAPI(
        title="SayDo API",
        version=__version__,
        openapi_url="/api/v1/openapi.json",
        docs_url="/api/v1/docs",
        redoc_url=None,
    )
    app.state.settings = settings
    app.state.session_factory = make_session_factory(engine)
    app.state.files = RecordingFiles(settings.data_dir)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Accept-Ranges", "Content-Length", "Content-Range"],
    )
    install_error_handlers(app)
    app.include_router(health.router)
    app.include_router(recordings.router)
    app.include_router(recordings.audio_router)
    return app


def run() -> None:
    import uvicorn

    uvicorn.run(
        "saydo.main:create_app",
        factory=True,
        host=os.environ.get("SAYDO_HOST", "127.0.0.1"),
        port=int(os.environ.get("SAYDO_PORT", "8000")),
    )
```

Update `api/tests/api/test_health.py`. Both tests already build the app with `create_app(settings)`. That still works, because `settings.data_dir` is under `tmp_path` and ffmpeg is installed, so no change is needed. Confirm by running it in Step 5.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd api && uv run pytest -q`
Expected: all pass. If `test_audio_supports_range_requests` returns 200 instead of 206, upgrade Starlette (`uv lock --upgrade-package starlette`). Range support in `FileResponse` needs Starlette >= 0.39.

- [ ] **Step 6: Commit**

```bash
git add api/
git commit -m "feat(api): recording upload, detail, signed audio playback and hard delete"
```

---

### Task 5: Pipeline types, plugin interfaces, registry and fake plugins (M2)

**Files:**
- Create: `api/src/saydo/pipeline/types.py`, `api/src/saydo/pipeline/interfaces.py`, `api/src/saydo/pipeline/registry.py`
- Create: `api/src/saydo/plugins/__init__.py` (empty), `api/src/saydo/plugins/fake.py`
- Test: `api/tests/unit/test_types.py`, `api/tests/unit/test_registry.py`, `api/tests/unit/test_fake_plugins.py`

**Interfaces:**
- Consumes: `PluginSettings` and `Settings` (Task 1); `PluginLoadError` (Task 3).
- Produces:
  - `saydo.pipeline.types`:
    - `Word(start_s, end_s, text, prob=1.0, asr_score=1.0)`
    - `AsrSegment(start_s, end_s, text, words: tuple[Word, ...], avg_logprob, no_speech_prob)`
    - `Turn(speaker_key, start_s, end_s)`
    - `Diarization(turns: tuple[Turn, ...])`
    - `Audio(samples: np.ndarray[(channels, n)], sample_rate)` with `.channels`, `.duration_s`, `.mono() -> np.ndarray` and `.slice(start_s, end_s) -> np.ndarray` (mono)
    - `load_audio(path) -> Audio`
  - `saydo.pipeline.interfaces`: the `Transcriber`, `Diarizer` (with `wants_multichannel: bool`) and `Embedder` protocols, plus `PluginFactory = Callable[[dict[str, Any], str], Any]`.
  - `saydo.pipeline.registry`:
    - `PLUGIN_KINDS = ("transcriber", "diarizer", "embedder")`
    - `PluginNotFoundError(LookupError)`
    - `resolve_factory(kind, name) -> PluginFactory`
    - `build_plugin(kind, plugin_settings, device) -> Any`
    - `check_plugins(settings) -> None`
    - `resolve_device(setting: str) -> str`
  - Built-in plugin names:
    - transcriber: `faster_whisper`, `fake`
    - diarizer: `pyannote`, `multichannel`, `fake`
    - embedder: `pyannote`, `fake`

    The real plugin modules are written in Task 11; the registry only stores their import paths.
  - `saydo.plugins.fake`: `FakeTranscriber`, `FakeDiarizer`, `FakeEmbedder`, and the factories `create_transcriber`, `create_diarizer`, `create_embedder`.
    - **Fake diarizer:** 4 s turns every 3 s, alternating `SPK0` and `SPK1`, so neighbouring turns overlap by 1 s.
    - **Fake transcriber:** words ` word1`, ` word2`, … every 0.5 s, each lasting 0.4 s, grouped into 3 s ASR segments with `avg_logprob=-0.2` and `no_speech_prob=0.01`.
    - **Both** skip any window whose RMS is below `1e-4`, so silence yields nothing.
    - **Fake embedder:** returns the constant vector `ones(8)`.

- [ ] **Step 1: Write the failing tests**

`api/tests/unit/test_types.py`:
```python
import numpy as np
import pytest
from audio_helpers import write_tone

from saydo.pipeline.types import Audio, load_audio


def test_load_audio_shapes(tmp_path):
    audio = load_audio(write_tone(tmp_path / "a.wav", 2.0, channels=2))
    assert audio.samples.shape == (2, 32000)
    assert audio.samples.dtype == np.float32
    assert audio.channels == 2
    assert audio.duration_s == pytest.approx(2.0)
    assert audio.mono().shape == (32000,)


def test_slice_is_clamped():
    audio = Audio(samples=np.ones((1, 16000), dtype=np.float32), sample_rate=16000)
    assert audio.slice(0.5, 0.75).shape == (4000,)
    assert audio.slice(0.9, 5.0).shape == (1600,)
    assert audio.slice(-1.0, 0.1).shape == (1600,)
```

`api/tests/unit/test_registry.py`:
```python
from importlib.metadata import EntryPoint

import pytest

from saydo.config import PluginSettings, Settings
from saydo.pipeline import registry
from saydo.pipeline.registry import (
    PluginNotFoundError,
    build_plugin,
    check_plugins,
    resolve_device,
    resolve_factory,
)
from saydo.plugins.fake import FakeTranscriber


def test_builtin_fake_resolves_and_builds():
    plugin = build_plugin("transcriber", PluginSettings(name="fake", options={"word_s": 0.25}), "cpu")
    assert isinstance(plugin, FakeTranscriber)
    assert plugin.word_s == 0.25


def test_unknown_plugin_names_the_problem():
    with pytest.raises(PluginNotFoundError, match="No diarizer plugin named 'nope'"):
        resolve_factory("diarizer", "nope")


def test_third_party_entry_point(monkeypatch):
    ep = EntryPoint(name="transcriber:mine", value="saydo.plugins.fake:create_transcriber",
                    group="saydo.plugins")
    monkeypatch.setattr(registry, "entry_points", lambda group: [ep] if group == "saydo.plugins" else [])
    assert resolve_factory("transcriber", "mine") is not None


def test_check_plugins_fails_fast_on_bad_name(settings):
    check_plugins(settings)  # fake plugins resolve
    bad = Settings(
        data_dir=settings.data_dir,
        pipeline={"transcriber": {"name": "fake"}, "diarizer": {"name": "fake"}, "embedder": {"name": "missing"}},
    )
    with pytest.raises(PluginNotFoundError):
        check_plugins(bad)


def test_explicit_device_is_returned_unchanged():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda"
    assert resolve_device("auto") in {"cpu", "cuda", "mps"}
```

`api/tests/unit/test_fake_plugins.py`:
```python
import numpy as np

from saydo.pipeline.types import Audio
from saydo.plugins.fake import FakeDiarizer, FakeEmbedder, FakeTranscriber


def tone(seconds: float) -> Audio:
    t = np.arange(int(seconds * 16000)) / 16000
    return Audio(samples=(0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)[None, :], sample_rate=16000)


def silence(seconds: float) -> Audio:
    return Audio(samples=np.zeros((1, int(seconds * 16000)), dtype=np.float32), sample_rate=16000)


def test_fake_diarizer_alternates_with_one_second_overlaps():
    turns = FakeDiarizer().diarize(tone(10.0)).turns
    assert [(t.speaker_key, t.start_s, t.end_s) for t in turns] == [
        ("SPK0", 0.0, 4.0), ("SPK1", 3.0, 7.0), ("SPK0", 6.0, 10.0), ("SPK1", 9.0, 10.0),
    ]


def test_fake_transcriber_emits_words_every_half_second():
    segments = FakeTranscriber().transcribe(tone(3.0))
    assert len(segments) == 1
    words = segments[0].words
    assert [w.start_s for w in words] == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    assert words[0].text == " word1"
    assert segments[0].text == "word1 word2 word3 word4 word5 word6"


def test_fake_plugins_return_nothing_for_silence():
    assert FakeDiarizer().diarize(silence(5.0)).turns == ()
    assert FakeTranscriber().transcribe(silence(5.0)) == []


def test_fake_embedder_is_constant():
    assert np.array_equal(FakeEmbedder().embed(np.zeros(100, dtype=np.float32), 16000), np.ones(8))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd api && uv run pytest tests/unit/test_types.py tests/unit/test_registry.py tests/unit/test_fake_plugins.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'saydo.pipeline.types'`.

- [ ] **Step 3: Implement**

`api/src/saydo/pipeline/types.py`:
```python
"""Value types exchanged between plugins and the pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass(frozen=True, slots=True)
class Word:
    start_s: float
    end_s: float
    text: str  # Whisper-style: usually carries a leading space
    prob: float = 1.0
    asr_score: float = 1.0  # score of the ASR segment the word came from (set by the pipeline)


@dataclass(frozen=True)
class AsrSegment:
    start_s: float
    end_s: float
    text: str
    words: tuple[Word, ...]
    avg_logprob: float
    no_speech_prob: float


@dataclass(frozen=True, slots=True)
class Turn:
    speaker_key: str
    start_s: float
    end_s: float


@dataclass(frozen=True)
class Diarization:
    turns: tuple[Turn, ...]  # may overlap in time


@dataclass(frozen=True, eq=False)
class Audio:
    samples: np.ndarray  # shape (channels, n), float32 in [-1, 1]
    sample_rate: int

    @property
    def channels(self) -> int:
        return int(self.samples.shape[0])

    @property
    def duration_s(self) -> float:
        return self.samples.shape[1] / self.sample_rate

    def mono(self) -> np.ndarray:
        if self.channels == 1:
            return self.samples[0]
        return self.samples.mean(axis=0).astype(np.float32)

    def slice(self, start_s: float, end_s: float) -> np.ndarray:
        n = self.samples.shape[1]
        start = min(max(int(start_s * self.sample_rate), 0), n)
        end = min(max(int(end_s * self.sample_rate), start), n)
        return self.mono()[start:end]


def load_audio(path: Path) -> Audio:
    data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    return Audio(samples=np.ascontiguousarray(data.T), sample_rate=int(sample_rate))
```

`api/src/saydo/pipeline/interfaces.py`:
```python
"""Plugin contracts. Implementations are chosen by name in config (pipeline.<kind>.name).

A plugin factory has the signature `create(options: dict, device: str) -> plugin`, where
`device` is "cpu", "cuda" or "mps". Third-party packages register factories under the
entry-point group "saydo.plugins" with names like "transcriber:my_asr".
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import numpy as np

from saydo.pipeline.types import AsrSegment, Audio, Diarization


class Transcriber(Protocol):
    name: str

    def transcribe(self, audio: Audio, *, language: str | None = None) -> list[AsrSegment]: ...


class Diarizer(Protocol):
    name: str
    wants_multichannel: bool  # True: the pipeline keeps all channels when normalizing

    def diarize(
        self, audio: Audio, *, min_speakers: int | None = None, max_speakers: int | None = None
    ) -> Diarization: ...


class Embedder(Protocol):
    name: str

    def embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray: ...


PluginFactory = Callable[[dict[str, Any], str], Any]
```

`api/src/saydo/pipeline/registry.py`:
```python
"""Resolve plugin names to factories: built-ins first, then 'saydo.plugins' entry points."""

from __future__ import annotations

import importlib
from importlib.metadata import entry_points
from typing import Any, Literal

from saydo.config import PluginSettings, Settings
from saydo.pipeline.interfaces import PluginFactory

PluginKind = Literal["transcriber", "diarizer", "embedder"]
PLUGIN_KINDS: tuple[PluginKind, ...] = ("transcriber", "diarizer", "embedder")
ENTRY_POINT_GROUP = "saydo.plugins"

BUILTINS: dict[str, dict[str, str]] = {
    "transcriber": {
        "faster_whisper": "saydo.plugins.faster_whisper_transcriber:create",
        "fake": "saydo.plugins.fake:create_transcriber",
    },
    "diarizer": {
        "pyannote": "saydo.plugins.pyannote_diarizer:create",
        "multichannel": "saydo.plugins.multichannel_diarizer:create",
        "fake": "saydo.plugins.fake:create_diarizer",
    },
    "embedder": {
        "pyannote": "saydo.plugins.pyannote_embedder:create",
        "fake": "saydo.plugins.fake:create_embedder",
    },
}


class PluginNotFoundError(LookupError):
    pass


def _load(target: str) -> PluginFactory:
    module_name, _, attr = target.partition(":")
    return getattr(importlib.import_module(module_name), attr)


def resolve_factory(kind: str, name: str) -> PluginFactory:
    builtin = BUILTINS.get(kind, {}).get(name)
    if builtin is not None:
        return _load(builtin)
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        if ep.name == f"{kind}:{name}":
            return ep.load()
    known = ", ".join(sorted(BUILTINS.get(kind, {})))
    raise PluginNotFoundError(
        f"No {kind} plugin named '{name}'. Built-in: {known}. Third-party plugins register an "
        f"entry point in group '{ENTRY_POINT_GROUP}' named '{kind}:<name>'."
    )


def build_plugin(kind: str, plugin: PluginSettings, device: str) -> Any:
    return resolve_factory(kind, plugin.name)(dict(plugin.options), device)


def check_plugins(settings: Settings) -> None:
    """Fail at startup if a configured plugin name does not resolve (models are not loaded)."""
    for kind in PLUGIN_KINDS:
        resolve_factory(kind, getattr(settings.pipeline, kind).name)


def resolve_device(setting: str) -> str:
    if setting != "auto":
        return setting
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
```

`api/src/saydo/plugins/fake.py`:
```python
"""Deterministic plugins for tests, CI and UI development. Never use them for real meetings.

FakeDiarizer: 4 s turns every 3 s, alternating SPK0/SPK1, so neighbours overlap by 1 s.
FakeTranscriber: a word every 0.5 s (" word1", " word2", ...) in 3 s ASR segments.
Both treat windows with RMS < 1e-4 as silence and emit nothing for them.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from saydo.pipeline.types import AsrSegment, Audio, Diarization, Turn, Word

SILENCE_RMS = 1e-4


def _is_silent(audio: Audio, start_s: float, end_s: float) -> bool:
    window = audio.slice(start_s, end_s)
    return window.size == 0 or float(np.sqrt(np.mean(window**2))) < SILENCE_RMS


class FakeDiarizer:
    name = "fake"
    wants_multichannel = False

    def __init__(self, turn_s: float = 4.0, step_s: float = 3.0, speakers: int = 2) -> None:
        self.turn_s, self.step_s, self.speakers = turn_s, step_s, speakers

    def diarize(
        self, audio: Audio, *, min_speakers: int | None = None, max_speakers: int | None = None
    ) -> Diarization:
        turns: list[Turn] = []
        duration = audio.duration_s
        start, index = 0.0, 0
        while start < duration:
            end = min(start + self.turn_s, duration)
            if not _is_silent(audio, start, end):
                turns.append(Turn(f"SPK{index % self.speakers}", round(start, 3), round(end, 3)))
            start += self.step_s
            index += 1
        return Diarization(tuple(turns))


class FakeTranscriber:
    name = "fake"

    def __init__(self, word_s: float = 0.5, window_s: float = 3.0) -> None:
        self.word_s, self.window_s = word_s, window_s

    def transcribe(self, audio: Audio, *, language: str | None = None) -> list[AsrSegment]:
        segments: list[AsrSegment] = []
        duration = audio.duration_s
        start, counter = 0.0, 0
        while start < duration:
            end = min(start + self.window_s, duration)
            words: list[Word] = []
            if not _is_silent(audio, start, end):
                t = start
                while t + self.word_s <= end + 1e-9:
                    counter += 1
                    words.append(Word(round(t, 3), round(t + self.word_s * 0.8, 3), f" word{counter}", 0.9))
                    t += self.word_s
            if words:
                text = "".join(w.text for w in words).strip()
                segments.append(AsrSegment(start, end, text, tuple(words), -0.2, 0.01))
            start = end
        return segments


class FakeEmbedder:
    name = "fake"

    def embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        return np.ones(8, dtype=np.float32)


def create_transcriber(options: dict[str, Any], device: str) -> FakeTranscriber:
    return FakeTranscriber(**options)


def create_diarizer(options: dict[str, Any], device: str) -> FakeDiarizer:
    return FakeDiarizer(**options)


def create_embedder(options: dict[str, Any], device: str) -> FakeEmbedder:
    return FakeEmbedder()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd api && uv run pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add api/
git commit -m "feat(pipeline): plugin interfaces, registry with entry points, fake plugins"
```

---

### Task 6: Overlap region detection (M3)

**Files:**
- Create: `api/src/saydo/pipeline/overlap.py`
- Test: `api/tests/unit/test_overlap.py`

**Interfaces:**
- Consumes: `Turn` (Task 5).
- Produces:
  - `Region(start_s, end_s, speaker_keys: frozenset[str])` with `.duration_s`.
  - `find_overlap_regions(turns, *, min_duration_s=0.2, merge_gap_s=0.1) -> list[Region]`, sorted by start.
  - `region_index_at(regions, t: float) -> int | None`.

- [ ] **Step 1: Write the failing tests**

`api/tests/unit/test_overlap.py`:
```python
import pytest

from saydo.pipeline.overlap import Region, find_overlap_regions, region_index_at
from saydo.pipeline.types import Turn


def regions(*turns):
    return find_overlap_regions([Turn(*t) for t in turns])


def test_no_overlap():
    assert regions(("A", 0, 3), ("B", 3.5, 5)) == []


def test_touching_turns_do_not_overlap():
    assert regions(("A", 0, 3), ("B", 3, 5)) == []


def test_simple_overlap():
    assert regions(("A", 0, 4), ("B", 3, 7)) == [Region(3, 4, frozenset({"A", "B"}))]


def test_short_overlap_is_ignored():
    assert regions(("A", 0, 3.1), ("B", 3, 5)) == []


def test_close_overlaps_merge_and_far_ones_do_not():
    merged = regions(("A", 0, 10), ("B", 2, 3), ("C", 3.05, 4))
    assert merged == [Region(2, 4, frozenset({"A", "B", "C"}))]
    separate = regions(("A", 0, 10), ("B", 2, 3), ("C", 3.5, 4))
    assert [(r.start_s, r.end_s) for r in separate] == [(2, 3), (3.5, 4)]


def test_three_way_overlap_is_one_region():
    (region,) = regions(("A", 0, 10), ("B", 2, 6), ("C", 4, 8))
    assert (region.start_s, region.end_s) == (2, 8)
    assert region.speaker_keys == {"A", "B", "C"}


def test_same_speaker_overlapping_turns_are_not_overlap():
    assert regions(("A", 0, 4), ("A", 3, 6)) == []


def test_zero_length_turns_are_ignored():
    assert regions(("A", 0, 4), ("B", 2, 2)) == []


def test_region_index_at():
    rs = [Region(1, 2, frozenset({"A", "B"})), Region(5, 6, frozenset({"A", "B"}))]
    assert region_index_at(rs, 0.5) is None
    assert region_index_at(rs, 1.0) == 0
    assert region_index_at(rs, 1.99) == 0
    assert region_index_at(rs, 2.0) is None
    assert region_index_at(rs, 5.5) == 1
    assert region_index_at([], 1.0) is None


def test_duration():
    assert Region(1, 2.5, frozenset()).duration_s == pytest.approx(1.5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd api && uv run pytest tests/unit/test_overlap.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'saydo.pipeline.overlap'`.

- [ ] **Step 3: Implement**

`api/src/saydo/pipeline/overlap.py`:
```python
"""Find time spans where two or more distinct speakers are active (spec section 6, step 4)."""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from saydo.pipeline.types import Turn


@dataclass(frozen=True)
class Region:
    start_s: float
    end_s: float
    speaker_keys: frozenset[str]

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def find_overlap_regions(
    turns: Sequence[Turn], *, min_duration_s: float = 0.2, merge_gap_s: float = 0.1
) -> list[Region]:
    events: list[tuple[float, int, str]] = []
    for turn in turns:
        if turn.end_s > turn.start_s:
            events.append((turn.start_s, 1, turn.speaker_key))
            events.append((turn.end_s, -1, turn.speaker_key))
    # At equal times, process ends (-1) before starts (+1): touching turns do not overlap.
    events.sort(key=lambda e: (e[0], e[1]))

    active: Counter[str] = Counter()
    raw: list[Region] = []
    previous: float | None = None
    for time, delta, key in events:
        if previous is not None and time > previous:
            speakers = frozenset(k for k, count in active.items() if count > 0)
            if len(speakers) >= 2:
                raw.append(Region(previous, time, speakers))
        active[key] += delta
        previous = time

    merged: list[Region] = []
    for region in raw:
        if merged and region.start_s - merged[-1].end_s < merge_gap_s:
            last = merged[-1]
            merged[-1] = Region(
                last.start_s, max(last.end_s, region.end_s), last.speaker_keys | region.speaker_keys
            )
        else:
            merged.append(region)
    return [r for r in merged if r.duration_s >= min_duration_s - 1e-9]


def region_index_at(regions: Sequence[Region], t: float) -> int | None:
    index = bisect_right([r.start_s for r in regions], t) - 1
    if index >= 0 and t < regions[index].end_s:
        return index
    return None
```

Note: `region_index_at` rebuilds the start list on every call. `build_segments` (Task 7) calls it once per word, so Task 7 passes a precomputed list through `RegionLookup` instead. Keep this simple function for one-off callers and tests.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd api && uv run pytest tests/unit/test_overlap.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add api/
git commit -m "feat(pipeline): overlap region detection"
```

---

### Task 7: Word-to-speaker merge and overlap placeholders (M3)

**Files:**
- Create: `api/src/saydo/pipeline/merge.py`
- Test: `api/tests/unit/test_merge.py`

**Interfaces:**
- Consumes: `Word`, `Turn` (Task 5); `Region` (Task 6).
- Produces:
  - `DraftSegment(speaker_key, start_s, end_s, words: list[Word], origin: Origin = "mixture", region_index: int | None = None, has_orphan: bool = False)`, with properties `.text`, `.duration_s` and `.is_overlap`.
  - `Origin = Literal["mixture", "separated", "placeholder"]`.
  - `build_segments(words, turns, regions, *, max_gap_s=1.0, orphan_window_s=0.5) -> tuple[list[DraftSegment], int]`, which returns the drafts sorted by `(start_s, speaker_key)` and the number of dropped words.

- [ ] **Step 1: Write the failing tests**

`api/tests/unit/test_merge.py`:
```python
from saydo.pipeline.merge import build_segments
from saydo.pipeline.overlap import find_overlap_regions
from saydo.pipeline.types import Turn, Word


def words_every(start, end, step=0.5, length=0.4):
    out, t, n = [], start, 0
    while t + length <= end + 1e-9:
        n += 1
        out.append(Word(round(t, 3), round(t + length, 3), f" w{n}"))
        t += step
    return out


def run(words, turns):
    turns = [Turn(*t) for t in turns]
    return build_segments(words, turns, find_overlap_regions(turns))


def test_single_speaker_is_one_segment_with_joined_text():
    drafts, dropped = run([Word(0, 0.4, " Hello"), Word(0.5, 0.9, " world.")], [("A", 0, 2)])
    assert dropped == 0
    assert len(drafts) == 1
    assert (drafts[0].speaker_key, drafts[0].text, drafts[0].origin) == ("A", "Hello world.", "mixture")
    assert (drafts[0].start_s, drafts[0].end_s) == (0, 0.9)
    assert not drafts[0].is_overlap


def test_speaker_change_splits():
    drafts, _ = run(words_every(0, 4), [("A", 0, 2), ("B", 2, 4)])
    assert [d.speaker_key for d in drafts] == ["A", "B"]


def test_long_gap_splits_same_speaker():
    drafts, _ = run([Word(0, 0.4, " a"), Word(1.5, 1.9, " b")], [("A", 0, 2)])
    assert len(drafts) == 2


def test_overlap_boundary_splits_and_silent_speaker_gets_placeholder():
    drafts, _ = run(words_every(3, 7), [("A", 0, 10), ("B", 4, 6)])
    spans = [(d.speaker_key, d.start_s, d.end_s, d.origin, d.region_index) for d in drafts]
    assert spans == [
        ("A", 3.0, 3.9, "mixture", None),
        ("A", 4.0, 5.9, "mixture", 0),
        ("B", 4.0, 6.0, "placeholder", 0),
        ("A", 6.0, 6.9, "mixture", None),
    ]
    placeholder = drafts[2]
    assert placeholder.words == [] and placeholder.text == "" and placeholder.is_overlap


def test_no_placeholder_when_both_speakers_have_words_in_region():
    # Region is 4-5. " a" (3.8-4.3) is covered more by A's turn; " b" (4.4-4.9) ties on
    # coverage and goes to B's longer turn. Both speakers have words in the region.
    words = [Word(3.8, 4.3, " a"), Word(4.4, 4.9, " b")]
    drafts, _ = run(words, [("A", 3, 5), ("B", 4, 10)])
    assert [(d.speaker_key, d.origin, d.region_index) for d in drafts] == [
        ("A", "mixture", 0), ("B", "mixture", 0),
    ]


def test_both_speakers_placeholders_when_region_has_no_words():
    drafts, _ = run([Word(0, 0.4, " a")], [("A", 0, 5), ("B", 3, 6)])
    placeholders = [d for d in drafts if d.origin == "placeholder"]
    assert {d.speaker_key for d in placeholders} == {"A", "B"}


def test_tie_breaks_by_longer_turn_then_key():
    # Word 4.0-4.6 (midpoint 4.3) overlaps both turns by 0.6 s; A's turn is longer.
    drafts, _ = run([Word(4.0, 4.6, " x")], [("A", 0, 6), ("B", 4, 8)])
    assert drafts[0].speaker_key == "A"


def test_orphan_word_is_assigned_and_flagged_far_word_dropped():
    drafts, dropped = run([Word(2.1, 2.3, " near"), Word(5.0, 5.4, " far")], [("A", 0, 2)])
    assert dropped == 1
    assert len(drafts) == 1
    assert drafts[0].has_orphan
    assert drafts[0].text == "near"


def test_output_sorted_by_start_then_speaker():
    drafts, _ = run(words_every(0, 10), [("A", 0, 4), ("B", 3, 7), ("A", 6, 10)])
    keys = [(d.start_s, d.speaker_key) for d in drafts]
    assert keys == sorted(keys)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd api && uv run pytest tests/unit/test_merge.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'saydo.pipeline.merge'`.

- [ ] **Step 3: Implement**

`api/src/saydo/pipeline/merge.py`:
```python
"""Assign ASR words to diarized speakers and build segments (spec section 6, step 4).

Rules:
- A word goes to the speaker active at its midpoint. With several active speakers it goes to
  the one whose turn covers more of the word, then the longer turn, then the lower key.
- With no active speaker, the nearest turn within `orphan_window_s` takes it (an "orphan");
  otherwise the word is dropped and counted.
- A new segment starts on a speaker change, a gap over `max_gap_s`, or an overlap-region
  boundary, so every segment is fully inside one region or fully outside all regions.
- In each overlap region, each active speaker without words gets a placeholder segment.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from saydo.pipeline.overlap import Region
from saydo.pipeline.types import Turn, Word

Origin = Literal["mixture", "separated", "placeholder"]


@dataclass
class DraftSegment:
    speaker_key: str
    start_s: float
    end_s: float
    words: list[Word] = field(default_factory=list)
    origin: Origin = "mixture"
    region_index: int | None = None
    has_orphan: bool = False

    @property
    def text(self) -> str:
        return "".join(w.text for w in self.words).strip()

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def is_overlap(self) -> bool:
        return self.region_index is not None


class _TurnIndex:
    """Buckets turns per second so each word only looks at nearby turns."""

    def __init__(self, turns: Sequence[Turn], pad_s: float) -> None:
        self._buckets: dict[int, list[Turn]] = defaultdict(list)
        for turn in turns:
            first = math.floor(turn.start_s - pad_s)
            last = math.floor(turn.end_s + pad_s)
            for bucket in range(first, last + 1):
                self._buckets[bucket].append(turn)

    def near(self, t: float) -> list[Turn]:
        return self._buckets.get(math.floor(t), [])


class _RegionLookup:
    def __init__(self, regions: Sequence[Region]) -> None:
        self._regions = regions
        self._starts = [r.start_s for r in regions]

    def index_at(self, t: float) -> int | None:
        i = bisect_right(self._starts, t) - 1
        return i if i >= 0 and t < self._regions[i].end_s else None


def _assign(word: Word, candidates: list[Turn], orphan_window_s: float) -> tuple[str, bool] | None:
    mid = (word.start_s + word.end_s) / 2
    active = [t for t in candidates if t.start_s <= mid < t.end_s]
    if active:

        def rank(turn: Turn) -> tuple[float, float, str]:
            covered = max(0.0, min(turn.end_s, word.end_s) - max(turn.start_s, word.start_s))
            return (-covered, -(turn.end_s - turn.start_s), turn.speaker_key)

        return min(active, key=rank).speaker_key, False
    best: Turn | None = None
    best_distance = math.inf
    for turn in candidates:
        distance = turn.start_s - mid if mid < turn.start_s else mid - turn.end_s
        if distance < best_distance:
            best, best_distance = turn, distance
    if best is not None and best_distance <= orphan_window_s:
        return best.speaker_key, True
    return None


def _placeholders(
    drafts: list[DraftSegment], turns: Sequence[Turn], regions: Sequence[Region]
) -> list[DraftSegment]:
    spoken = {(d.region_index, d.speaker_key) for d in drafts if d.region_index is not None}
    out: list[DraftSegment] = []
    for index, region in enumerate(regions):
        for key in sorted(region.speaker_keys):
            if (index, key) in spoken:
                continue
            spans = [
                (max(t.start_s, region.start_s), min(t.end_s, region.end_s))
                for t in turns
                if t.speaker_key == key and t.start_s < region.end_s and t.end_s > region.start_s
            ]
            if spans:
                out.append(
                    DraftSegment(
                        speaker_key=key,
                        start_s=min(s for s, _ in spans),
                        end_s=max(e for _, e in spans),
                        origin="placeholder",
                        region_index=index,
                    )
                )
    return out


def build_segments(
    words: Sequence[Word],
    turns: Sequence[Turn],
    regions: Sequence[Region],
    *,
    max_gap_s: float = 1.0,
    orphan_window_s: float = 0.5,
) -> tuple[list[DraftSegment], int]:
    turn_index = _TurnIndex(turns, pad_s=orphan_window_s)
    region_lookup = _RegionLookup(regions)
    drafts: list[DraftSegment] = []
    current: DraftSegment | None = None
    dropped = 0

    for word in sorted(words, key=lambda w: (w.start_s, w.end_s)):
        mid = (word.start_s + word.end_s) / 2
        assigned = _assign(word, turn_index.near(mid), orphan_window_s)
        if assigned is None:
            dropped += 1
            continue
        speaker_key, orphan = assigned
        region = region_lookup.index_at(mid)
        if (
            current is None
            or current.speaker_key != speaker_key
            or current.region_index != region
            or word.start_s - current.end_s > max_gap_s
        ):
            current = DraftSegment(speaker_key, word.start_s, word.end_s, region_index=region)
            drafts.append(current)
        current.words.append(word)
        current.end_s = max(current.end_s, word.end_s)
        current.has_orphan = current.has_orphan or orphan

    drafts.extend(_placeholders(drafts, turns, regions))
    drafts.sort(key=lambda d: (d.start_s, d.speaker_key))
    return drafts, dropped
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd api && uv run pytest tests/unit/test_merge.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add api/
git commit -m "feat(pipeline): word-to-speaker merge with overlap placeholders"
```

---

### Task 8: Confidence scoring and review flags (M3)

**Files:**
- Create: `api/src/saydo/pipeline/confidence.py`
- Test: `api/tests/unit/test_confidence.py`

**Interfaces:**
- Consumes: `ReviewSettings` (Task 1); `Word` (Task 5).
- Produces:
  - `asr_score(avg_logprob, no_speech_prob) -> float`
  - `margin_score(cos_assigned: float | None, cos_next_best: float | None, *, temperature=0.05) -> float`
  - `length_score(duration_s) -> float`
  - `segment_asr(words: Sequence[Word]) -> float`
  - `ConfidenceResult(confidence: float, parts: dict[str, float], needs_review: bool)`
  - `score_segment(*, origin, asr, margin, duration_s, is_overlap, has_orphan, review) -> ConfidenceResult`
  - `MIN_EMBED_DURATION_S = 0.5`

- [ ] **Step 1: Write the failing tests**

`api/tests/unit/test_confidence.py`:
```python
import math

import pytest

from saydo.config import ReviewSettings
from saydo.pipeline.confidence import (
    asr_score,
    length_score,
    margin_score,
    score_segment,
    segment_asr,
)
from saydo.pipeline.types import Word

REVIEW = ReviewSettings()
GOOD_ASR = asr_score(-0.2, 0.01)


def score(**overrides):
    args = dict(origin="mixture", asr=GOOD_ASR, margin=1.0, duration_s=3.0,
                is_overlap=False, has_orphan=False, review=REVIEW)
    args.update(overrides)
    return score_segment(**args)


def test_asr_score():
    assert asr_score(-0.2, 0.01) == pytest.approx(math.exp(-0.2) * 0.99)
    assert asr_score(0.5, 0.0) == 1.0  # clamped
    assert asr_score(-1.0, 1.0) == 0.0


def test_margin_score_cases():
    assert margin_score(None, None) == 0.5  # too short to embed / no centroid
    assert margin_score(0.8, None) == 1.0  # only one speaker in the recording
    assert margin_score(0.7, 0.7) == pytest.approx(0.5)
    assert margin_score(0.9, 0.6) > 0.99
    assert margin_score(0.6, 0.9) < 0.01


def test_length_score():
    assert length_score(0.75) == pytest.approx(0.5)
    assert length_score(10) == 1.0
    assert length_score(-1) == 0.0


def test_segment_asr_is_duration_weighted():
    words = [Word(0, 1, " a", asr_score=1.0), Word(1, 4, " b", asr_score=0.0)]
    assert segment_asr(words) == pytest.approx(0.25)
    assert segment_asr([]) == 0.0


def test_clean_segment_is_not_flagged():
    result = score()
    assert result.confidence == pytest.approx(0.4 * GOOD_ASR + 0.4 + 0.2, abs=1e-4)
    assert not result.needs_review
    assert set(result.parts) == {"asr", "margin", "length", "penalty"}


def test_overlap_penalty_flags_even_a_clean_segment():
    result = score(is_overlap=True)
    assert result.confidence == pytest.approx((0.4 * GOOD_ASR + 0.6) * 0.6, abs=1e-4)
    assert result.needs_review
    assert result.parts["penalty"] == pytest.approx(0.6)


def test_penalties_multiply():
    result = score(is_overlap=True, origin="separated", has_orphan=True)
    assert result.parts["penalty"] == pytest.approx(0.6 * 0.8 * 0.7)


def test_placeholder_is_zero_and_flagged():
    result = score(origin="placeholder")
    assert result.confidence == 0.0
    assert result.needs_review


def test_threshold_comes_from_settings():
    lenient = ReviewSettings(threshold=0.5)
    assert not score(is_overlap=True, review=lenient).needs_review
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd api && uv run pytest tests/unit/test_confidence.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'saydo.pipeline.confidence'`.

- [ ] **Step 3: Implement**

`api/src/saydo/pipeline/confidence.py`:
```python
"""Segment confidence (spec section 7). A heuristic ranking for review, not a probability."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from saydo.config import ReviewSettings
from saydo.pipeline.types import Word

MIN_EMBED_DURATION_S = 0.5
LENGTH_SATURATION_S = 1.5


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))


def asr_score(avg_logprob: float, no_speech_prob: float) -> float:
    return _clamp(math.exp(avg_logprob)) * _clamp(1.0 - no_speech_prob)


def margin_score(
    cos_assigned: float | None, cos_next_best: float | None, *, temperature: float = 0.05
) -> float:
    if cos_assigned is None:
        return 0.5
    if cos_next_best is None:
        return 1.0
    return 1.0 / (1.0 + math.exp(-(cos_assigned - cos_next_best) / temperature))


def length_score(duration_s: float) -> float:
    return _clamp(duration_s / LENGTH_SATURATION_S)


def segment_asr(words: Sequence[Word]) -> float:
    total = sum(max(w.end_s - w.start_s, 1e-3) for w in words)
    if total == 0:
        return 0.0
    return sum(w.asr_score * max(w.end_s - w.start_s, 1e-3) for w in words) / total


@dataclass(frozen=True)
class ConfidenceResult:
    confidence: float
    parts: dict[str, float]
    needs_review: bool


def score_segment(
    *,
    origin: str,
    asr: float,
    margin: float,
    duration_s: float,
    is_overlap: bool,
    has_orphan: bool,
    review: ReviewSettings,
) -> ConfidenceResult:
    if origin == "placeholder":
        return ConfidenceResult(0.0, {"asr": 0.0, "margin": 0.0, "length": 0.0, "penalty": 0.0}, True)
    length = length_score(duration_s)
    w = review.weights
    base = w.asr * asr + w.margin * margin + w.length * length
    penalty = 1.0
    if is_overlap:
        penalty *= review.overlap_penalty
    if origin == "separated":
        penalty *= review.separated_penalty
    if has_orphan:
        penalty *= review.orphan_penalty
    confidence = round(_clamp(base * penalty), 4)
    parts = {
        "asr": round(asr, 4),
        "margin": round(margin, 4),
        "length": round(length, 4),
        "penalty": round(penalty, 4),
    }
    return ConfidenceResult(confidence, parts, confidence < review.threshold)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd api && uv run pytest tests/unit/test_confidence.py -q`
Expected: all pass. With the default settings every overlapping segment is flagged: the best possible overlap score is 0.92 × 0.6 = 0.55, which is below 0.6. This is intended for v1, and M3's calibration step may revisit it.

- [ ] **Step 5: Commit**

```bash
git add api/
git commit -m "feat(pipeline): segment confidence scoring and review flags"
```

---

### Task 9: Pipeline runner: analyze, persist, retention (M2/M3)

**Files:**
- Create: `api/src/saydo/pipeline/speakers.py`, `api/src/saydo/pipeline/run.py`
- Modify: `api/tests/conftest.py` (append the `files` and `make_recording` fixtures)
- Test: `api/tests/unit/test_speakers.py`, `api/tests/unit/test_run.py`

**Interfaces:**
- Consumes: everything from Tasks 2–8.
- Produces:
  - `saydo.pipeline.speakers`:
    - `cosine(a, b) -> float`
    - `segment_embeddings(drafts, audio: Audio, embedder) -> dict[int, np.ndarray]`, keyed by draft index
    - `speaker_centroids(drafts, embeddings) -> dict[str, np.ndarray]`
    - `margin_inputs(index, draft, embeddings, centroids) -> tuple[float | None, float | None]`
    - `number_speakers(drafts) -> dict[str, str]`, mapping speaker key to `"Speaker N"`
  - `saydo.pipeline.run`:
    - `Plugins(transcriber, diarizer, embedder, device: str)`
    - `build_plugins(settings) -> Plugins`
    - `filter_hallucinations(segments) -> list[AsrSegment]`
    - `words_with_scores(segments) -> list[Word]`
    - `Analysis(turns, regions, drafts, scores, centroids, dropped_words)`
    - `analyze(audio, plugins, review, *, min_speakers=None, max_speakers=None, language=None, on_stage=lambda s: None) -> Analysis`
    - `PipelineRunner(settings, session_factory, files, plugins_factory: Callable[[], Plugins])` with `.process(recording_id) -> None`
  - Stage names written to `recording.stage`: `normalize`, `diarize`, `transcribe`, `merge`, `score`, `persist`.
  - Test fixtures: `files` (`RecordingFiles`) and `make_recording(seconds=10.0, silent=False, garbage=False) -> str`.

- [ ] **Step 1: Write the failing tests**

Append to `api/tests/conftest.py`:
```python
from audio_helpers import write_tone  # noqa: E402

from saydo.db.models import Recording  # noqa: E402
from saydo.ids import uuid7  # noqa: E402
from saydo.storage.files import RecordingFiles  # noqa: E402


@pytest.fixture
def files(settings) -> RecordingFiles:
    return RecordingFiles(settings.data_dir)


@pytest.fixture
def make_recording(session_factory, files):
    def _make(seconds: float = 10.0, *, silent: bool = False, garbage: bool = False) -> str:
        rid = uuid7()
        original = files.original_path(rid, "wav")
        if garbage:
            original.parent.mkdir(parents=True, exist_ok=True)
            original.write_bytes(b"not audio" * 200)
        else:
            write_tone(original, seconds, silent=silent)
        with session_factory() as s:
            s.add(Recording(id=rid, source="upload", status="queued"))
            s.commit()
        return rid

    return _make
```

`api/tests/unit/test_speakers.py`:
```python
import numpy as np
import pytest

from saydo.pipeline.merge import DraftSegment
from saydo.pipeline.speakers import (
    cosine,
    margin_inputs,
    number_speakers,
    segment_embeddings,
    speaker_centroids,
)
from saydo.pipeline.types import Audio


class KeyedEmbedder:
    """Returns +x for audio before 5 s and +y after, so two 'voices' separate cleanly."""

    name = "keyed"

    def embed(self, samples, sample_rate):
        return np.array([1.0, 0.0]) if samples[0] < 0.5 else np.array([0.0, 1.0])


def ramp_audio(seconds=10.0):
    n = int(seconds * 16000)
    return Audio(samples=(np.arange(n) / n).astype(np.float32)[None, :], sample_rate=16000)


def test_cosine():
    assert cosine(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == pytest.approx(1.0)
    assert cosine(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)
    assert cosine(np.zeros(2), np.array([1.0, 0.0])) == 0.0


def test_embeddings_skip_placeholders_and_short_segments():
    drafts = [
        DraftSegment("A", 0, 2),
        DraftSegment("A", 2, 2.3),
        DraftSegment("B", 6, 8, origin="placeholder", region_index=0),
    ]
    embeddings = segment_embeddings(drafts, ramp_audio(), KeyedEmbedder())
    assert set(embeddings) == {0}


def test_centroids_prefer_long_non_overlap_segments_and_margins():
    drafts = [
        DraftSegment("A", 0, 2),
        DraftSegment("A", 4.0, 4.8, region_index=0),
        DraftSegment("B", 6, 8),
    ]
    embeddings = segment_embeddings(drafts, ramp_audio(), KeyedEmbedder())
    centroids = speaker_centroids(drafts, embeddings)
    assert set(centroids) == {"A", "B"}
    assigned, next_best = margin_inputs(0, drafts[0], embeddings, centroids)
    assert assigned == pytest.approx(1.0) and next_best == pytest.approx(0.0)


def test_single_speaker_has_no_next_best():
    drafts = [DraftSegment("A", 0, 2)]
    embeddings = segment_embeddings(drafts, ramp_audio(), KeyedEmbedder())
    assert margin_inputs(0, drafts[0], embeddings, speaker_centroids(drafts, embeddings))[1] is None


def test_number_speakers_by_first_appearance():
    drafts = [DraftSegment("SPK3", 0, 1), DraftSegment("SPK0", 2, 3), DraftSegment("SPK3", 4, 5)]
    assert number_speakers(drafts) == {"SPK3": "Speaker 1", "SPK0": "Speaker 2"}
```

`api/tests/unit/test_run.py`:
```python
import pytest
from sqlalchemy import select

from saydo.db.models import OverlapRegion, Recording, Segment, Speaker
from saydo.pipeline.errors import AudioDecodeError, RecordingGone
from saydo.pipeline.run import PipelineRunner, Plugins, build_plugins, filter_hallucinations
from saydo.pipeline.types import AsrSegment
from saydo.plugins.fake import FakeDiarizer
from saydo.services import delete_recording


def runner(settings, session_factory, files, plugins=None):
    return PipelineRunner(settings, session_factory, files, lambda: plugins or build_plugins(settings))


def rows(session_factory, model, rid):
    with session_factory() as s:
        return s.scalars(select(model).where(model.recording_id == rid)).all()


def test_process_produces_ready_transcript_with_overlaps(settings, session_factory, files, make_recording):
    rid = make_recording(10.0)
    runner(settings, session_factory, files).process(rid)

    with session_factory() as s:
        rec = s.get(Recording, rid)
        assert (rec.status, rec.stage, rec.error) == ("ready", None, None)
        assert rec.transcript_version == 1
        assert rec.duration_s == pytest.approx(10.0)
        assert rec.channels == 1
        assert rec.pipeline_config["transcriber"]["name"] == "fake"
        assert rec.pipeline_config["device"] == "cpu"

    speakers = sorted(rows(session_factory, Speaker, rid), key=lambda sp: sp.label)
    assert [sp.label for sp in speakers] == ["Speaker 1", "Speaker 2"]
    regions = rows(session_factory, OverlapRegion, rid)
    assert len(regions) == 3  # fake turns overlap at 3-4, 6-7, 9-10
    segments = rows(session_factory, Segment, rid)
    for region in regions:
        members = [seg for seg in segments if seg.overlap_region_id == region.id]
        assert len(members) == 2
        assert all(seg.is_overlap and seg.needs_review for seg in members)
        (placeholder,) = [seg for seg in members if seg.origin == "placeholder"]
        assert placeholder.confidence == 0.0 and placeholder.text == ""
    first = min(segments, key=lambda seg: seg.start_s)
    by_id = {sp.id: sp.label for sp in speakers}
    assert by_id[first.speaker_id] == "Speaker 1"
    assert all(seg.original_speaker_id == seg.speaker_id for seg in segments)
    assert all(0.0 <= seg.confidence <= 1.0 for seg in segments)


def test_processing_twice_does_not_duplicate(settings, session_factory, files, make_recording):
    rid = make_recording(10.0)
    r = runner(settings, session_factory, files)
    r.process(rid)
    count = len(rows(session_factory, Segment, rid))
    r.process(rid)
    assert len(rows(session_factory, Segment, rid)) == count
    assert len(rows(session_factory, Speaker, rid)) == 2


def test_silent_audio_is_ready_with_no_segments(settings, session_factory, files, make_recording):
    rid = make_recording(5.0, silent=True)
    runner(settings, session_factory, files).process(rid)
    with session_factory() as s:
        assert s.get(Recording, rid).status == "ready"
    assert rows(session_factory, Segment, rid) == []
    assert rows(session_factory, Speaker, rid) == []


def test_garbage_audio_raises_decode_error(settings, session_factory, files, make_recording):
    rid = make_recording(garbage=True)
    with pytest.raises(AudioDecodeError):
        runner(settings, session_factory, files).process(rid)


def test_process_raises_recording_gone_when_deleted_mid_run(settings, session_factory, files, make_recording):
    rid = make_recording(10.0)

    class DeletingDiarizer(FakeDiarizer):
        def diarize(self, audio, **kwargs):
            with session_factory() as s:
                delete_recording(s, files, rid)
            return super().diarize(audio, **kwargs)

    base = build_plugins(settings)
    plugins = Plugins(base.transcriber, DeletingDiarizer(), base.embedder, "cpu")
    with pytest.raises(RecordingGone):
        runner(settings, session_factory, files, plugins).process(rid)
    with session_factory() as s:
        assert s.get(Recording, rid) is None
    assert not files.dir(rid).exists()


def test_retention_deletes_raw_audio_after_processing(settings, session_factory, files, make_recording):
    settings.retention.delete_raw_audio_after_processing = True
    rid = make_recording(4.0)
    runner(settings, session_factory, files).process(rid)
    assert files.find_original(rid) is None
    assert files.normalized_path(rid).exists()
    with session_factory() as s:
        assert s.get(Recording, rid).raw_audio_deleted_at is not None


def test_retention_can_drop_playback_audio(settings, session_factory, files, make_recording):
    settings.retention.keep_playback_audio = False
    rid = make_recording(4.0)
    runner(settings, session_factory, files).process(rid)
    assert not files.normalized_path(rid).exists()


def test_filter_hallucinations():
    keep = AsrSegment(0, 1, "a", (), -0.5, 0.9)  # high no-speech but confident text
    drop = AsrSegment(1, 2, "b", (), -1.5, 0.7)
    also_keep = AsrSegment(2, 3, "c", (), -1.5, 0.2)
    assert filter_hallucinations([keep, drop, also_keep]) == [keep, also_keep]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd api && uv run pytest tests/unit/test_speakers.py tests/unit/test_run.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'saydo.pipeline.speakers'`.

- [ ] **Step 3: Implement speaker helpers**

`api/src/saydo/pipeline/speakers.py`:
```python
"""Speaker embeddings, centroids, confidence margins and "Speaker N" numbering.

Centroids are computed by SayDo (not taken from the diarizer) so margins live in one
embedding space whatever diarizer is configured. Only non-overlapping speech is used for
centroids, preferring segments of at least 1.5 s.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from saydo.pipeline.confidence import LENGTH_SATURATION_S, MIN_EMBED_DURATION_S
from saydo.pipeline.interfaces import Embedder
from saydo.pipeline.merge import DraftSegment
from saydo.pipeline.types import Audio


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 0 else 0.0


def segment_embeddings(
    drafts: Sequence[DraftSegment], audio: Audio, embedder: Embedder
) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for index, draft in enumerate(drafts):
        if draft.origin == "placeholder" or draft.duration_s < MIN_EMBED_DURATION_S:
            continue
        samples = audio.slice(draft.start_s, draft.end_s)
        out[index] = np.asarray(embedder.embed(samples, audio.sample_rate), dtype=np.float32).reshape(-1)
    return out


def speaker_centroids(
    drafts: Sequence[DraftSegment], embeddings: dict[int, np.ndarray]
) -> dict[str, np.ndarray]:
    centroids: dict[str, np.ndarray] = {}
    for key in sorted({d.speaker_key for d in drafts}):
        candidates = [
            i for i, d in enumerate(drafts)
            if d.speaker_key == key and i in embeddings and not d.is_overlap
        ]
        long = [i for i in candidates if drafts[i].duration_s >= LENGTH_SATURATION_S]
        chosen = long or candidates
        if chosen:
            centroids[key] = _unit(np.mean([_unit(embeddings[i]) for i in chosen], axis=0))
    return centroids


def margin_inputs(
    index: int,
    draft: DraftSegment,
    embeddings: dict[int, np.ndarray],
    centroids: dict[str, np.ndarray],
) -> tuple[float | None, float | None]:
    embedding = embeddings.get(index)
    if embedding is None or draft.speaker_key not in centroids:
        return None, None
    assigned = cosine(embedding, centroids[draft.speaker_key])
    others = [cosine(embedding, c) for key, c in centroids.items() if key != draft.speaker_key]
    return assigned, (max(others) if others else None)


def number_speakers(drafts: Sequence[DraftSegment]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for draft in sorted(drafts, key=lambda d: (d.start_s, d.speaker_key)):
        if draft.speaker_key not in labels:
            labels[draft.speaker_key] = f"Speaker {len(labels) + 1}"
    return labels
```

- [ ] **Step 4: Implement the runner**

`api/src/saydo/pipeline/run.py`:
```python
"""Run the transcription pipeline for one recording (spec section 6).

`analyze` is pure (audio in, analysis out) and is reused by the evaluation harness.
`PipelineRunner.process` adds normalization, stage tracking, persistence and retention.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from sqlalchemy import delete
from sqlalchemy.orm import Session, sessionmaker

from saydo import __version__
from saydo.config import ReviewSettings, Settings
from saydo.db.models import OverlapRegion, Recording, Segment, Speaker, utcnow
from saydo.ids import uuid7
from saydo.pipeline.confidence import ConfidenceResult, asr_score, margin_score, score_segment, segment_asr
from saydo.pipeline.errors import PipelineError, RecordingGone
from saydo.pipeline.interfaces import Diarizer, Embedder, Transcriber
from saydo.pipeline.merge import DraftSegment, build_segments
from saydo.pipeline.normalize import normalize
from saydo.pipeline.overlap import Region, find_overlap_regions
from saydo.pipeline.registry import build_plugin, resolve_device
from saydo.pipeline.speakers import margin_inputs, number_speakers, segment_embeddings, speaker_centroids
from saydo.pipeline.types import AsrSegment, Audio, Turn, Word, load_audio
from saydo.storage.files import RecordingFiles

log = logging.getLogger("saydo.pipeline")

SPEAKER_COLORS = 12


@dataclass
class Plugins:
    transcriber: Transcriber
    diarizer: Diarizer
    embedder: Embedder
    device: str


def build_plugins(settings: Settings) -> Plugins:
    device = resolve_device(settings.pipeline.device)
    p = settings.pipeline
    return Plugins(
        transcriber=build_plugin("transcriber", p.transcriber, device),
        diarizer=build_plugin("diarizer", p.diarizer, device),
        embedder=build_plugin("embedder", p.embedder, device),
        device=device,
    )


def filter_hallucinations(segments: list[AsrSegment]) -> list[AsrSegment]:
    return [s for s in segments if not (s.no_speech_prob > 0.6 and s.avg_logprob < -1.0)]


def words_with_scores(segments: list[AsrSegment]) -> list[Word]:
    return [
        replace(word, asr_score=asr_score(segment.avg_logprob, segment.no_speech_prob))
        for segment in segments
        for word in segment.words
    ]


@dataclass
class Analysis:
    turns: tuple[Turn, ...]
    regions: list[Region]
    drafts: list[DraftSegment]
    scores: list[ConfidenceResult]
    centroids: dict[str, np.ndarray]
    dropped_words: int


def analyze(
    audio: Audio,
    plugins: Plugins,
    review: ReviewSettings,
    *,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    language: str | None = None,
    on_stage: Callable[[str], None] = lambda _stage: None,
) -> Analysis:
    on_stage("diarize")
    diarization = plugins.diarizer.diarize(audio, min_speakers=min_speakers, max_speakers=max_speakers)
    on_stage("transcribe")
    asr = filter_hallucinations(plugins.transcriber.transcribe(audio, language=language))
    on_stage("merge")
    regions = find_overlap_regions(diarization.turns)
    drafts, dropped = build_segments(words_with_scores(asr), diarization.turns, regions)
    on_stage("score")
    embeddings = segment_embeddings(drafts, audio, plugins.embedder)
    centroids = speaker_centroids(drafts, embeddings)
    scores = []
    for index, draft in enumerate(drafts):
        cos_assigned, cos_next = margin_inputs(index, draft, embeddings, centroids)
        scores.append(
            score_segment(
                origin=draft.origin,
                asr=segment_asr(draft.words),
                margin=margin_score(cos_assigned, cos_next),
                duration_s=draft.duration_s,
                is_overlap=draft.is_overlap,
                has_orphan=draft.has_orphan,
                review=review,
            )
        )
    return Analysis(diarization.turns, regions, drafts, scores, centroids, dropped)


class PipelineRunner:
    def __init__(
        self,
        settings: Settings,
        session_factory: sessionmaker[Session],
        files: RecordingFiles,
        plugins_factory: Callable[[], Plugins],
    ) -> None:
        self._settings = settings
        self._sessions = session_factory
        self._files = files
        self._plugins_factory = plugins_factory
        self._plugins: Plugins | None = None

    @property
    def plugins(self) -> Plugins:
        if self._plugins is None:  # models load once, on the first job
            self._plugins = self._plugins_factory()
        return self._plugins

    def process(self, recording_id: str) -> None:
        recording = self._set_stage(recording_id, "normalize")
        source = self._files.find_original(recording_id)
        if source is None:
            raise PipelineError("The original audio file is missing")
        normalized = normalize(
            source,
            self._files.normalized_path(recording_id),
            keep_channels=self.plugins.diarizer.wants_multichannel,
        )
        self._update(recording_id, duration_s=normalized.duration_s, channels=normalized.channels)
        audio = load_audio(normalized.path)

        analysis = analyze(
            audio,
            self.plugins,
            self._settings.review,
            min_speakers=recording.min_speakers,
            max_speakers=recording.max_speakers,
            language=recording.language,
            on_stage=lambda stage: self._set_stage(recording_id, stage),
        )
        self._set_stage(recording_id, "persist")
        self._persist(recording_id, analysis)
        self._apply_retention(recording_id)
        log.info(
            "recording %s ready: %d segments, %d overlap regions, %d dropped words",
            recording_id, len(analysis.drafts), len(analysis.regions), analysis.dropped_words,
        )

    def _set_stage(self, recording_id: str, stage: str) -> Recording:
        with self._sessions() as session:
            recording = session.get(Recording, recording_id)
            if recording is None:
                raise RecordingGone(recording_id)
            recording.status = "processing"
            recording.stage = stage
            recording.error = None
            session.commit()
            return recording

    def _update(self, recording_id: str, **values: Any) -> None:
        with self._sessions() as session:
            recording = session.get(Recording, recording_id)
            if recording is None:
                raise RecordingGone(recording_id)
            for key, value in values.items():
                setattr(recording, key, value)
            session.commit()

    def _pipeline_config(self, analysis: Analysis) -> dict[str, Any]:
        p = self._settings.pipeline
        return {
            "saydo_version": __version__,
            "device": self.plugins.device,
            "transcriber": p.transcriber.model_dump(),
            "diarizer": p.diarizer.model_dump(),
            "embedder": p.embedder.model_dump(),
            "review": self._settings.review.model_dump(),
            "dropped_words": analysis.dropped_words,
        }

    def _persist(self, recording_id: str, analysis: Analysis) -> None:
        with self._sessions() as session, session.begin():
            recording = session.get(Recording, recording_id)
            if recording is None:
                raise RecordingGone(recording_id)
            # Re-processing (retry) replaces earlier output.
            session.execute(delete(Segment).where(Segment.recording_id == recording_id))
            session.execute(delete(OverlapRegion).where(OverlapRegion.recording_id == recording_id))
            session.execute(delete(Speaker).where(Speaker.recording_id == recording_id))

            speaker_ids: dict[str, str] = {}
            for key, label in number_speakers(analysis.drafts).items():
                centroid = analysis.centroids.get(key)
                speaker = Speaker(
                    id=uuid7(),
                    recording_id=recording_id,
                    label=label,
                    color_index=(int(label.rsplit(" ", 1)[-1]) - 1) % SPEAKER_COLORS,
                    embedding=centroid.astype(np.float32).tobytes() if centroid is not None else None,
                )
                session.add(speaker)
                speaker_ids[key] = speaker.id

            region_ids = [uuid7() for _ in analysis.regions]
            for region_id, region in zip(region_ids, analysis.regions, strict=True):
                session.add(
                    OverlapRegion(
                        id=region_id, recording_id=recording_id,
                        start_s=round(region.start_s, 3), end_s=round(region.end_s, 3),
                        n_speakers=len(region.speaker_keys),
                    )
                )
            session.flush()

            for draft, result in zip(analysis.drafts, analysis.scores, strict=True):
                speaker_id = speaker_ids[draft.speaker_key]
                session.add(
                    Segment(
                        id=uuid7(),
                        recording_id=recording_id,
                        speaker_id=speaker_id,
                        original_speaker_id=speaker_id,
                        start_s=round(draft.start_s, 3),
                        end_s=round(draft.end_s, 3),
                        text=draft.text,
                        words=[
                            {"start_s": round(w.start_s, 3), "end_s": round(w.end_s, 3),
                             "text": w.text, "prob": round(w.prob, 3)}
                            for w in draft.words
                        ],
                        origin=draft.origin,
                        is_overlap=draft.is_overlap,
                        overlap_region_id=(
                            region_ids[draft.region_index] if draft.region_index is not None else None
                        ),
                        confidence=result.confidence,
                        confidence_parts=result.parts,
                        needs_review=result.needs_review,
                    )
                )
            recording.status = "ready"
            recording.stage = None
            recording.error = None
            recording.transcript_version = 1
            recording.pipeline_config = self._pipeline_config(analysis)

    def _apply_retention(self, recording_id: str) -> None:
        retention = self._settings.retention
        if retention.delete_raw_audio_after_processing:
            self._files.delete_raw(recording_id)
            self._update(recording_id, raw_audio_deleted_at=utcnow())
        if not retention.keep_playback_audio:
            self._files.delete_playback(recording_id)
```

Notes:
- `_set_stage` returns a detached `Recording`. That's safe because sessions use `expire_on_commit=False`, so `recording.min_speakers` and the other attributes stay readable.
- In `test_process_raises_recording_gone_when_deleted_mid_run`, `_set_stage("transcribe")` is what raises `RecordingGone`: the diarizer deletes the recording while it runs, and the next stage update finds no row.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd api && uv run pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add api/
git commit -m "feat(pipeline): runner with analysis, persistence, retry safety and retention"
```

---

### Task 10: Job worker, retention sweeper, retry endpoint, app wiring (M2)

**Files:**
- Create: `api/src/saydo/jobs/__init__.py` (empty), `api/src/saydo/jobs/worker.py`, `api/src/saydo/jobs/sweeper.py`
- Modify: `api/src/saydo/api/recordings.py` (add the retry endpoint)
- Modify: `api/src/saydo/main.py` (replace the whole file)
- Modify: `api/tests/conftest.py` (replace the `app` fixture; add `ready_recording`)
- Test: `api/tests/unit/test_worker.py`, `api/tests/unit/test_sweeper.py`, `api/tests/api/test_pipeline_api.py`

**Interfaces:**
- Consumes: `PipelineRunner`, `build_plugins` (Task 9); `delete_recording` (Task 4); `check_plugins` (Task 5).
- Produces:
  - `saydo.jobs.worker`:
    - `describe_error(exc) -> str`
    - `Worker(session_factory, files, runner, *, poll_interval_s=1.0)` with `.recover() -> int`, `.claim_next() -> Job | None`, `.run_once() -> bool`, `.start()` and `.stop(timeout_s=10)`
    - `MAX_ATTEMPTS = 2`
  - `saydo.jobs.sweeper`: `sweep_expired(session_factory, files, now=None) -> int` and `Sweeper(session_factory, files, interval_s=3600)` with `.start()` and `.stop()`.
  - `create_app(settings=None, *, start_background=True, check_environment=True)`. It also sets `app.state.worker` and `app.state.runner`.
  - `POST /api/v1/recordings/{id}/retry`, which returns 202 with `RecordingCreated`, or 409 unless the recording's status is `failed`.
  - Test fixture: `ready_recording` returns the recording ID of a processed 10 s recording.

- [ ] **Step 1: Write the failing tests**

In `api/tests/conftest.py`, replace the `app` fixture and append `ready_recording`:
```python
@pytest.fixture
def app(settings):
    return create_app(settings, start_background=False)


@pytest.fixture
def ready_recording(client, app, wav_bytes) -> str:
    import io

    import numpy as np
    import soundfile as sf

    t = np.arange(16000 * 10) / 16000
    buffer = io.BytesIO()
    sf.write(buffer, (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), 16000, format="WAV")
    rid = client.post(
        "/api/v1/recordings",
        files={"file": ("m.wav", buffer.getvalue(), "audio/wav")},
        data={"source": "upload"},
    ).json()["id"]
    assert app.state.worker.run_once()
    return rid
```

`api/tests/unit/test_worker.py`:
```python
from sqlalchemy import select

from saydo.db.models import Job, Recording
from saydo.jobs.worker import Worker, describe_error
from saydo.pipeline.errors import AudioDecodeError, RecordingGone
from saydo.pipeline.run import PipelineRunner, build_plugins


class StubRunner:
    def __init__(self, *effects):
        self.effects = list(effects)
        self.calls = []

    def process(self, recording_id):
        self.calls.append(recording_id)
        effect = self.effects.pop(0) if self.effects else None
        if isinstance(effect, BaseException):
            raise effect


def queue(session_factory, rid):
    with session_factory() as s:
        s.add(Job(recording_id=rid, kind="transcribe", status="queued"))
        s.commit()


def job_for(session_factory, rid):
    with session_factory() as s:
        return s.scalars(select(Job).where(Job.recording_id == rid)).one()


def recording(session_factory, rid):
    with session_factory() as s:
        return s.get(Recording, rid)


def test_run_once_without_jobs_returns_false(session_factory, files):
    assert Worker(session_factory, files, StubRunner()).run_once() is False


def test_successful_job_is_done(session_factory, files, make_recording):
    rid = make_recording()
    queue(session_factory, rid)
    runner = StubRunner()
    assert Worker(session_factory, files, runner).run_once() is True
    assert runner.calls == [rid]
    job = job_for(session_factory, rid)
    assert (job.status, job.attempts) == ("done", 1)
    assert job.finished_at is not None


def test_failure_marks_job_and_recording_failed(session_factory, files, make_recording):
    rid = make_recording()
    queue(session_factory, rid)
    with session_factory() as s:
        s.get(Recording, rid).stage = "normalize"
        s.commit()
    Worker(session_factory, files, StubRunner(AudioDecodeError("Could not decode audio: bad"))).run_once()
    assert job_for(session_factory, rid).status == "failed"
    rec = recording(session_factory, rid)
    assert (rec.status, rec.stage, rec.error) == ("failed", "normalize", "Could not decode audio: bad")


def test_transient_error_is_retried_once(session_factory, files, make_recording):
    rid = make_recording()
    queue(session_factory, rid)
    worker = Worker(session_factory, files, StubRunner(OSError("disk busy"), OSError("disk busy")))
    worker.run_once()
    assert job_for(session_factory, rid).status == "queued"
    worker.run_once()
    job = job_for(session_factory, rid)
    assert (job.status, job.attempts) == ("failed", 2)


def test_recording_deleted_mid_job_cleans_up(session_factory, files, make_recording):
    rid = make_recording()
    queue(session_factory, rid)
    assert files.dir(rid).exists()
    Worker(session_factory, files, StubRunner(RecordingGone(rid))).run_once()
    assert not files.dir(rid).exists()


def test_corrupt_upload_fails_cleanly_and_worker_continues(settings, session_factory, files, make_recording):
    bad = make_recording(garbage=True)
    good = make_recording(4.0)
    queue(session_factory, bad)
    queue(session_factory, good)
    runner = PipelineRunner(settings, session_factory, files, lambda: build_plugins(settings))
    worker = Worker(session_factory, files, runner)
    assert worker.run_once() and worker.run_once()
    failed = recording(session_factory, bad)
    assert (failed.status, failed.stage) == ("failed", "normalize")
    assert failed.error.startswith("Could not")
    assert recording(session_factory, good).status == "ready"


def test_recover_requeues_interrupted_jobs(session_factory, files, make_recording):
    rid = make_recording()
    with session_factory() as s:
        s.add(Job(recording_id=rid, kind="transcribe", status="running"))
        s.get(Recording, rid).status = "processing"
        s.commit()
    assert Worker(session_factory, files, StubRunner()).recover() == 1
    assert job_for(session_factory, rid).status == "queued"
    assert recording(session_factory, rid).status == "queued"


def test_claim_is_exclusive(session_factory, files, make_recording):
    rid = make_recording()
    queue(session_factory, rid)
    worker = Worker(session_factory, files, StubRunner())
    assert worker.claim_next() is not None
    assert worker.claim_next() is None


def test_describe_error_hides_internals_and_explains_oom():
    assert "smaller transcriber model" in describe_error(RuntimeError("CUDA out of memory. Tried"))
    assert describe_error(ValueError("x" * 1000)).startswith("ValueError: ")
    assert len(describe_error(ValueError("x" * 1000))) < 320
```

`api/tests/unit/test_sweeper.py`:
```python
from datetime import timedelta

from saydo.db.models import Recording, utcnow
from saydo.jobs.sweeper import sweep_expired


def test_sweep_deletes_only_expired(session_factory, files, make_recording):
    expired, kept, forever = make_recording(1.0), make_recording(1.0), make_recording(1.0)
    with session_factory() as s:
        s.get(Recording, expired).retain_until = utcnow() - timedelta(hours=1)
        s.get(Recording, kept).retain_until = utcnow() + timedelta(days=1)
        s.commit()
    assert sweep_expired(session_factory, files) == 1
    with session_factory() as s:
        assert s.get(Recording, expired) is None
        assert s.get(Recording, kept) is not None
        assert s.get(Recording, forever) is not None
    assert not files.dir(expired).exists()
```

`api/tests/api/test_pipeline_api.py`:
```python
from saydo.db.models import Recording


def test_end_to_end_with_fake_plugins(client, ready_recording):
    detail = client.get(f"/api/v1/recordings/{ready_recording}").json()
    assert detail["recording"]["status"] == "ready"
    assert detail["recording"]["playback_available"] is True
    assert [s["name"] for s in detail["speakers"]] == ["Speaker 1", "Speaker 2"]
    assert len(detail["overlap_regions"]) == 3
    assert any(s["origin"] == "placeholder" for s in detail["segments"])
    assert all(seg["start_s"] <= nxt["start_s"] for seg, nxt in zip(detail["segments"], detail["segments"][1:]))


def test_retry_requeues_failed_recording(client, app, ready_recording):
    assert client.post(f"/api/v1/recordings/{ready_recording}/retry").status_code == 409
    with app.state.session_factory() as s:
        rec = s.get(Recording, ready_recording)
        rec.status, rec.error = "failed", "boom"
        s.commit()
    response = client.post(f"/api/v1/recordings/{ready_recording}/retry")
    assert response.status_code == 202
    assert client.get(f"/api/v1/recordings/{ready_recording}/status").json()["status"] == "queued"
    assert app.state.worker.run_once()
    assert client.get(f"/api/v1/recordings/{ready_recording}/status").json()["status"] == "ready"


def test_unknown_plugin_name_fails_at_startup(settings):
    import pytest

    from saydo.main import create_app
    from saydo.pipeline.registry import PluginNotFoundError

    settings.pipeline.diarizer.name = "does-not-exist"
    with pytest.raises(PluginNotFoundError):
        create_app(settings, start_background=False)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd api && uv run pytest -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'saydo.jobs'` and `TypeError: create_app() got an unexpected keyword argument 'start_background'`.

- [ ] **Step 3: Implement worker and sweeper**

`api/src/saydo/jobs/worker.py`:
```python
"""Single background worker that drains the job table (one job at a time; it owns the GPU)."""

from __future__ import annotations

import logging
import threading
from typing import Protocol

from sqlalchemy import select, update
from sqlalchemy.orm import Session, sessionmaker

from saydo.db.models import Job, Recording, utcnow
from saydo.pipeline.errors import PipelineError, RecordingGone
from saydo.storage.files import RecordingFiles

log = logging.getLogger("saydo.worker")

MAX_ATTEMPTS = 2
TRANSIENT_ERRORS: tuple[type[BaseException], ...] = (OSError,)


class Runner(Protocol):
    def process(self, recording_id: str) -> None: ...


def describe_error(exc: BaseException) -> str:
    text = str(exc)
    if "out of memory" in text.lower():
        return (
            "Ran out of GPU memory. Use a smaller transcriber model "
            "(pipeline.transcriber.options.model) or set pipeline.device to cpu."
        )
    if isinstance(exc, PipelineError):
        return text[:500]
    return f"{type(exc).__name__}: {text[:300]}"


class Worker:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        files: RecordingFiles,
        runner: Runner,
        *,
        poll_interval_s: float = 1.0,
    ) -> None:
        self._sessions = session_factory
        self._files = files
        self._runner = runner
        self._poll_interval_s = poll_interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def recover(self) -> int:
        """Jobs left 'running' by a crash or restart go back to the queue."""
        with self._sessions() as session:
            stuck = session.scalars(select(Job).where(Job.status == "running")).all()
            for job in stuck:
                job.status = "queued"
                recording = session.get(Recording, job.recording_id)
                if recording is not None and recording.status == "processing":
                    recording.status = "queued"
            session.commit()
            return len(stuck)

    def claim_next(self) -> Job | None:
        with self._sessions() as session:
            job = session.scalars(
                select(Job).where(Job.status == "queued").order_by(Job.created_at, Job.id).limit(1)
            ).first()
            if job is None:
                return None
            claimed = session.execute(
                update(Job)
                .where(Job.id == job.id, Job.status == "queued")
                .values(status="running", attempts=Job.attempts + 1, started_at=utcnow())
            )
            session.commit()
            if claimed.rowcount != 1:
                return None
            session.refresh(job)
            return job

    def run_once(self) -> bool:
        job = self.claim_next()
        if job is None:
            return False
        try:
            self._runner.process(job.recording_id)
        except RecordingGone:
            log.info("recording %s deleted during processing; cleaning up", job.recording_id)
            self._files.delete_all(job.recording_id)
        except TRANSIENT_ERRORS as exc:
            if job.attempts < MAX_ATTEMPTS:
                log.warning("job %s transient failure (%s); requeueing", job.id, type(exc).__name__)
                self._set_job(job.id, status="queued")
            else:
                self._fail(job, exc)
        except Exception as exc:  # noqa: BLE001 - any failure must not kill the worker
            self._fail(job, exc)
        else:
            self._set_job(job.id, status="done", finished_at=utcnow())
        return True

    def _set_job(self, job_id: str, **values: object) -> None:
        with self._sessions() as session:
            session.execute(update(Job).where(Job.id == job_id).values(**values))
            session.commit()

    def _fail(self, job: Job, exc: BaseException) -> None:
        message = describe_error(exc)
        log.error("job %s for recording %s failed: %s", job.id, job.recording_id, type(exc).__name__)
        with self._sessions() as session:
            session.execute(
                update(Job)
                .where(Job.id == job.id)
                .values(status="failed", error=message, finished_at=utcnow())
            )
            recording = session.get(Recording, job.recording_id)
            if recording is not None:
                recording.status = "failed"
                recording.error = message
            session.commit()
        if recording is None:
            self._files.delete_all(job.recording_id)

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="saydo-worker", daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = 10.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout_s)

    def _loop(self) -> None:
        recovered = self.recover()
        if recovered:
            log.info("requeued %d interrupted job(s)", recovered)
        while not self._stop.is_set():
            try:
                if not self.run_once():
                    self._stop.wait(self._poll_interval_s)
            except Exception:  # noqa: BLE001
                log.exception("worker loop error")
                self._stop.wait(self._poll_interval_s)
```

`api/src/saydo/jobs/sweeper.py`:
```python
"""Hourly hard-delete of recordings past retain_until (spec section 12)."""

from __future__ import annotations

import logging
import threading
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from saydo.db.models import Recording, utcnow
from saydo.services import delete_recording
from saydo.storage.files import RecordingFiles

log = logging.getLogger("saydo.sweeper")


def sweep_expired(
    session_factory: sessionmaker[Session], files: RecordingFiles, now: datetime | None = None
) -> int:
    cutoff = now or utcnow()
    with session_factory() as session:
        expired = session.scalars(
            select(Recording.id).where(
                Recording.retain_until.is_not(None), Recording.retain_until < cutoff
            )
        ).all()
    for recording_id in expired:
        with session_factory() as session:
            delete_recording(session, files, recording_id)
    if expired:
        log.info("retention sweep deleted %d recording(s)", len(expired))
    return len(expired)


class Sweeper:
    def __init__(
        self, session_factory: sessionmaker[Session], files: RecordingFiles, interval_s: float = 3600
    ) -> None:
        self._sessions = session_factory
        self._files = files
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="saydo-sweeper", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(5)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                sweep_expired(self._sessions, self._files)
            except Exception:  # noqa: BLE001
                log.exception("retention sweep failed")
            self._stop.wait(self._interval_s)
```

- [ ] **Step 4: Add the retry endpoint and wire up the app**

Append to `api/src/saydo/api/recordings.py`:
```python
@router.post("/recordings/{recording_id}/retry", status_code=202, response_model=RecordingCreated)
def retry_recording(recording_id: str, session: SessionDep) -> RecordingCreated:
    recording = get_recording_or_404(session, recording_id)
    if recording.status != "failed":
        raise ApiError(409, "not_failed", "Only failed recordings can be retried")
    recording.status = "queued"
    recording.error = None
    session.add(Job(recording_id=recording_id, kind="transcribe", status="queued"))
    session.commit()
    return RecordingCreated(id=recording_id, status="queued")
```

Replace `api/src/saydo/main.py`:
```python
"""FastAPI application factory and the `saydo-api` CLI entry point."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from saydo import __version__
from saydo.api import health, recordings
from saydo.api.errors import install_error_handlers
from saydo.config import Settings
from saydo.db.session import make_engine, make_session_factory, migrate
from saydo.jobs.sweeper import Sweeper
from saydo.jobs.worker import Worker
from saydo.pipeline.normalize import check_ffmpeg
from saydo.pipeline.registry import check_plugins
from saydo.pipeline.run import PipelineRunner, build_plugins
from saydo.storage.files import RecordingFiles


def create_app(
    settings: Settings | None = None,
    *,
    start_background: bool = True,
    check_environment: bool = True,
) -> FastAPI:
    settings = settings or Settings()
    if check_environment:
        check_ffmpeg()
        check_plugins(settings)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    engine = make_engine(settings.resolved_database_url)
    migrate(engine)
    session_factory = make_session_factory(engine)
    files = RecordingFiles(settings.data_dir)
    runner = PipelineRunner(settings, session_factory, files, lambda: build_plugins(settings))
    worker = Worker(session_factory, files, runner)
    sweeper = Sweeper(session_factory, files)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if start_background:
            worker.start()
            sweeper.start()
        yield
        if start_background:
            worker.stop()
            sweeper.stop()

    app = FastAPI(
        title="SayDo API",
        version=__version__,
        openapi_url="/api/v1/openapi.json",
        docs_url="/api/v1/docs",
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.session_factory = session_factory
    app.state.files = files
    app.state.runner = runner
    app.state.worker = worker

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Accept-Ranges", "Content-Length", "Content-Range"],
    )
    install_error_handlers(app)
    app.include_router(health.router)
    app.include_router(recordings.router)
    app.include_router(recordings.audio_router)
    return app


def run() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    uvicorn.run(
        "saydo.main:create_app",
        factory=True,
        host=os.environ.get("SAYDO_HOST", "127.0.0.1"),
        port=int(os.environ.get("SAYDO_PORT", "8000")),
    )
```

Update `api/tests/api/test_health.py` so the tests don't start background threads: change both `create_app(settings)` calls to `create_app(settings, start_background=False)`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd api && uv run pytest -q && uv run ruff check . && uv run mypy src`
Expected: all tests pass; ruff is clean. mypy may report missing stubs only for third-party modules (`ignore_missing_imports` is on); fix any error in `src/saydo`.

- [ ] **Step 6: Smoke-test the real server with fake plugins**

```bash
cd api
SAYDO__DATA_DIR=/tmp/saydo-smoke SAYDO__PIPELINE__DEVICE=cpu \
SAYDO__PIPELINE__TRANSCRIBER__NAME=fake SAYDO__PIPELINE__DIARIZER__NAME=fake SAYDO__PIPELINE__EMBEDDER__NAME=fake \
uv run saydo-api &
sleep 3
ffmpeg -loglevel error -y -f lavfi -i sine=frequency=300:duration=8 /tmp/smoke.wav
ID=$(curl -s -F file=@/tmp/smoke.wav -F source=upload localhost:8000/api/v1/recordings | python3 -c 'import sys,json;print(json.load(sys.stdin)["id"])')
sleep 3
curl -s localhost:8000/api/v1/recordings/$ID/status
kill %1
```
Expected: `{"status":"ready","stage":null,"error":null}`.

- [ ] **Step 7: Commit**

```bash
git add api/
git commit -m "feat(api): background worker with crash recovery, retention sweeper, retry endpoint"
```

---

### Task 11: Real model plugins and plugin contract tests (M2)

**Files:**
- Create: `api/src/saydo/plugins/faster_whisper_transcriber.py`, `api/src/saydo/plugins/pyannote_diarizer.py`, `api/src/saydo/plugins/pyannote_embedder.py`, `api/src/saydo/plugins/multichannel_diarizer.py`
- Modify: `api/tests/audio_helpers.py` (append `pyannote_sample`)
- Test: `api/tests/unit/test_multichannel_diarizer.py`, `api/tests/contract/test_plugin_contracts.py`, `api/tests/integration/test_real_pipeline.py`

**Interfaces:**
- Consumes: the protocols and types (Task 5), `PluginLoadError` and `PipelineError` (Task 3), and `analyze`, `build_plugins` and `number_speakers` (Task 9).
- Produces factories `create(options: dict, device: str)` in each plugin module, matching the paths in `registry.BUILTINS`:
  - `FasterWhisperTranscriber(device, model="large-v3-turbo", compute_type="auto", beam_size=5, vad_filter=True)`;
  - `PyannoteDiarizer(device, model="pyannote/speaker-diarization-community-1")`;
  - `PyannoteEmbedder(device, model="pyannote/wespeaker-voxceleb-resnet34-LM")`;
  - `MultichannelDiarizer(threshold_db=-40.0, frame_s=0.03, min_speech_s=0.2, min_silence_s=0.3)` with `wants_multichannel = True`.
- Test helper: `audio_helpers.pyannote_sample() -> tuple[Path, Path]` returns the paths of `sample.wav` and `sample.rttm`. These are a 30 s AMI excerpt (CC-BY-4.0) shipped in the pyannote.audio repository (MIT), downloaded and cached in `~/.cache/saydo-tests`.

- [ ] **Step 1: Write the failing fast test (multichannel)**

`api/tests/unit/test_multichannel_diarizer.py`:
```python
import numpy as np
import pytest

from saydo.pipeline.errors import PipelineError
from saydo.pipeline.types import Audio
from saydo.plugins.multichannel_diarizer import MultichannelDiarizer

SR = 16000


def tone(seconds: float, start: float, end: float) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    signal = 0.3 * np.sin(2 * np.pi * 220 * t)
    signal[(t < start) | (t >= end)] = 0.0
    return signal.astype(np.float32)


def test_each_channel_becomes_a_speaker():
    audio = Audio(samples=np.stack([tone(4, 0.0, 2.0), tone(4, 1.5, 3.0)]), sample_rate=SR)
    turns = MultichannelDiarizer().diarize(audio).turns
    simplified = [(t.speaker_key, round(t.start_s, 1), round(t.end_s, 1)) for t in turns]
    assert simplified == [("CH1", 0.0, 2.0), ("CH2", 1.5, 3.0)]


def test_short_gaps_are_bridged_and_blips_dropped():
    channel = tone(4, 0.0, 1.0) + tone(4, 1.1, 2.0) + tone(4, 3.0, 3.05)
    audio = Audio(samples=np.stack([channel, np.zeros_like(channel)]), sample_rate=SR)
    turns = MultichannelDiarizer().diarize(audio).turns
    assert [(t.speaker_key, round(t.start_s, 1), round(t.end_s, 1)) for t in turns] == [("CH1", 0.0, 2.0)]


def test_mono_input_is_rejected_with_a_clear_message():
    audio = Audio(samples=tone(1, 0, 1)[None, :], sample_rate=SR)
    with pytest.raises(PipelineError, match="one channel per speaker"):
        MultichannelDiarizer().diarize(audio)


def test_wants_multichannel():
    assert MultichannelDiarizer.wants_multichannel is True
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd api && uv run pytest tests/unit/test_multichannel_diarizer.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'saydo.plugins.multichannel_diarizer'`.

- [ ] **Step 3: Implement the multichannel diarizer**

`api/src/saydo/plugins/multichannel_diarizer.py`:
```python
"""Diarizer for recordings with one channel per speaker (per-person mics, Zoom per-participant
tracks combined into one multi-channel file). Each channel's energy-based voice activity
becomes that speaker's turns. Cross-talk bleeding into other mics can create false overlaps;
raise threshold_db if that happens.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from saydo.pipeline.errors import PipelineError
from saydo.pipeline.types import Audio, Diarization, Turn


class MultichannelDiarizer:
    name = "multichannel"
    wants_multichannel = True

    def __init__(
        self,
        threshold_db: float = -40.0,
        frame_s: float = 0.03,
        min_speech_s: float = 0.2,
        min_silence_s: float = 0.3,
    ) -> None:
        self.threshold_db = threshold_db
        self.frame_s = frame_s
        self.min_speech_s = min_speech_s
        self.min_silence_s = min_silence_s

    def diarize(
        self, audio: Audio, *, min_speakers: int | None = None, max_speakers: int | None = None
    ) -> Diarization:
        if audio.channels < 2:
            raise PipelineError(
                "The multichannel diarizer needs a recording with one channel per speaker; "
                f"this file has {audio.channels} channel."
            )
        turns = [
            Turn(f"CH{channel + 1}", start, end)
            for channel in range(audio.channels)
            for start, end in self._active_spans(audio.samples[channel], audio.sample_rate)
        ]
        return Diarization(tuple(sorted(turns, key=lambda t: (t.start_s, t.speaker_key))))

    def _active_spans(self, samples: np.ndarray, sample_rate: int) -> list[tuple[float, float]]:
        hop = max(1, int(sample_rate * self.frame_s))
        n_frames = len(samples) // hop
        if n_frames == 0:
            return []
        frames = samples[: n_frames * hop].reshape(n_frames, hop)
        level_db = 20 * np.log10(np.sqrt(np.mean(frames**2, axis=1)) + 1e-12)
        active = level_db > self.threshold_db
        frame_s = hop / sample_rate

        spans: list[tuple[float, float]] = []
        start: int | None = None
        for i, is_active in enumerate(active):
            if is_active and start is None:
                start = i
            elif not is_active and start is not None:
                spans.append((start * frame_s, i * frame_s))
                start = None
        if start is not None:
            spans.append((start * frame_s, n_frames * frame_s))

        bridged: list[tuple[float, float]] = []
        for span in spans:
            if bridged and span[0] - bridged[-1][1] < self.min_silence_s:
                bridged[-1] = (bridged[-1][0], span[1])
            else:
                bridged.append(span)
        return [(round(s, 3), round(e, 3)) for s, e in bridged if e - s >= self.min_speech_s]


def create(options: dict[str, Any], device: str) -> MultichannelDiarizer:
    return MultichannelDiarizer(**options)
```

Run: `cd api && uv run pytest tests/unit/test_multichannel_diarizer.py -q`. Expected: PASS.

- [ ] **Step 4: Implement the model-backed plugins**

`api/src/saydo/plugins/faster_whisper_transcriber.py`:
```python
"""Transcriber backed by faster-whisper (CTranslate2, MIT). Models download from Hugging Face
on first use and are cached under HF_HOME; afterwards HF_HUB_OFFLINE=1 works."""

from __future__ import annotations

from typing import Any

from saydo.pipeline.errors import PluginLoadError
from saydo.pipeline.types import AsrSegment, Audio, Word


class FasterWhisperTranscriber:
    name = "faster_whisper"

    def __init__(
        self,
        *,
        device: str,
        model: str = "large-v3-turbo",
        compute_type: str = "auto",
        beam_size: int = 5,
        vad_filter: bool = True,
    ) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise PluginLoadError(
                "faster-whisper is not installed. Install the ML extras: uv sync --extra ml"
            ) from exc
        # CTranslate2 has no Metal backend: on Apple Silicon Whisper runs on the CPU.
        ct2_device = "cuda" if device == "cuda" else "cpu"
        if compute_type == "auto":
            compute_type = "float16" if ct2_device == "cuda" else "int8"
        try:
            self._model = WhisperModel(model, device=ct2_device, compute_type=compute_type)
        except Exception as exc:  # noqa: BLE001
            raise PluginLoadError(
                f"Could not load Whisper model '{model}': {type(exc).__name__}: {exc}"
            ) from exc
        self.beam_size = beam_size
        self.vad_filter = vad_filter

    def transcribe(self, audio: Audio, *, language: str | None = None) -> list[AsrSegment]:
        segments, _info = self._model.transcribe(
            audio.mono(),
            language=language,
            beam_size=self.beam_size,
            word_timestamps=True,
            vad_filter=self.vad_filter,
            condition_on_previous_text=False,  # limits runaway repetition/hallucination
        )
        out: list[AsrSegment] = []
        for segment in segments:
            words = tuple(
                Word(float(w.start), float(w.end), w.word, float(w.probability))
                for w in (segment.words or [])
            )
            out.append(
                AsrSegment(
                    float(segment.start), float(segment.end), segment.text.strip(), words,
                    float(segment.avg_logprob), float(segment.no_speech_prob),
                )
            )
        return out


def create(options: dict[str, Any], device: str) -> FasterWhisperTranscriber:
    return FasterWhisperTranscriber(device=device, **options)
```

`api/src/saydo/plugins/pyannote_diarizer.py`:
```python
"""Overlap-aware diarization with pyannote.audio 4.x (MIT code; check each model card).

Default model pyannote/speaker-diarization-community-1 is gated: accept its conditions on
Hugging Face once and provide HF_TOKEN. Its segmentation model represents at most two
simultaneous speakers per frame.
"""

from __future__ import annotations

import os

# Privacy: pyannote telemetry stays off unless the operator explicitly enables it.
os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "0")
# Some ops lack MPS kernels; let PyTorch fall back to CPU for them on Apple Silicon.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

from saydo.pipeline.errors import PluginLoadError  # noqa: E402
from saydo.pipeline.types import Audio, Diarization, Turn  # noqa: E402


def gated_message(model: str, exc: BaseException | None) -> str:
    detail = f" ({type(exc).__name__}: {exc})" if exc else ""
    return (
        f"Could not load '{model}'. If it is gated, accept its conditions at "
        f"https://hf.co/{model} and set HF_TOKEN.{detail}"
    )


class PyannoteDiarizer:
    name = "pyannote"
    wants_multichannel = False

    def __init__(self, *, device: str, model: str = "pyannote/speaker-diarization-community-1") -> None:
        try:
            import torch
            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise PluginLoadError(
                "pyannote.audio is not installed. Install the ML extras: uv sync --extra ml"
            ) from exc
        try:
            pipeline = Pipeline.from_pretrained(model, token=os.environ.get("HF_TOKEN"))
        except Exception as exc:  # noqa: BLE001
            raise PluginLoadError(gated_message(model, exc)) from exc
        if pipeline is None:
            raise PluginLoadError(gated_message(model, None))
        pipeline.to(torch.device(device))
        self._pipeline = pipeline

    def diarize(
        self, audio: Audio, *, min_speakers: int | None = None, max_speakers: int | None = None
    ) -> Diarization:
        import torch

        waveform = torch.from_numpy(np.ascontiguousarray(audio.mono()))[None, :]
        output = self._pipeline(
            {"waveform": waveform, "sample_rate": audio.sample_rate},
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        # pyannote 4 returns DiarizeOutput; legacy 3.1 pipelines return an Annotation.
        annotation = getattr(output, "speaker_diarization", output)
        turns = [
            Turn(str(label), float(segment.start), float(segment.end))
            for segment, _track, label in annotation.itertracks(yield_label=True)
        ]
        return Diarization(tuple(sorted(turns, key=lambda t: (t.start_s, t.speaker_key))))


def create(options: dict[str, Any], device: str) -> PyannoteDiarizer:
    return PyannoteDiarizer(device=device, **options)
```

`api/src/saydo/plugins/pyannote_embedder.py`:
```python
"""Speaker embeddings (WeSpeaker ResNet34 via pyannote.audio) for confidence margins."""

from __future__ import annotations

import os

os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

from saydo.pipeline.errors import PluginLoadError  # noqa: E402
from saydo.plugins.pyannote_diarizer import gated_message  # noqa: E402


class PyannoteEmbedder:
    name = "pyannote"

    def __init__(self, *, device: str, model: str = "pyannote/wespeaker-voxceleb-resnet34-LM") -> None:
        try:
            import torch
            from pyannote.audio import Inference, Model
        except ImportError as exc:
            raise PluginLoadError(
                "pyannote.audio is not installed. Install the ML extras: uv sync --extra ml"
            ) from exc
        try:
            loaded = Model.from_pretrained(model, token=os.environ.get("HF_TOKEN"))
        except Exception as exc:  # noqa: BLE001
            raise PluginLoadError(gated_message(model, exc)) from exc
        if loaded is None:
            raise PluginLoadError(gated_message(model, None))
        self._inference = Inference(loaded, window="whole")
        self._inference.to(torch.device(device))

    def embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        import torch

        waveform = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32))[None, :]
        embedding = self._inference({"waveform": waveform, "sample_rate": sample_rate})
        return np.asarray(embedding, dtype=np.float32).reshape(-1)


def create(options: dict[str, Any], device: str) -> PyannoteEmbedder:
    return PyannoteEmbedder(device=device, **options)
```

- [ ] **Step 5: Add the sample-audio helper and the contract and integration tests**

Append to `api/tests/audio_helpers.py`:
```python
import urllib.request  # noqa: E402

SAMPLE_BASE = "https://raw.githubusercontent.com/pyannote/pyannote-audio/main/tutorials/assets/"


def pyannote_sample() -> tuple[Path, Path]:
    """30 s AMI excerpt (CC-BY-4.0) with a reference RTTM, from the pyannote.audio repo (MIT)."""
    cache = Path.home() / ".cache" / "saydo-tests"
    cache.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in ("sample.wav", "sample.rttm"):
        target = cache / name
        if not target.exists():
            urllib.request.urlretrieve(SAMPLE_BASE + name, target)  # noqa: S310
        paths.append(target)
    return paths[0], paths[1]
```

`api/tests/contract/test_plugin_contracts.py`:
```python
"""Every plugin, built-in or third-party, must satisfy these contracts.

Fake plugins always run. Real plugins run with: uv run --extra ml --extra eval pytest -m slow
"""

import numpy as np
import pytest
from audio_helpers import pyannote_sample

from saydo.config import PluginSettings
from saydo.pipeline.normalize import normalize
from saydo.pipeline.registry import build_plugin, resolve_device
from saydo.pipeline.types import Audio, load_audio

CASES = [
    ("transcriber", "fake", {}),
    ("diarizer", "fake", {}),
    ("embedder", "fake", {}),
    pytest.param("transcriber", "faster_whisper", {"model": "base"}, marks=pytest.mark.slow),
    pytest.param("diarizer", "pyannote", {}, marks=pytest.mark.slow),
    pytest.param("embedder", "pyannote", {}, marks=pytest.mark.slow),
]


@pytest.fixture
def audio(request, tmp_path) -> Audio:
    if "slow" in request.node.keywords:
        wav, _ = pyannote_sample()
        return load_audio(normalize(wav, tmp_path / "s.wav", keep_channels=False).path)
    t = np.arange(16000 * 8) / 16000
    return Audio(samples=(0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)[None, :], sample_rate=16000)


@pytest.mark.parametrize(("kind", "name", "options"), CASES)
def test_plugin_contract(kind, name, options, audio):
    plugin = build_plugin(kind, PluginSettings(name=name, options=options), resolve_device("auto"))
    assert isinstance(plugin.name, str)
    if kind == "transcriber":
        segments = plugin.transcribe(audio)
        assert segments, "speech audio must produce at least one segment"
        for segment in segments:
            assert 0.0 <= segment.start_s <= segment.end_s <= audio.duration_s + 0.5
            starts = [w.start_s for w in segment.words]
            assert starts == sorted(starts)
            assert all(w.start_s <= w.end_s for w in segment.words)
            assert all(0.0 <= w.prob <= 1.0 for w in segment.words)
    elif kind == "diarizer":
        turns = plugin.diarize(audio).turns
        assert turns
        assert all(0.0 <= t.start_s < t.end_s <= audio.duration_s + 0.5 for t in turns)
        assert all(isinstance(t.speaker_key, str) and t.speaker_key for t in turns)
        assert isinstance(plugin.wants_multichannel, bool)
    else:
        first = plugin.embed(audio.slice(0, 3), audio.sample_rate)
        second = plugin.embed(audio.slice(3, 6), audio.sample_rate)
        assert first.ndim == 1 and first.shape == second.shape
        assert np.isfinite(first).all()
```

`api/tests/integration/test_real_pipeline.py`:
```python
"""End-to-end check of the real models on a 30 s AMI excerpt with known overlaps."""

import pytest
from audio_helpers import pyannote_sample

from saydo.config import Settings
from saydo.pipeline.normalize import normalize
from saydo.pipeline.run import analyze, build_plugins
from saydo.pipeline.speakers import number_speakers
from saydo.pipeline.types import load_audio

pytestmark = pytest.mark.slow


def test_real_pipeline_on_ami_sample(tmp_path):
    from pyannote.core import Annotation, Segment
    from pyannote.metrics.diarization import DiarizationErrorRate

    wav, rttm = pyannote_sample()
    audio = load_audio(normalize(wav, tmp_path / "s.wav", keep_channels=False).path)
    settings = Settings(pipeline={"transcriber": {"options": {"model": "base"}}})
    analysis = analyze(audio, build_plugins(settings), settings.review)

    assert len(number_speakers(analysis.drafts)) == 2
    assert analysis.regions, "the sample contains overlapping speech"
    assert any(d.text for d in analysis.drafts)

    reference = Annotation()
    for line in rttm.read_text().splitlines():
        parts = line.split()
        start, duration = float(parts[3]), float(parts[4])
        reference[Segment(start, start + duration)] = parts[7]
    hypothesis = Annotation()
    for turn in analysis.turns:
        hypothesis[Segment(turn.start_s, turn.end_s)] = turn.speaker_key
    der = DiarizationErrorRate()(reference, hypothesis)
    assert der < 0.35, f"DER {der:.3f} is far worse than expected for this clip"
```

- [ ] **Step 6: Run fast tests, then slow tests**

Run: `cd api && uv run pytest -q`
Expected: all fast tests pass; the slow cases are deselected.

Then, once: accept the conditions at https://hf.co/pyannote/speaker-diarization-community-1, create a token at https://hf.co/settings/tokens, and run:
```bash
cd api && HF_TOKEN=hf_xxx uv run --extra ml --extra eval pytest -m slow -q
```
Expected: 4 passed (3 contract cases and 1 integration test). The first run downloads about 2 GB of models. Record the DER the test computed in your commit message.

- [ ] **Step 7: Commit**

```bash
git add api/
git commit -m "feat(plugins): faster-whisper, pyannote diarizer/embedder, multichannel diarizer with contract tests"
```

---

### Task 12: Speaker rename, segment reassignment, speaker merge and review acknowledgement API (M4)

**Files:**
- Create: `api/src/saydo/api/speakers.py`, `api/src/saydo/api/segments.py`
- Modify: `api/src/saydo/api/schemas.py` (append the request models)
- Modify: `api/src/saydo/services.py` (append the helpers)
- Modify: `api/src/saydo/main.py` (include two routers)
- Test: `api/tests/api/test_corrections.py`

**Interfaces:**
- Consumes: `load_detail`, `get_recording_or_404` (Task 4); the `ready_recording` fixture (Task 10).
- Produces:
  - Schemas: `SpeakerRename(display_name: str | None)`, `SpeakerCreate(display_name: str | None)`, `SpeakerMerge(into_speaker_id: str)`, `SegmentPatch(speaker_id: str | None, reviewed: bool | None)`, `SpeakerCreated(speaker: SpeakerOut, detail: RecordingDetail)`.
  - Services: `editable_recording(session, recording_id) -> Recording` (409 unless ready), `get_speaker_or_404(session, speaker_id) -> Speaker` (merged speakers count as not found), `detail_or_404(session, files, recording_id) -> RecordingDetail`, `clean_name(value) -> str | None`, `bump_version(recording) -> None`.
  - Endpoints. Every mutation increments `recording.transcript_version` and returns the updated `RecordingDetail`, so the client can replace its state in one step.
    - `PATCH /api/v1/speakers/{id}`
    - `POST /api/v1/recordings/{id}/speakers` (returns `201 SpeakerCreated`)
    - `POST /api/v1/speakers/{id}/merge`
    - `PATCH /api/v1/segments/{id}`

- [ ] **Step 1: Write the failing tests**

`api/tests/api/test_corrections.py`:
```python
from saydo.db.models import Recording


def detail(client, rid):
    return client.get(f"/api/v1/recordings/{rid}").json()


def test_rename_updates_name_everywhere_and_bumps_version(client, ready_recording):
    before = detail(client, ready_recording)
    speaker = before["speakers"][0]
    response = client.patch(f"/api/v1/speakers/{speaker['id']}", json={"display_name": "  Priya "})
    assert response.status_code == 200
    after = response.json()
    renamed = next(s for s in after["speakers"] if s["id"] == speaker["id"])
    assert (renamed["name"], renamed["label"], renamed["display_name"]) == ("Priya", "Speaker 1", "Priya")
    assert after["recording"]["transcript_version"] == before["recording"]["transcript_version"] + 1
    assert [s["speaker_id"] for s in after["segments"]] == [s["speaker_id"] for s in before["segments"]]


def test_empty_rename_reverts_to_label(client, ready_recording):
    speaker = detail(client, ready_recording)["speakers"][0]
    client.patch(f"/api/v1/speakers/{speaker['id']}", json={"display_name": "Priya"})
    after = client.patch(f"/api/v1/speakers/{speaker['id']}", json={"display_name": "   "}).json()
    assert after["speakers"][0]["name"] == "Speaker 1"


def test_rename_too_long_is_422(client, ready_recording):
    speaker = detail(client, ready_recording)["speakers"][0]
    response = client.patch(f"/api/v1/speakers/{speaker['id']}", json={"display_name": "x" * 81})
    assert response.status_code == 422


def test_create_speaker_gets_next_label(client, ready_recording):
    response = client.post(f"/api/v1/recordings/{ready_recording}/speakers", json={"display_name": "Sam"})
    assert response.status_code == 201
    body = response.json()
    assert (body["speaker"]["label"], body["speaker"]["name"]) == ("Speaker 3", "Sam")
    assert len(body["detail"]["speakers"]) == 3


def test_reassign_segment_keeps_original_and_clears_review(client, ready_recording):
    d = detail(client, ready_recording)
    segment = next(s for s in d["segments"] if s["needs_review"])
    other = next(sp for sp in d["speakers"] if sp["id"] != segment["speaker_id"])
    after = client.patch(f"/api/v1/segments/{segment['id']}", json={"speaker_id": other["id"]}).json()
    updated = next(s for s in after["segments"] if s["id"] == segment["id"])
    assert updated["speaker_id"] == other["id"]
    assert updated["original_speaker_id"] == segment["original_speaker_id"]
    assert (updated["user_corrected"], updated["needs_review"], updated["reviewed"]) == (True, False, True)
    assert after["recording"]["transcript_version"] == d["recording"]["transcript_version"] + 1


def test_reassign_to_speaker_of_other_recording_is_rejected(client, app, ready_recording, wav_bytes):
    other_rid = client.post(
        "/api/v1/recordings", files={"file": ("b.wav", wav_bytes, "audio/wav")}, data={"source": "upload"}
    ).json()["id"]
    app.state.worker.run_once()
    foreign = detail(client, other_rid)["speakers"][0]
    segment = detail(client, ready_recording)["segments"][0]
    response = client.patch(f"/api/v1/segments/{segment['id']}", json={"speaker_id": foreign["id"]})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_speaker"


def test_acknowledge_review(client, ready_recording):
    segment = next(s for s in detail(client, ready_recording)["segments"] if s["needs_review"])
    after = client.patch(f"/api/v1/segments/{segment['id']}", json={"reviewed": True}).json()
    updated = next(s for s in after["segments"] if s["id"] == segment["id"])
    assert updated["reviewed"] is True
    assert updated["speaker_id"] == segment["speaker_id"]


def test_empty_segment_patch_is_422(client, ready_recording):
    segment = detail(client, ready_recording)["segments"][0]
    assert client.patch(f"/api/v1/segments/{segment['id']}", json={}).status_code == 422


def test_merge_moves_segments_and_hides_source(client, ready_recording):
    d = detail(client, ready_recording)
    first, second = d["speakers"]
    after = client.post(f"/api/v1/speakers/{second['id']}/merge", json={"into_speaker_id": first["id"]}).json()
    assert [s["id"] for s in after["speakers"]] == [first["id"]]
    assert {s["speaker_id"] for s in after["segments"]} == {first["id"]}
    assert client.patch(f"/api/v1/speakers/{second['id']}", json={"display_name": "x"}).status_code == 404


def test_merge_into_self_is_422(client, ready_recording):
    speaker = detail(client, ready_recording)["speakers"][0]
    response = client.post(f"/api/v1/speakers/{speaker['id']}/merge", json={"into_speaker_id": speaker["id"]})
    assert response.status_code == 422


def test_edits_require_ready_recording(client, app, ready_recording):
    speaker = detail(client, ready_recording)["speakers"][0]
    with app.state.session_factory() as s:
        s.get(Recording, ready_recording).status = "processing"
        s.commit()
    response = client.patch(f"/api/v1/speakers/{speaker['id']}", json={"display_name": "x"})
    assert response.status_code == 409


def test_unknown_ids_are_404(client):
    missing = "0192f2a0-0000-7000-8000-000000000000"
    assert client.patch(f"/api/v1/speakers/{missing}", json={"display_name": "x"}).status_code == 404
    assert client.patch(f"/api/v1/segments/{missing}", json={"reviewed": True}).status_code == 404
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd api && uv run pytest tests/api/test_corrections.py -q`
Expected: FAIL with 404 responses, because the routes don't exist yet.

- [ ] **Step 3: Implement**

Append to `api/src/saydo/api/schemas.py`:
```python
from pydantic import Field, model_validator  # noqa: E402


class SpeakerRename(BaseModel):
    display_name: str | None = Field(None, max_length=80)


class SpeakerCreate(BaseModel):
    display_name: str | None = Field(None, max_length=80)


class SpeakerMerge(BaseModel):
    into_speaker_id: str


class SegmentPatch(BaseModel):
    speaker_id: str | None = None
    reviewed: bool | None = None

    @model_validator(mode="after")
    def _requires_a_change(self) -> SegmentPatch:
        if self.speaker_id is None and self.reviewed is None:
            raise ValueError("Provide speaker_id and/or reviewed")
        return self


class SpeakerCreated(BaseModel):
    speaker: SpeakerOut
    detail: RecordingDetail
```

Append to `api/src/saydo/services.py`:
```python
def editable_recording(session: Session, recording_id: str) -> Recording:
    recording = get_recording_or_404(session, recording_id)
    if recording.status != "ready":
        raise ApiError(409, "not_ready", "Edits are possible once processing has finished")
    return recording


def get_speaker_or_404(session: Session, speaker_id: str) -> Speaker:
    speaker = session.get(Speaker, speaker_id)
    if speaker is None or speaker.merged_into_id is not None:
        raise ApiError(404, "not_found", "Speaker not found")
    return speaker


def detail_or_404(session: Session, files: RecordingFiles, recording_id: str) -> RecordingDetail:
    detail = load_detail(session, files, recording_id)
    if detail is None:
        raise ApiError(404, "not_found", "Recording not found")
    return detail


def clean_name(value: str | None) -> str | None:
    if value is None:
        return None
    return value.strip() or None


def bump_version(recording: Recording) -> None:
    recording.transcript_version += 1
```

`api/src/saydo/api/speakers.py`:
```python
"""Speaker rename, creation and merge. Names are never copied into segments, so a rename is
one row update and shows everywhere."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select, update

from saydo.api.auth import require_token
from saydo.api.deps import FilesDep, SessionDep
from saydo.api.errors import ApiError
from saydo.api.schemas import RecordingDetail, SpeakerCreate, SpeakerCreated, SpeakerMerge, SpeakerRename
from saydo.db.models import Segment, Speaker
from saydo.services import (
    bump_version,
    clean_name,
    detail_or_404,
    editable_recording,
    get_speaker_or_404,
    speaker_out,
)

router = APIRouter(prefix="/api/v1", tags=["speakers"], dependencies=[Depends(require_token)])
SPEAKER_COLORS = 12


@router.patch("/speakers/{speaker_id}", response_model=RecordingDetail)
def rename_speaker(speaker_id: str, body: SpeakerRename, session: SessionDep, files: FilesDep) -> RecordingDetail:
    speaker = get_speaker_or_404(session, speaker_id)
    recording = editable_recording(session, speaker.recording_id)
    speaker.display_name = clean_name(body.display_name)
    bump_version(recording)
    session.commit()
    return detail_or_404(session, files, recording.id)


@router.post("/recordings/{recording_id}/speakers", status_code=201, response_model=SpeakerCreated)
def create_speaker(recording_id: str, body: SpeakerCreate, session: SessionDep, files: FilesDep) -> SpeakerCreated:
    recording = editable_recording(session, recording_id)
    labels = session.scalars(select(Speaker.label).where(Speaker.recording_id == recording_id)).all()
    number = max((int(label.rsplit(" ", 1)[-1]) for label in labels), default=0) + 1
    speaker = Speaker(
        recording_id=recording_id,
        label=f"Speaker {number}",
        display_name=clean_name(body.display_name),
        color_index=(number - 1) % SPEAKER_COLORS,
    )
    session.add(speaker)
    bump_version(recording)
    session.commit()
    return SpeakerCreated(speaker=speaker_out(speaker), detail=detail_or_404(session, files, recording_id))


@router.post("/speakers/{speaker_id}/merge", response_model=RecordingDetail)
def merge_speaker(speaker_id: str, body: SpeakerMerge, session: SessionDep, files: FilesDep) -> RecordingDetail:
    source = get_speaker_or_404(session, speaker_id)
    target = get_speaker_or_404(session, body.into_speaker_id)
    if source.id == target.id:
        raise ApiError(422, "invalid_speaker", "A speaker cannot be merged into itself")
    if source.recording_id != target.recording_id:
        raise ApiError(422, "invalid_speaker", "Speakers belong to different recordings")
    recording = editable_recording(session, source.recording_id)
    session.execute(update(Segment).where(Segment.speaker_id == source.id).values(speaker_id=target.id))
    source.merged_into_id = target.id
    bump_version(recording)
    session.commit()
    return detail_or_404(session, files, recording.id)
```

`api/src/saydo/api/segments.py`:
```python
"""Manual speaker correction and review acknowledgement for a single segment."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from saydo.api.auth import require_token
from saydo.api.deps import FilesDep, SessionDep
from saydo.api.errors import ApiError
from saydo.api.schemas import RecordingDetail, SegmentPatch
from saydo.db.models import Segment, Speaker
from saydo.services import bump_version, detail_or_404, editable_recording

router = APIRouter(prefix="/api/v1", tags=["segments"], dependencies=[Depends(require_token)])


@router.patch("/segments/{segment_id}", response_model=RecordingDetail)
def patch_segment(segment_id: str, body: SegmentPatch, session: SessionDep, files: FilesDep) -> RecordingDetail:
    segment = session.get(Segment, segment_id)
    if segment is None:
        raise ApiError(404, "not_found", "Segment not found")
    recording = editable_recording(session, segment.recording_id)
    if body.speaker_id is not None:
        speaker = session.get(Speaker, body.speaker_id)
        if (
            speaker is None
            or speaker.recording_id != segment.recording_id
            or speaker.merged_into_id is not None
        ):
            raise ApiError(422, "invalid_speaker", "Speaker does not belong to this recording")
        segment.speaker_id = speaker.id
        segment.user_corrected = True
        segment.needs_review = False
        segment.reviewed = True
    if body.reviewed is not None:
        segment.reviewed = body.reviewed
    bump_version(recording)
    session.commit()
    return detail_or_404(session, files, recording.id)
```

In `api/src/saydo/main.py`, change the import line `from saydo.api import health, recordings` to `from saydo.api import health, recordings, segments, speakers`, and after `app.include_router(recordings.audio_router)` add:
```python
    app.include_router(speakers.router)
    app.include_router(segments.router)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd api && uv run pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add api/
git commit -m "feat(api): speaker rename/create/merge and segment correction endpoints"
```

---

### Task 13: Packaging: API Docker image, compose, CI (API), README and licenses (M0/M7 groundwork)

**Files:**
- Create: `docker/api.Dockerfile`, `docker-compose.yml`, `docker-compose.gpu.yml`, `.dockerignore`
- Create: `.github/workflows/ci.yml`
- Create: `LICENSE`, `README.md`, `THIRD_PARTY_LICENSES.md`

**Interfaces:**
- Consumes: the `saydo-api` entry point (Task 1) and the settings environment variables.
- Produces: the compose service `api` on `127.0.0.1:8000`; volumes `saydo-data` (`/data`) and `saydo-models` (`/models`, which is `HF_HOME`); the CI job `api`. Task 19 adds the `web` service and the `web`/`e2e` CI jobs.

- [ ] **Step 1: Write the Docker files**

`.dockerignore`:
```
**/node_modules
**/.venv
**/.next
**/dist
data
eval/data
.git
```

`docker/api.Dockerfile`:
```dockerfile
# syntax=docker/dockerfile:1
FROM python:3.12-slim

# cpu, or a CUDA flavour listed at https://pytorch.org/get-started/locally (e.g. cu128)
ARG TORCH_FLAVOR=cpu

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SAYDO__DATA_DIR=/data \
    SAYDO_HOST=0.0.0.0 \
    HF_HOME=/models \
    PYANNOTE_METRICS_ENABLED=0

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
 && rm -rf /var/lib/apt/lists/*
RUN pip install uv

WORKDIR /app
COPY api/pyproject.toml api/alembic.ini ./
COPY api/src ./src
RUN uv pip install --system torch --index-url "https://download.pytorch.org/whl/${TORCH_FLAVOR}" \
 && uv pip install --system ".[ml]"

RUN useradd --create-home --uid 1000 saydo && mkdir -p /data /models && chown saydo /data /models
USER saydo
VOLUME ["/data", "/models"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health')"
CMD ["saydo-api"]
```

`docker-compose.yml`:
```yaml
services:
  api:
    build:
      context: .
      dockerfile: docker/api.Dockerfile
      args: { TORCH_FLAVOR: cpu }
    image: saydo-api:cpu
    environment:
      HF_TOKEN: ${HF_TOKEN:-}
      SAYDO__API__TOKEN: ${SAYDO_API_TOKEN:-}
      SAYDO__API__CORS_ORIGINS: '["http://localhost:3000"]'
      # Demo mode without models: uncomment the three lines below.
      # SAYDO__PIPELINE__TRANSCRIBER__NAME: fake
      # SAYDO__PIPELINE__DIARIZER__NAME: fake
      # SAYDO__PIPELINE__EMBEDDER__NAME: fake
    volumes:
      - saydo-data:/data
      - saydo-models:/models
    ports:
      - "127.0.0.1:8000:8000"

volumes:
  saydo-data: {}
  saydo-models: {}
```

`docker-compose.gpu.yml`:
```yaml
# Usage: docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
services:
  api:
    build:
      args: { TORCH_FLAVOR: cu128 }
    image: saydo-api:cuda
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

- [ ] **Step 2: Write CI**

`.github/workflows/ci.yml`:
```yaml
name: ci
on:
  push:
    branches: [main]
  pull_request:

jobs:
  api:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: api
    steps:
      - uses: actions/checkout@v4
      - run: sudo apt-get update && sudo apt-get install -y ffmpeg
      - uses: astral-sh/setup-uv@v6
      - run: uv sync --locked
      - run: uv run ruff check .
      - run: uv run ruff format --check .
      - run: uv run mypy src
      - run: uv run pytest -q
```

- [ ] **Step 3: Write the license files and README**

```bash
curl -sSL https://www.apache.org/licenses/LICENSE-2.0.txt -o LICENSE
head -3 LICENSE
```
Expected: the first lines contain `Apache License` and `Version 2.0, January 2004`.

`THIRD_PARTY_LICENSES.md`:
```markdown
# Third-party components

Re-verify every row against the upstream model card or repository before each release.

| Component | Used for | License | Notes |
|---|---|---|---|
| faster-whisper | Speech recognition runtime | MIT | |
| Whisper weights (CTranslate2 conversions) | Speech recognition model | MIT | Downloaded at first use |
| pyannote.audio | Diarization / embedding runtime | MIT | Telemetry disabled by SayDo by default |
| pyannote/speaker-diarization-community-1 | Diarization pipeline | CC-BY-4.0 (per model card) | Gated: accept conditions on Hugging Face |
| pyannote/wespeaker-voxceleb-resnet34-LM | Speaker embeddings | CC-BY-4.0 (per model card) | |
| PyTorch | Model runtime | BSD-3-Clause | |
| FastAPI, Starlette, Pydantic, SQLAlchemy, Alembic, uvicorn | API | MIT / BSD | |
| NumPy, soundfile | Audio arrays and I/O | BSD-3-Clause | libsndfile (LGPL-2.1) is dynamically linked |
| FFmpeg | Audio decoding | LGPL-2.1+ / GPL depending on build | Invoked as an external program, not linked |
| AMI Meeting Corpus excerpt (tests and evaluation only) | Test fixture / evaluation data | CC-BY-4.0 | Downloaded at test time; not redistributed |
```

`README.md`:
````markdown
# SayDo

Turn a recorded meeting into a speaker-labelled transcript, with overlapping speech handled
honestly instead of hidden, and (Phase 2) into per-person action items. Self-hosted, open
source (Apache-2.0), with swappable models.

> Status: Phase 1 (transcript, diarization, overlap handling, corrections). Action items are next.

## What it does
- Record in the browser (microphone, a browser tab, or both) or upload a file.
- Transcribe with Whisper and attribute speech to "Speaker 1", "Speaker 2", ... by voice.
- Detect overlapping speech, show it as a side-by-side block, and add a placeholder when a
  second speaker's words could not be transcribed.
- Score every segment and flag doubtful ones as **Needs review** instead of presenting guesses.
- Rename a speaker once and it updates everywhere; reassign a segment; merge two speakers.

## Quick start (Docker)
1. Accept the conditions of `pyannote/speaker-diarization-community-1` on Hugging Face and create
   a read token at https://hf.co/settings/tokens.
2. `HF_TOKEN=hf_xxx docker compose up --build` (GPU: add `-f docker-compose.gpu.yml`).
3. Open http://localhost:3000. The first transcription downloads about 2 GB of models.

No token yet? Uncomment the three "demo mode" lines in `docker-compose.yml` to run with fake
models and explore the UI.

## Local development
- API: `cd api && uv sync && uv run saydo-api` (needs ffmpeg; `uv sync --extra ml` for real models).
- Web: `pnpm install && pnpm dev` (see `web/`).
- Tests: `cd api && uv run pytest`, `pnpm test`, `pnpm --filter saydo-web e2e`.

## Recording guide
- In a room: a central omnidirectional or boundary microphone within about 1–2 m of everyone,
  not a laptop at one end of the table. A quiet room with soft furnishings. Ask everyone to say
  their name at the start; it makes renaming trivial.
- Best quality: one microphone per person recorded as a multi-channel file; set
  `pipeline.diarizer.name: multichannel` and diarization becomes channel-per-speaker.
- Online meetings: prefer uploading the platform's own recording. In Chrome/Edge you can
  capture a meeting running in a browser tab ("Browser tab" source, tick "Share tab audio").
  Capturing a desktop meeting app's audio is OS-dependent and not supported in Firefox/Safari.
- SayDo turns off the browser's echo cancellation, noise suppression and auto gain: they hide
  the quieter voice when people talk over each other.

## Configuration
Copy `saydo.example.yaml` to `saydo.yaml`. Any key can be overridden with
`SAYDO__SECTION__KEY` (e.g. `SAYDO__PIPELINE__TRANSCRIBER__OPTIONS__MODEL=small`).

## Embedding in your app
- REST API under `/api/v1` (OpenAPI at `/api/v1/docs`). Set `api.token` to require a bearer token.
- React components from `@saydo/react`: `<SaydoProvider>`, `<MeetingRecorder>`, `<TranscriptView>`.

## Swapping models
Plugins are chosen by name in `pipeline.transcriber|diarizer|embedder`. Third-party packages
register a factory `create(options, device)` under the entry-point group `saydo.plugins`
named `<kind>:<name>`. Every plugin must pass `api/tests/contract/test_plugin_contracts.py`.

## Privacy and data handling
- Everything runs on your machine. Nothing leaves it.
- Files live under `data_dir/recordings/<id>/`. Logs never contain audio or transcript text.
- Deleting a recording removes its audio, normalized audio, voice embeddings and all
  transcript rows. `retention.days` deletes recordings automatically;
  `retention.delete_raw_audio_after_processing` removes the original upload once transcribed.
- After the first run has downloaded the models, set `HF_HUB_OFFLINE=1` so nothing contacts
  Hugging Face again. pyannote telemetry is disabled by default.
- Encrypt the data volume at rest with your OS or volume tooling. Backups are your responsibility.

## Known limitations
- **Overlapping speech:** SayDo detects when two people talk at once and shows both, but
  Whisper transcribes a single stream, so the quieter speaker's words are usually lost; you see
  a placeholder flagged for review. Speech separation is planned as an opt-in plugin.
- **At most two simultaneous speakers** are represented per moment by the diarization model.
- **Short interjections** ("yeah", "ok") are the hardest to attribute.
- **Similar voices** may be merged into one speaker; use *Merge* / *Reassign* to fix.
- **More than about 8 speakers**, noisy rooms, echoey rooms and distant microphones all
  increase errors.
- **Confidence is a heuristic score, not a probability.** It ranks what to review first.
- Whisper can produce text during silence or noise; SayDo filters the most likely cases.
- Only English has been evaluated so far.
- Browser capture limits are described in the recording guide above.

## Accuracy
Evaluation reports live in `eval/reports/`. Reproduce with `eval/README.md`.

## License
Apache-2.0. Third-party models and libraries: see `THIRD_PARTY_LICENSES.md`.
````

- [ ] **Step 4: Verify the image builds and serves**

```bash
docker build -f docker/api.Dockerfile -t saydo-api:cpu .
docker run --rm -d --name saydo-smoke -p 127.0.0.1:8001:8000 saydo-api:cpu
sleep 8 && curl -s localhost:8001/api/v1/health && docker stop saydo-smoke
```
Expected: `{"status":"ok","version":"0.1.0"}`. The build takes several minutes because of torch.

- [ ] **Step 5: Commit**

```bash
git add .dockerignore docker/ docker-compose.yml docker-compose.gpu.yml .github/ LICENSE README.md THIRD_PARTY_LICENSES.md
git commit -m "build: api docker image, compose, CI, README and license files"
```

---

### Task 14: JS workspace, `@saydo/react` scaffold, typed API client, provider and styles (M1)

**Files:**
- Create: `package.json`, `pnpm-workspace.yaml`, `tsconfig.base.json`
- Create: `api/scripts/export_openapi.py`
- Create: `packages/react/package.json`, `packages/react/tsconfig.json`, `packages/react/tsup.config.ts`, `packages/react/vitest.config.ts`, `packages/react/vitest.setup.ts`
- Generate: `packages/react/openapi.json`, `packages/react/src/client/schema.d.ts`
- Create: `packages/react/src/client/client.ts`, `packages/react/src/context.tsx`, `packages/react/src/format.ts`, `packages/react/src/styles.css`, `packages/react/src/index.ts`
- Test: `packages/react/src/client/client.test.ts`, `packages/react/src/context.test.tsx`, `packages/react/src/format.test.ts`

**Interfaces:**
- Consumes: the OpenAPI document of the API (Tasks 4, 10, 12).
- Produces:
  - Types: `RecordingDetail`, `RecordingOut`, `RecordingSummary`, `RecordingCreated`, `RecordingStatus`, `RecordingSource`, `StatusOut`, `Speaker`, `Segment`, `OverlapRegion`, `SpeakerCreated`, and `UploadInput { file: Blob; filename: string; source: RecordingSource; title?: string; minSpeakers?: number; maxSpeakers?: number; language?: string }`.
  - `SaydoApiError(status: number, code: string, message: string)`.
  - `createSaydoClient({ baseUrl, token?, fetch? })` returns a client with these methods:
    - `uploadRecording(input: UploadInput): Promise<RecordingCreated>`
    - `listRecordings(): Promise<RecordingSummary[]>`
    - `getRecording(id): Promise<RecordingDetail>`
    - `getStatus(id): Promise<StatusOut>`
    - `retryRecording(id): Promise<RecordingCreated>`
    - `deleteRecording(id): Promise<void>`
    - `getAudioUrl(id): Promise<string>` (absolute URL)
    - `renameSpeaker(speakerId, displayName: string | null): Promise<RecordingDetail>`
    - `createSpeaker(recordingId, displayName?: string): Promise<SpeakerCreated>`
    - `mergeSpeaker(speakerId, intoSpeakerId): Promise<RecordingDetail>`
    - `reassignSegment(segmentId, speakerId): Promise<RecordingDetail>`
    - `markReviewed(segmentId, reviewed?: boolean): Promise<RecordingDetail>`
  - `SaydoClient = ReturnType<typeof createSaydoClient>`.
  - `<SaydoProvider apiBaseUrl? token? client? />` and `useSaydoClient(): SaydoClient`.
  - `formatTimestamp(seconds: number): string`, which gives `"mm:ss"`, or `"h:mm:ss"` from one hour up.
  - `cx(...classes: Array<string | false | null | undefined>): string`.
  - CSS class names used by later tasks, all defined in `styles.css`:
    - recorder: `saydo-recorder`, `saydo-recorder-controls`, `saydo-banner`, `saydo-file`, `saydo-timer`;
    - buttons: `saydo-btn` and its variants `-primary`, `-record`, `-stop`, plus `saydo-link`;
    - messages: `saydo-error`, `saydo-muted`, `saydo-progress`;
    - transcript: `saydo-transcript`, `saydo-transcript-header`, `saydo-items`, `saydo-segment` and its variants `-review`, `-placeholder`, `saydo-ts`, `saydo-speaker`, `saydo-text`;
    - badges: `saydo-badges`, `saydo-badge` and `saydo-badge-review`;
    - overlap groups: `saydo-overlap`, `saydo-overlap-title`, `saydo-overlap-columns`, `saydo-overlap-column`;
    - speaker menu: `saydo-chip-root`, `saydo-menu`, `saydo-menu-heading`.

    CSS variables: `--saydo-speaker-0` … `--saydo-speaker-11` and `--saydo-speaker-color`.

- [ ] **Step 1: Create the workspace and package config**

`package.json`:
```json
{
  "name": "saydo",
  "private": true,
  "packageManager": "pnpm@11.8.0",
  "engines": { "node": ">=22" },
  "scripts": {
    "build": "pnpm --filter @saydo/react build && pnpm --filter saydo-web build",
    "dev": "pnpm --filter @saydo/react build && pnpm --parallel --filter @saydo/react --filter saydo-web dev",
    "test": "pnpm --filter @saydo/react test",
    "typecheck": "pnpm -r typecheck",
    "gen:client": "cd api && uv run python scripts/export_openapi.py > ../packages/react/openapi.json && cd .. && pnpm --filter @saydo/react gen:client"
  }
}
```

`pnpm-workspace.yaml`:
```yaml
packages:
  - packages/*
  - web
onlyBuiltDependencies:
  - esbuild
```

`tsconfig.base.json`:
```json
{
  "compilerOptions": {
    "target": "ES2022",
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "moduleResolution": "Bundler",
    "jsx": "react-jsx",
    "strict": true,
    "noUncheckedIndexedAccess": true,
    "skipLibCheck": true,
    "esModuleInterop": true,
    "isolatedModules": true,
    "resolveJsonModule": true,
    "forceConsistentCasingInFileNames": true
  }
}
```

`packages/react/package.json`:
```json
{
  "name": "@saydo/react",
  "version": "0.1.0",
  "license": "Apache-2.0",
  "type": "module",
  "exports": {
    ".": { "types": "./dist/index.d.ts", "import": "./dist/index.js" },
    "./styles.css": "./dist/styles.css"
  },
  "files": ["dist"],
  "sideEffects": ["**/*.css"],
  "scripts": {
    "build": "tsup",
    "dev": "tsup --watch",
    "test": "vitest run",
    "typecheck": "tsc --noEmit",
    "gen:client": "openapi-typescript openapi.json -o src/client/schema.d.ts"
  },
  "peerDependencies": {
    "react": "^19.0.0",
    "react-dom": "^19.0.0"
  },
  "devDependencies": {
    "@testing-library/dom": "^10.4.0",
    "@testing-library/jest-dom": "^6.6.0",
    "@testing-library/react": "^16.3.0",
    "@testing-library/user-event": "^14.6.0",
    "@types/react": "^19.3.0",
    "@types/react-dom": "^19.3.0",
    "@vitejs/plugin-react": "^6.1.0",
    "fake-indexeddb": "^6.2.0",
    "jsdom": "^30.1.0",
    "openapi-typescript": "^7.13.0",
    "react": "^19.3.0",
    "react-dom": "^19.3.0",
    "tsup": "^8.5.0",
    "typescript": "^5.9.3",
    "vite": "^8.3.0",
    "vitest": "^5.0.0"
  }
}
```

`packages/react/tsconfig.json`:
```json
{
  "extends": "../../tsconfig.base.json",
  "compilerOptions": { "noEmit": true, "types": ["vitest/globals", "@testing-library/jest-dom"] },
  "include": ["src", "vitest.setup.ts"]
}
```

`packages/react/tsup.config.ts`:
```ts
import { defineConfig } from "tsup";

export default defineConfig({
  entry: ["src/index.ts"],
  format: ["esm"],
  dts: true,
  clean: true,
  sourcemap: true,
  external: ["react", "react-dom"],
  // Hooks and components need the client boundary when imported from Next.js server files.
  banner: { js: '"use client";' },
  onSuccess: "cp src/styles.css dist/styles.css",
});
```

`packages/react/vitest.config.ts`:
```ts
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react()],
  test: { environment: "jsdom", globals: true, setupFiles: ["./vitest.setup.ts"] },
});
```

`packages/react/vitest.setup.ts`:
```ts
import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

afterEach(() => cleanup());
```

`api/scripts/export_openapi.py`:
```python
"""Print the OpenAPI document; used to generate the TypeScript client in packages/react."""

import json
import sys
import tempfile
from pathlib import Path

from saydo.config import Settings
from saydo.main import create_app


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        app = create_app(Settings(data_dir=Path(tmp)), start_background=False, check_environment=False)
        json.dump(app.openapi(), sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Install and generate the client types**

```bash
pnpm install
pnpm gen:client
grep -c "RecordingDetail" packages/react/src/client/schema.d.ts
```
Expected: `pnpm install` succeeds; `schema.d.ts` exists and the count is at least 1.

- [ ] **Step 3: Write the failing tests**

`packages/react/src/client/client.test.ts`:
```ts
import { describe, expect, it, vi } from "vitest";
import { SaydoApiError, createSaydoClient } from "./client";

function jsonResponse(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

describe("createSaydoClient", () => {
  it("uploads multipart form data", async () => {
    const fetch = vi.fn(async () => jsonResponse({ id: "r1", status: "queued" }, 202));
    const client = createSaydoClient({ baseUrl: "http://api.test/", fetch });
    const result = await client.uploadRecording({
      file: new Blob(["abc"], { type: "audio/webm" }),
      filename: "m.webm",
      source: "mic",
      title: "Planning",
      minSpeakers: 2,
    });
    expect(result).toEqual({ id: "r1", status: "queued" });
    const [url, init] = fetch.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe("http://api.test/api/v1/recordings");
    expect(init.method).toBe("POST");
    const form = init.body as FormData;
    expect(form.get("source")).toBe("mic");
    expect(form.get("title")).toBe("Planning");
    expect(form.get("min_speakers")).toBe("2");
    expect(form.get("max_speakers")).toBeNull();
  });

  it("turns the error envelope into SaydoApiError", async () => {
    const fetch = vi.fn(async () =>
      jsonResponse({ error: { code: "not_found", message: "Recording not found" } }, 404),
    );
    const client = createSaydoClient({ baseUrl: "http://api.test", fetch });
    await expect(client.getRecording("x")).rejects.toMatchObject({
      name: "SaydoApiError",
      status: 404,
      code: "not_found",
      message: "Recording not found",
    });
  });

  it("reports network failures", async () => {
    const fetch = vi.fn(async () => {
      throw new TypeError("Failed to fetch");
    });
    const client = createSaydoClient({ baseUrl: "http://api.test", fetch });
    const error = await client.listRecordings().catch((e: unknown) => e);
    expect(error).toBeInstanceOf(SaydoApiError);
    expect((error as SaydoApiError).code).toBe("network_error");
  });

  it("sends the bearer token and JSON bodies", async () => {
    const fetch = vi.fn(async () => jsonResponse({}));
    const client = createSaydoClient({ baseUrl: "http://api.test", token: "s3cret", fetch });
    await client.renameSpeaker("spk 1", "Priya");
    const [url, init] = fetch.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe("http://api.test/api/v1/speakers/spk%201");
    expect(init.method).toBe("PATCH");
    expect(init.headers).toMatchObject({ Authorization: "Bearer s3cret", "Content-Type": "application/json" });
    expect(JSON.parse(init.body as string)).toEqual({ display_name: "Priya" });
  });

  it("returns undefined for 204 and absolute audio URLs", async () => {
    const fetch = vi
      .fn()
      .mockResolvedValueOnce(new Response(null, { status: 204 }))
      .mockResolvedValueOnce(jsonResponse({ url: "/api/v1/recordings/r1/audio?sig=a", expires_at: 1 }));
    const client = createSaydoClient({ baseUrl: "http://api.test", fetch });
    await expect(client.deleteRecording("r1")).resolves.toBeUndefined();
    await expect(client.getAudioUrl("r1")).resolves.toBe("http://api.test/api/v1/recordings/r1/audio?sig=a");
  });
});
```

`packages/react/src/context.test.tsx`:
```tsx
import { render, renderHook, screen } from "@testing-library/react";
import { Component, type ReactNode } from "react";
import { describe, expect, it, vi } from "vitest";
import { createSaydoClient } from "./client/client";
import { SaydoProvider, useSaydoClient } from "./context";

// React 19 reports render errors instead of rethrowing them, so catch them in a boundary.
class Boundary extends Component<{ children: ReactNode }, { message: string | null }> {
  state = { message: null as string | null };
  static getDerivedStateFromError(error: Error) {
    return { message: error.message };
  }
  render() {
    return this.state.message ? <p>{this.state.message}</p> : this.props.children;
  }
}

describe("SaydoProvider", () => {
  it("provides a client built from apiBaseUrl", () => {
    const { result } = renderHook(() => useSaydoClient(), {
      wrapper: ({ children }) => <SaydoProvider apiBaseUrl="http://api.test">{children}</SaydoProvider>,
    });
    expect(typeof result.current.getRecording).toBe("function");
  });

  it("accepts an injected client", () => {
    const client = createSaydoClient({ baseUrl: "http://x" });
    const { result } = renderHook(() => useSaydoClient(), {
      wrapper: ({ children }) => <SaydoProvider client={client}>{children}</SaydoProvider>,
    });
    expect(result.current).toBe(client);
  });

  it("throws a helpful error outside the provider", () => {
    function Probe() {
      useSaydoClient();
      return <span>never</span>;
    }
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => {});
    render(
      <Boundary>
        <Probe />
      </Boundary>,
    );
    expect(screen.getByText(/inside <SaydoProvider>/)).toBeInTheDocument();
    expect(screen.queryByText("never")).toBeNull();
    consoleError.mockRestore();
  });
});
```

`packages/react/src/format.test.ts`:
```ts
import { describe, expect, it } from "vitest";
import { cx, formatTimestamp } from "./format";

describe("formatTimestamp", () => {
  it("formats minutes and hours", () => {
    expect(formatTimestamp(0)).toBe("00:00");
    expect(formatTimestamp(65.9)).toBe("01:05");
    expect(formatTimestamp(3723)).toBe("1:02:03");
    expect(formatTimestamp(-4)).toBe("00:00");
  });
});

describe("cx", () => {
  it("joins truthy class names", () => {
    expect(cx("a", false, null, undefined, "b")).toBe("a b");
  });
});
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `pnpm --filter @saydo/react test`
Expected: FAIL with `Failed to resolve import "./client"` (and likewise for `./context` and `./format`).

- [ ] **Step 5: Implement**

`packages/react/src/client/client.ts`:
```ts
import type { components } from "./schema";

type Schemas = components["schemas"];
export type RecordingDetail = Schemas["RecordingDetail"];
export type RecordingOut = Schemas["RecordingOut"];
export type RecordingSummary = Schemas["RecordingSummary"];
export type RecordingCreated = Schemas["RecordingCreated"];
export type RecordingStatus = RecordingOut["status"];
export type RecordingSource = RecordingOut["source"];
export type StatusOut = Schemas["StatusOut"];
export type Speaker = Schemas["SpeakerOut"];
export type Segment = Schemas["SegmentOut"];
export type OverlapRegion = Schemas["OverlapRegionOut"];
export type SpeakerCreated = Schemas["SpeakerCreated"];
type AudioUrlOut = Schemas["AudioUrlOut"];

export class SaydoApiError extends Error {
  readonly status: number;
  readonly code: string;

  constructor(status: number, code: string, message: string) {
    super(message);
    this.name = "SaydoApiError";
    this.status = status;
    this.code = code;
  }
}

export interface UploadInput {
  file: Blob;
  filename: string;
  source: RecordingSource;
  title?: string;
  minSpeakers?: number;
  maxSpeakers?: number;
  language?: string;
}

export interface SaydoClientOptions {
  baseUrl: string;
  token?: string;
  fetch?: typeof fetch;
}

export function createSaydoClient({ baseUrl, token, fetch: fetchImpl }: SaydoClientOptions) {
  const root = baseUrl.replace(/\/+$/, "");
  const doFetch = fetchImpl ?? ((input: RequestInfo | URL, init?: RequestInit) => fetch(input, init));
  const path = encodeURIComponent;

  async function request<T>(method: string, url: string, body?: unknown): Promise<T> {
    const headers: Record<string, string> = token ? { Authorization: `Bearer ${token}` } : {};
    const init: RequestInit = { method, headers };
    if (body instanceof FormData) {
      init.body = body;
    } else if (body !== undefined) {
      headers["Content-Type"] = "application/json";
      init.body = JSON.stringify(body);
    }
    let response: Response;
    try {
      response = await doFetch(`${root}${url}`, init);
    } catch {
      throw new SaydoApiError(0, "network_error", "Could not reach the SayDo server");
    }
    if (response.status === 204) return undefined as T;
    const data: unknown = await response.json().catch(() => null);
    if (!response.ok) {
      const error = (data as { error?: { code?: string; message?: string } } | null)?.error;
      throw new SaydoApiError(
        response.status,
        error?.code ?? "http_error",
        error?.message ?? `Request failed (${response.status})`,
      );
    }
    return data as T;
  }

  return {
    uploadRecording(input: UploadInput): Promise<RecordingCreated> {
      const form = new FormData();
      form.append("file", input.file, input.filename);
      form.append("source", input.source);
      if (input.title) form.append("title", input.title);
      if (input.minSpeakers) form.append("min_speakers", String(input.minSpeakers));
      if (input.maxSpeakers) form.append("max_speakers", String(input.maxSpeakers));
      if (input.language) form.append("language", input.language);
      return request("POST", "/api/v1/recordings", form);
    },
    listRecordings: () => request<RecordingSummary[]>("GET", "/api/v1/recordings"),
    getRecording: (id: string) => request<RecordingDetail>("GET", `/api/v1/recordings/${path(id)}`),
    getStatus: (id: string) => request<StatusOut>("GET", `/api/v1/recordings/${path(id)}/status`),
    retryRecording: (id: string) =>
      request<RecordingCreated>("POST", `/api/v1/recordings/${path(id)}/retry`),
    deleteRecording: (id: string) => request<void>("DELETE", `/api/v1/recordings/${path(id)}`),
    async getAudioUrl(id: string): Promise<string> {
      const { url } = await request<AudioUrlOut>("GET", `/api/v1/recordings/${path(id)}/audio-url`);
      return `${root}${url}`;
    },
    renameSpeaker: (speakerId: string, displayName: string | null) =>
      request<RecordingDetail>("PATCH", `/api/v1/speakers/${path(speakerId)}`, { display_name: displayName }),
    createSpeaker: (recordingId: string, displayName?: string) =>
      request<SpeakerCreated>("POST", `/api/v1/recordings/${path(recordingId)}/speakers`, {
        display_name: displayName ?? null,
      }),
    mergeSpeaker: (speakerId: string, intoSpeakerId: string) =>
      request<RecordingDetail>("POST", `/api/v1/speakers/${path(speakerId)}/merge`, {
        into_speaker_id: intoSpeakerId,
      }),
    reassignSegment: (segmentId: string, speakerId: string) =>
      request<RecordingDetail>("PATCH", `/api/v1/segments/${path(segmentId)}`, { speaker_id: speakerId }),
    markReviewed: (segmentId: string, reviewed = true) =>
      request<RecordingDetail>("PATCH", `/api/v1/segments/${path(segmentId)}`, { reviewed }),
  };
}

export type SaydoClient = ReturnType<typeof createSaydoClient>;
```

`packages/react/src/context.tsx`:
```tsx
import { type ReactNode, createContext, use, useMemo } from "react";
import { type SaydoClient, createSaydoClient } from "./client/client";

const SaydoContext = createContext<SaydoClient | null>(null);

export interface SaydoProviderProps {
  apiBaseUrl?: string;
  token?: string;
  client?: SaydoClient;
  children: ReactNode;
}

export function SaydoProvider({ apiBaseUrl, token, client, children }: SaydoProviderProps) {
  const value = useMemo(() => {
    if (client) return client;
    if (!apiBaseUrl) throw new Error("SaydoProvider needs apiBaseUrl or client");
    return createSaydoClient({ baseUrl: apiBaseUrl, token });
  }, [apiBaseUrl, token, client]);
  return <SaydoContext value={value}>{children}</SaydoContext>;
}

export function useSaydoClient(): SaydoClient {
  const client = use(SaydoContext);
  if (!client) throw new Error("useSaydoClient must be used inside <SaydoProvider>");
  return client;
}
```

`packages/react/src/format.ts`:
```ts
export function formatTimestamp(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const ss = String(s).padStart(2, "0");
  if (h > 0) return `${h}:${String(m).padStart(2, "0")}:${ss}`;
  return `${String(m).padStart(2, "0")}:${ss}`;
}

export function cx(...classes: Array<string | false | null | undefined>): string {
  return classes.filter(Boolean).join(" ");
}
```

`packages/react/src/styles.css`:
```css
/* @saydo/react styles. Theme by overriding the custom properties on :root or a wrapper. */
:root {
  --saydo-font: system-ui, -apple-system, "Segoe UI", sans-serif;
  --saydo-fg: #1f2328;
  --saydo-muted: #59636e;
  --saydo-bg: #ffffff;
  --saydo-surface: #f6f8fa;
  --saydo-border: #d1d9e0;
  --saydo-accent: #0969da;
  --saydo-danger: #cf222e;
  --saydo-warn-bg: #fff8c5;
  --saydo-warn-fg: #7d4e00;
  --saydo-radius: 8px;
  --saydo-speaker-0: #0969da;
  --saydo-speaker-1: #8250df;
  --saydo-speaker-2: #1a7f37;
  --saydo-speaker-3: #bc4c00;
  --saydo-speaker-4: #bf3989;
  --saydo-speaker-5: #0a7d8c;
  --saydo-speaker-6: #9a6700;
  --saydo-speaker-7: #6639ba;
  --saydo-speaker-8: #116329;
  --saydo-speaker-9: #a40e26;
  --saydo-speaker-10: #3a5a97;
  --saydo-speaker-11: #57606a;
}

.saydo-recorder,
.saydo-transcript {
  font-family: var(--saydo-font);
  color: var(--saydo-fg);
  display: grid;
  gap: 12px;
}
.saydo-recorder fieldset { border: 1px solid var(--saydo-border); border-radius: var(--saydo-radius); }
.saydo-recorder-controls { display: flex; align-items: center; gap: 12px; }
.saydo-timer { font-variant-numeric: tabular-nums; font-size: 1.25rem; }
.saydo-banner { background: var(--saydo-warn-bg); color: var(--saydo-warn-fg); padding: 8px 12px; border-radius: var(--saydo-radius); }
.saydo-file { color: var(--saydo-muted); }

.saydo-btn {
  font: inherit; padding: 6px 14px; border-radius: var(--saydo-radius);
  border: 1px solid var(--saydo-border); background: var(--saydo-surface); cursor: pointer;
}
.saydo-btn:disabled { opacity: 0.6; cursor: default; }
.saydo-btn-primary { background: var(--saydo-accent); border-color: var(--saydo-accent); color: #fff; }
.saydo-btn-record { background: var(--saydo-danger); border-color: var(--saydo-danger); color: #fff; }
.saydo-btn-stop { border-color: var(--saydo-danger); color: var(--saydo-danger); }
.saydo-link { font: inherit; background: none; border: none; color: var(--saydo-accent); cursor: pointer; padding: 0; }

.saydo-error { color: var(--saydo-danger); }
.saydo-muted { color: var(--saydo-muted); }
.saydo-progress { color: var(--saydo-muted); font-style: italic; }

.saydo-transcript-header { display: grid; gap: 8px; }
.saydo-transcript-header audio { width: 100%; }
.saydo-items { list-style: none; margin: 0; padding: 0; display: grid; gap: 6px; }
.saydo-segment {
  display: grid; grid-template-columns: auto auto 1fr auto; gap: 8px; align-items: baseline;
  padding: 6px 8px; border-radius: var(--saydo-radius);
}
.saydo-segment-review { background: var(--saydo-warn-bg); }
.saydo-segment-placeholder .saydo-text { color: var(--saydo-muted); }
.saydo-ts { font: inherit; font-variant-numeric: tabular-nums; color: var(--saydo-muted); background: none; border: none; cursor: pointer; }
.saydo-speaker {
  --saydo-speaker-color: var(--saydo-speaker-0);
  font: inherit; font-weight: 600; color: var(--saydo-speaker-color);
  background: none; border: 1px solid transparent; border-radius: 999px; padding: 0 8px; cursor: pointer;
}
.saydo-speaker:hover, .saydo-speaker[aria-expanded="true"] { border-color: var(--saydo-speaker-color); }
.saydo-text { margin: 0; }
.saydo-badges { display: inline-flex; gap: 6px; align-items: center; }
.saydo-badge { font-size: 0.75rem; border: 1px solid var(--saydo-border); border-radius: 999px; padding: 0 6px; }
.saydo-badge-review { background: var(--saydo-warn-fg); color: #fff; border-color: var(--saydo-warn-fg); }

.saydo-overlap { border: 1px dashed var(--saydo-warn-fg); border-radius: var(--saydo-radius); padding: 6px; }
.saydo-overlap-title { margin: 0 0 4px; font-size: 0.8rem; font-weight: 600; color: var(--saydo-warn-fg); }
.saydo-overlap-columns { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 6px; }
.saydo-overlap-column .saydo-segment { grid-template-columns: auto auto 1fr; }
.saydo-overlap-column .saydo-badges { grid-column: 1 / -1; }

.saydo-chip-root { position: relative; }
.saydo-menu {
  position: absolute; z-index: 10; top: 100%; left: 0; min-width: 220px; display: grid; gap: 2px;
  background: var(--saydo-bg); border: 1px solid var(--saydo-border); border-radius: var(--saydo-radius);
  box-shadow: 0 8px 24px rgb(0 0 0 / 0.12); padding: 6px;
}
.saydo-menu [role="menuitem"] { font: inherit; text-align: left; background: none; border: none; padding: 4px 8px; border-radius: 4px; cursor: pointer; }
.saydo-menu [role="menuitem"]:hover { background: var(--saydo-surface); }
.saydo-menu-heading { margin: 6px 8px 0; font-size: 0.75rem; color: var(--saydo-muted); }

@media (prefers-color-scheme: dark) {
  :root {
    --saydo-fg: #e6edf3; --saydo-muted: #9198a1; --saydo-bg: #0d1117; --saydo-surface: #161b22;
    --saydo-border: #3d444d; --saydo-warn-bg: #3b2e00; --saydo-warn-fg: #e3b341;
  }
}
```

`packages/react/src/index.ts`:
```ts
export {
  SaydoApiError,
  createSaydoClient,
  type OverlapRegion,
  type RecordingCreated,
  type RecordingDetail,
  type RecordingOut,
  type RecordingSource,
  type RecordingStatus,
  type RecordingSummary,
  type SaydoClient,
  type SaydoClientOptions,
  type Segment,
  type Speaker,
  type SpeakerCreated,
  type StatusOut,
  type UploadInput,
} from "./client/client";
export { SaydoProvider, useSaydoClient, type SaydoProviderProps } from "./context";
export { cx, formatTimestamp } from "./format";
```

- [ ] **Step 6: Run tests, typecheck and build**

Run: `pnpm --filter @saydo/react test && pnpm --filter @saydo/react typecheck && pnpm --filter @saydo/react build && ls packages/react/dist`
Expected: tests pass; no type errors; `dist/` contains `index.js`, `index.d.ts` and `styles.css`.

- [ ] **Step 7: Commit**

```bash
git add package.json pnpm-workspace.yaml pnpm-lock.yaml tsconfig.base.json api/scripts packages/react
git commit -m "feat(react): workspace, generated API types, client, provider and base styles"
```

---

### Task 15: Browser capture, crash-safe chunk store and `useRecorder` (M1)

**Files:**
- Create: `packages/react/src/recorder/capture.ts`, `packages/react/src/recorder/chunkStore.ts`, `packages/react/src/recorder/useRecorder.ts`
- Create: `packages/react/src/test/recorderFakes.ts`
- Modify: `packages/react/src/index.ts` (add exports)
- Test: `packages/react/src/recorder/capture.test.ts`, `packages/react/src/recorder/chunkStore.test.ts`, `packages/react/src/recorder/useRecorder.test.tsx`

**Interfaces:**
- Consumes: nothing from earlier front-end tasks.
- Produces:
  - `capture.ts`:
    - `CaptureSource = "mic" | "tab" | "mic+tab"`.
    - `CaptureError(code, message)`, where `code` is one of `"insecure_context" | "unsupported" | "permission_denied" | "no_device" | "no_tab_audio"`.
    - `MIC_CONSTRAINTS` and `MIME_CANDIDATES`.
    - `pickMimeType(isSupported?) -> string | undefined` and `supportsTabCapture(env?) -> boolean`.
    - `interface Capture { stream: MediaStream; analyser: AnalyserNode | null; onEnded(cb: () => void): void; stop(): void }`.
    - `interface CaptureEnv { mediaDevices?: MediaDevices; isSecureContext: boolean; createAudioContext?: () => AudioContext }`.
    - `openCapture(source, env?) -> Promise<Capture>`.
  - `chunkStore.ts`:
    - `interface SessionInfo { sessionId; mimeType; source: CaptureSource; chunks: number; createdAt: number }`.
    - `interface ChunkStore` with `put(sessionId, seq, blob, mimeType, source)`, `sessions()`, `assemble(sessionId) -> Promise<Blob | null>` and `remove(sessionId)`.
    - `openChunkStore(factory?: IDBFactory) -> ChunkStore` (falls back to memory when IndexedDB is unavailable) and `memoryChunkStore() -> ChunkStore`.
  - `useRecorder.ts`:
    - `RecorderState = "idle" | "requesting" | "recording" | "stopping" | "stopped" | "error"`.
    - `interface RecordedAudio { blob; mimeType; source; sessionId; durationMs }`.
    - `interface UseRecorderOptions { store?; openCapture?; createRecorder?; pickMimeType?; timesliceMs?; now? }`.
    - `useRecorder(options?) -> UseRecorder`, which exposes `state`, `error`, `elapsedMs`, `level`, `recording`, `recoverable`, `start(source)`, `stop()`, `reset()`, `recover(sessionId)` and `discard(sessionId)`.
  - Test helpers (`src/test/recorderFakes.ts`): `FakeRecorder` and `fakeCapture() -> { capture, end(): void }`.

- [ ] **Step 1: Write test fakes and failing tests**

`packages/react/src/test/recorderFakes.ts`:
```ts
import { vi } from "vitest";
import type { Capture } from "../recorder/capture";

export class FakeRecorder {
  static last: FakeRecorder | null = null;
  state: "inactive" | "recording" = "inactive";
  mimeType: string;
  timeslice: number | undefined;
  ondataavailable: ((event: { data: Blob }) => void) | null = null;
  onstop: (() => void) | null = null;

  constructor(
    public stream: unknown,
    mimeType?: string,
  ) {
    this.mimeType = mimeType ?? "";
    FakeRecorder.last = this;
  }

  start(timeslice?: number) {
    this.state = "recording";
    this.timeslice = timeslice;
  }

  emit(text: string) {
    this.ondataavailable?.({ data: new Blob([text]) });
  }

  stop() {
    this.state = "inactive";
    this.emit("final");
    queueMicrotask(() => this.onstop?.());
  }
}

export function fakeCapture() {
  let ended: (() => void) | undefined;
  const capture: Capture = {
    stream: {} as MediaStream,
    analyser: null,
    onEnded: (cb) => {
      ended = cb;
    },
    stop: vi.fn(),
  };
  return { capture, end: () => ended?.() };
}

export function recorderOptions(capture: Capture) {
  return {
    openCapture: vi.fn(async () => capture),
    createRecorder: (stream: MediaStream, mimeType: string | undefined) =>
      new FakeRecorder(stream, mimeType) as unknown as MediaRecorder,
    pickMimeType: () => "audio/webm;codecs=opus",
    timesliceMs: 1000,
  };
}
```

`packages/react/src/recorder/capture.test.ts`:
```ts
import { beforeEach, describe, expect, it, vi } from "vitest";
import { CaptureError, MIC_CONSTRAINTS, openCapture, pickMimeType } from "./capture";

function track(kind: "audio" | "video") {
  return { kind, stop: vi.fn(), addEventListener: vi.fn() } as unknown as MediaStreamTrack;
}

function stream(...tracks: MediaStreamTrack[]) {
  return {
    getTracks: () => tracks,
    getAudioTracks: () => tracks.filter((t) => t.kind === "audio"),
    getVideoTracks: () => tracks.filter((t) => t.kind === "video"),
  } as unknown as MediaStream;
}

beforeEach(() => {
  vi.stubGlobal(
    "MediaStream",
    class {
      constructor(public tracks: MediaStreamTrack[]) {}
      getTracks() {
        return this.tracks;
      }
    },
  );
});

function fakeAudioContext() {
  const destination = { stream: stream(track("audio")) };
  return {
    createAnalyser: () => ({ fftSize: 0 }),
    createMediaStreamDestination: () => destination,
    createMediaStreamSource: () => ({ connect: vi.fn() }),
    close: vi.fn(async () => {}),
    destination,
  };
}

describe("openCapture", () => {
  it("refuses insecure contexts", async () => {
    await expect(openCapture("mic", { isSecureContext: false })).rejects.toMatchObject({
      code: "insecure_context",
    });
  });

  it("asks for the mic with browser processing disabled", async () => {
    const getUserMedia = vi.fn(async () => stream(track("audio")));
    await openCapture("mic", { isSecureContext: true, mediaDevices: { getUserMedia } as unknown as MediaDevices });
    expect(getUserMedia).toHaveBeenCalledWith({ audio: MIC_CONSTRAINTS });
    expect(MIC_CONSTRAINTS).toMatchObject({ echoCancellation: false, noiseSuppression: false, autoGainControl: false });
  });

  it("maps a denied permission", async () => {
    const getUserMedia = vi.fn(async () => {
      throw Object.assign(new Error("denied"), { name: "NotAllowedError" });
    });
    const error = await openCapture("mic", {
      isSecureContext: true,
      mediaDevices: { getUserMedia } as unknown as MediaDevices,
    }).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(CaptureError);
    expect((error as CaptureError).code).toBe("permission_denied");
  });

  it("throws no_tab_audio when the shared tab has no audio, stopping its tracks", async () => {
    const video = track("video");
    const mediaDevices = {
      getUserMedia: vi.fn(),
      getDisplayMedia: vi.fn(async () => stream(video)),
    } as unknown as MediaDevices;
    await expect(openCapture("tab", { isSecureContext: true, mediaDevices })).rejects.toMatchObject({
      code: "no_tab_audio",
    });
    expect(video.stop).toHaveBeenCalled();
  });

  it("reports tab capture as unsupported without getDisplayMedia and releases the mic", async () => {
    const mic = track("audio");
    const mediaDevices = { getUserMedia: vi.fn(async () => stream(mic)) } as unknown as MediaDevices;
    await expect(openCapture("mic+tab", { isSecureContext: true, mediaDevices })).rejects.toMatchObject({
      code: "unsupported",
    });
    expect(mic.stop).toHaveBeenCalled();
  });

  it("mixes mic and tab audio through Web Audio", async () => {
    const ctx = fakeAudioContext();
    const mediaDevices = {
      getUserMedia: vi.fn(async () => stream(track("audio"))),
      getDisplayMedia: vi.fn(async () => stream(track("audio"), track("video"))),
    } as unknown as MediaDevices;
    const capture = await openCapture("mic+tab", {
      isSecureContext: true,
      mediaDevices,
      createAudioContext: () => ctx as unknown as AudioContext,
    });
    expect(capture.stream).toBe(ctx.destination.stream);
    capture.stop();
    expect(ctx.close).toHaveBeenCalled();
  });
});

describe("pickMimeType", () => {
  it("prefers webm/opus, then mp4", () => {
    expect(pickMimeType(() => true)).toBe("audio/webm;codecs=opus");
    expect(pickMimeType((t) => t === "audio/mp4")).toBe("audio/mp4");
    expect(pickMimeType(() => false)).toBeUndefined();
  });
});
```

`packages/react/src/recorder/chunkStore.test.ts`:
```ts
// @vitest-environment node
import { IDBFactory } from "fake-indexeddb";
import { describe, expect, it } from "vitest";
import { type ChunkStore, memoryChunkStore, openChunkStore } from "./chunkStore";

async function exercise(store: ChunkStore) {
  await store.put("s1", 1, new Blob(["world"]), "audio/webm", "mic");
  await store.put("s1", 0, new Blob(["hello "]), "audio/webm", "mic");
  await store.put("s2", 0, new Blob(["x"]), "audio/mp4", "tab");

  const sessions = await store.sessions();
  expect(sessions.map((s) => [s.sessionId, s.chunks, s.source])).toEqual([
    ["s1", 2, "mic"],
    ["s2", 1, "tab"],
  ]);

  const blob = await store.assemble("s1");
  expect(blob?.type).toBe("audio/webm");
  expect(await blob?.text()).toBe("hello world");

  await store.remove("s1");
  expect((await store.sessions()).map((s) => s.sessionId)).toEqual(["s2"]);
  expect(await store.assemble("s1")).toBeNull();
}

describe("chunk stores", () => {
  it("IndexedDB store keeps ordered chunks per session", async () => {
    await exercise(openChunkStore(new IDBFactory()));
  });

  it("memory store behaves the same", async () => {
    await exercise(memoryChunkStore());
  });
});
```

`packages/react/src/recorder/useRecorder.test.tsx`:
```tsx
import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { FakeRecorder, fakeCapture, recorderOptions } from "../test/recorderFakes";
import { CaptureError } from "./capture";
import { memoryChunkStore } from "./chunkStore";
import { type RecordedAudio, useRecorder } from "./useRecorder";

describe("useRecorder", () => {
  it("records chunks and returns one blob on stop", async () => {
    const { capture } = fakeCapture();
    const store = memoryChunkStore();
    const { result } = renderHook(() => useRecorder({ store, ...recorderOptions(capture) }));

    await act(() => result.current.start("mic"));
    expect(result.current.state).toBe("recording");
    expect(FakeRecorder.last?.timeslice).toBe(1000);

    act(() => FakeRecorder.last?.emit("abc"));
    let recorded = null as RecordedAudio | null; // cast stops TS narrowing it to null
    await act(async () => {
      recorded = await result.current.stop();
    });

    expect(result.current.state).toBe("stopped");
    expect(recorded!.blob.size).toBe("abcfinal".length);
    expect(recorded!.mimeType).toBe("audio/webm;codecs=opus");
    expect(recorded!.source).toBe("mic");
    expect(result.current.recording).toEqual(recorded);
    expect(capture.stop).toHaveBeenCalled();
  });

  it("persists chunks for crash recovery while recording", async () => {
    const { capture } = fakeCapture();
    const store = memoryChunkStore();
    const { result } = renderHook(() => useRecorder({ store, ...recorderOptions(capture) }));
    await act(() => result.current.start("tab"));
    act(() => FakeRecorder.last?.emit("abc"));
    await waitFor(async () => expect(await store.sessions()).toHaveLength(1));
    expect((await store.sessions())[0]?.source).toBe("tab");
  });

  it("reports capture errors", async () => {
    const options = {
      ...recorderOptions(fakeCapture().capture),
      openCapture: vi.fn(async () => {
        throw new CaptureError("permission_denied", "Microphone access was blocked.");
      }),
    };
    const { result } = renderHook(() => useRecorder({ store: memoryChunkStore(), ...options }));
    await act(() => result.current.start("mic"));
    expect(result.current.state).toBe("error");
    expect(result.current.error).toMatchObject({ code: "permission_denied" });
  });

  it("auto-stops and keeps audio when a captured track ends", async () => {
    const { capture, end } = fakeCapture();
    const { result } = renderHook(() => useRecorder({ store: memoryChunkStore(), ...recorderOptions(capture) }));
    await act(() => result.current.start("tab"));
    act(() => FakeRecorder.last?.emit("abc"));
    act(() => end());
    await waitFor(() => expect(result.current.state).toBe("stopped"));
    expect(result.current.recording?.blob.size).toBe("abcfinal".length);
  });

  it("lists, recovers and discards unfinished sessions", async () => {
    const store = memoryChunkStore();
    await store.put("old", 0, new Blob(["xy"]), "audio/webm", "mic");
    const { result } = renderHook(() =>
      useRecorder({ store, ...recorderOptions(fakeCapture().capture) }),
    );
    await waitFor(() => expect(result.current.recoverable).toHaveLength(1));
    let recovered = null as RecordedAudio | null;
    await act(async () => {
      recovered = await result.current.recover("old");
    });
    expect(recovered!.blob.size).toBe(2);
    await act(() => result.current.discard("old"));
    expect(result.current.recoverable).toHaveLength(0);
  });
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pnpm --filter @saydo/react test`
Expected: FAIL with `Failed to resolve import "./capture"` (and likewise for `./chunkStore` and `./useRecorder`).

- [ ] **Step 3: Implement capture**

`packages/react/src/recorder/capture.ts`:
```ts
export type CaptureSource = "mic" | "tab" | "mic+tab";
export type CaptureErrorCode =
  | "insecure_context"
  | "unsupported"
  | "permission_denied"
  | "no_device"
  | "no_tab_audio";

export class CaptureError extends Error {
  readonly code: CaptureErrorCode;

  constructor(code: CaptureErrorCode, message: string) {
    super(message);
    this.name = "CaptureError";
    this.code = code;
  }
}

/** Browser echo cancellation, noise suppression and auto gain suppress the quieter voice
 *  when people talk over each other, which defeats overlap detection. */
export const MIC_CONSTRAINTS: MediaTrackConstraints = {
  echoCancellation: false,
  noiseSuppression: false,
  autoGainControl: false,
  channelCount: { ideal: 2 },
};

export const MIME_CANDIDATES = ["audio/webm;codecs=opus", "audio/mp4", "audio/webm"] as const;

export function pickMimeType(
  isSupported: (type: string) => boolean = (type) =>
    typeof MediaRecorder !== "undefined" && MediaRecorder.isTypeSupported(type),
): string | undefined {
  return MIME_CANDIDATES.find((type) => isSupported(type));
}

export interface Capture {
  /** The stream to record (mixed when both mic and tab are captured). */
  stream: MediaStream;
  analyser: AnalyserNode | null;
  /** Called when a source track ends, e.g. the user clicks the browser's "Stop sharing". */
  onEnded(callback: () => void): void;
  stop(): void;
}

export interface CaptureEnv {
  mediaDevices?: MediaDevices;
  isSecureContext: boolean;
  createAudioContext?: () => AudioContext;
}

function defaultEnv(): CaptureEnv {
  return {
    mediaDevices: typeof navigator !== "undefined" ? navigator.mediaDevices : undefined,
    isSecureContext: typeof window !== "undefined" ? window.isSecureContext : false,
    createAudioContext: typeof AudioContext !== "undefined" ? () => new AudioContext() : undefined,
  };
}

export function supportsTabCapture(env: CaptureEnv = defaultEnv()): boolean {
  return typeof env.mediaDevices?.getDisplayMedia === "function";
}

function micError(error: unknown): CaptureError {
  const name = (error as { name?: string } | null)?.name;
  if (name === "NotAllowedError" || name === "SecurityError") {
    return new CaptureError(
      "permission_denied",
      "Microphone access was blocked. Allow it in the browser's site settings and try again.",
    );
  }
  if (name === "NotFoundError" || name === "OverconstrainedError") {
    return new CaptureError("no_device", "No microphone was found.");
  }
  return new CaptureError("unsupported", "The microphone could not be started.");
}

export async function openCapture(source: CaptureSource, env: CaptureEnv = defaultEnv()): Promise<Capture> {
  if (!env.isSecureContext) {
    throw new CaptureError("insecure_context", "Recording needs HTTPS (or localhost).");
  }
  const devices = env.mediaDevices;
  if (!devices?.getUserMedia) {
    throw new CaptureError("unsupported", "This browser cannot record audio.");
  }

  const owned: MediaStreamTrack[] = [];
  const inputs: MediaStream[] = [];
  try {
    if (source !== "tab") {
      let mic: MediaStream;
      try {
        mic = await devices.getUserMedia({ audio: MIC_CONSTRAINTS });
      } catch (error) {
        throw micError(error);
      }
      owned.push(...mic.getTracks());
      inputs.push(mic);
    }
    if (source !== "mic") {
      if (typeof devices.getDisplayMedia !== "function") {
        throw new CaptureError("unsupported", "Tab audio capture needs Chrome or Edge.");
      }
      let display: MediaStream;
      try {
        display = await devices.getDisplayMedia({ audio: true, video: true });
      } catch {
        throw new CaptureError("permission_denied", "Screen sharing was cancelled or blocked.");
      }
      owned.push(...display.getTracks());
      for (const video of display.getVideoTracks()) video.stop();
      const audio = display.getAudioTracks();
      if (audio.length === 0) {
        throw new CaptureError(
          "no_tab_audio",
          "No tab audio was shared. Choose a browser tab and tick “Share tab audio”.",
        );
      }
      inputs.push(new MediaStream(audio));
    }
  } catch (error) {
    for (const track of owned) track.stop();
    throw error;
  }

  const context = env.createAudioContext?.() ?? null;
  if (!context && inputs.length > 1) {
    for (const track of owned) track.stop();
    throw new CaptureError("unsupported", "Mixing microphone and tab audio needs Web Audio support.");
  }
  let stream = inputs[0] as MediaStream;
  let analyser: AnalyserNode | null = null;
  if (context) {
    analyser = context.createAnalyser();
    analyser.fftSize = 1024;
    if (inputs.length > 1) {
      const destination = context.createMediaStreamDestination();
      for (const input of inputs) {
        const node = context.createMediaStreamSource(input);
        node.connect(destination);
        node.connect(analyser);
      }
      stream = destination.stream;
    } else {
      context.createMediaStreamSource(stream).connect(analyser);
    }
  }

  const endedCallbacks: Array<() => void> = [];
  for (const track of owned) {
    if (track.kind === "audio") {
      track.addEventListener("ended", () => {
        for (const callback of endedCallbacks) callback();
      });
    }
  }
  return {
    stream,
    analyser,
    onEnded: (callback) => {
      endedCallbacks.push(callback);
    },
    stop: () => {
      for (const track of owned) track.stop();
      void context?.close();
    },
  };
}
```

- [ ] **Step 4: Implement the chunk store**

`packages/react/src/recorder/chunkStore.ts`:
```ts
import type { CaptureSource } from "./capture";

/** Recording chunks are persisted while recording so a crashed or closed tab doesn't lose
 *  the meeting. Chunks stay on this device until the upload succeeds or the user discards them. */

export interface SessionInfo {
  sessionId: string;
  mimeType: string;
  source: CaptureSource;
  chunks: number;
  createdAt: number;
}

export interface ChunkStore {
  put(sessionId: string, seq: number, blob: Blob, mimeType: string, source: CaptureSource): Promise<void>;
  sessions(): Promise<SessionInfo[]>;
  assemble(sessionId: string): Promise<Blob | null>;
  remove(sessionId: string): Promise<void>;
}

interface ChunkRecord {
  sessionId: string;
  seq: number;
  data: ArrayBuffer | Blob;
  mimeType: string;
  source: CaptureSource;
  createdAt: number;
}

function summarize(records: ChunkRecord[]): SessionInfo[] {
  const sessions = new Map<string, SessionInfo>();
  for (const record of records) {
    const existing = sessions.get(record.sessionId);
    if (existing) {
      existing.chunks += 1;
      existing.createdAt = Math.min(existing.createdAt, record.createdAt);
    } else {
      sessions.set(record.sessionId, {
        sessionId: record.sessionId,
        mimeType: record.mimeType,
        source: record.source,
        chunks: 1,
        createdAt: record.createdAt,
      });
    }
  }
  return [...sessions.values()].sort((a, b) => a.createdAt - b.createdAt);
}

function join(records: ChunkRecord[]): Blob | null {
  const ordered = [...records].sort((a, b) => a.seq - b.seq);
  const first = ordered[0];
  return first ? new Blob(ordered.map((r) => r.data), { type: first.mimeType }) : null;
}

export function memoryChunkStore(): ChunkStore {
  let records: ChunkRecord[] = [];
  return {
    async put(sessionId, seq, blob, mimeType, source) {
      records = records.filter((r) => !(r.sessionId === sessionId && r.seq === seq));
      records.push({ sessionId, seq, data: blob, mimeType, source, createdAt: Date.now() });
    },
    async sessions() {
      return summarize(records);
    },
    async assemble(sessionId) {
      return join(records.filter((r) => r.sessionId === sessionId));
    },
    async remove(sessionId) {
      records = records.filter((r) => r.sessionId !== sessionId);
    },
  };
}

const DB_NAME = "saydo-recorder";
const STORE = "chunks";

function request<T>(req: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function completed(tx: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error);
  });
}

export function openChunkStore(
  factory: IDBFactory | undefined = typeof indexedDB !== "undefined" ? indexedDB : undefined,
): ChunkStore {
  if (!factory) return memoryChunkStore(); // e.g. some private-browsing modes: no crash recovery
  let database: Promise<IDBDatabase> | null = null;
  const db = () =>
    (database ??= new Promise((resolve, reject) => {
      const open = factory.open(DB_NAME, 1);
      open.onupgradeneeded = () => {
        const store = open.result.createObjectStore(STORE, { keyPath: ["sessionId", "seq"] });
        store.createIndex("bySession", "sessionId");
      };
      open.onsuccess = () => resolve(open.result);
      open.onerror = () => reject(open.error);
    }));

  async function forSession(sessionId: string): Promise<ChunkRecord[]> {
    const tx = (await db()).transaction(STORE, "readonly");
    return request(tx.objectStore(STORE).index("bySession").getAll(sessionId) as IDBRequest<ChunkRecord[]>);
  }

  return {
    async put(sessionId, seq, blob, mimeType, source) {
      const data = await blob.arrayBuffer(); // ArrayBuffers clone reliably into IndexedDB
      const tx = (await db()).transaction(STORE, "readwrite");
      tx.objectStore(STORE).put({ sessionId, seq, data, mimeType, source, createdAt: Date.now() });
      await completed(tx);
    },
    async sessions() {
      const tx = (await db()).transaction(STORE, "readonly");
      return summarize(await request(tx.objectStore(STORE).getAll() as IDBRequest<ChunkRecord[]>));
    },
    async assemble(sessionId) {
      return join(await forSession(sessionId));
    },
    async remove(sessionId) {
      const keys = (await forSession(sessionId)).map((r) => [r.sessionId, r.seq]);
      const tx = (await db()).transaction(STORE, "readwrite");
      const store = tx.objectStore(STORE);
      for (const key of keys) store.delete(key);
      await completed(tx);
    },
  };
}
```

- [ ] **Step 5: Implement the hook**

`packages/react/src/recorder/useRecorder.ts`:
```ts
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { type Capture, type CaptureSource, openCapture as defaultOpenCapture, pickMimeType as defaultPick } from "./capture";
import { type ChunkStore, type SessionInfo, openChunkStore } from "./chunkStore";

export type RecorderState = "idle" | "requesting" | "recording" | "stopping" | "stopped" | "error";

export interface RecordedAudio {
  blob: Blob;
  mimeType: string;
  source: CaptureSource;
  sessionId: string;
  durationMs: number;
}

export interface UseRecorderOptions {
  store?: ChunkStore;
  openCapture?: (source: CaptureSource) => Promise<Capture>;
  createRecorder?: (stream: MediaStream, mimeType: string | undefined) => MediaRecorder;
  pickMimeType?: () => string | undefined;
  timesliceMs?: number;
  now?: () => number;
}

export interface UseRecorder {
  state: RecorderState;
  error: Error | null;
  elapsedMs: number;
  level: number;
  recording: RecordedAudio | null;
  recoverable: SessionInfo[];
  start(source: CaptureSource): Promise<void>;
  stop(): Promise<RecordedAudio | null>;
  reset(): void;
  recover(sessionId: string): Promise<RecordedAudio | null>;
  discard(sessionId: string): Promise<void>;
}

interface ActiveRecording {
  recorder: MediaRecorder;
  capture: Capture;
  chunks: Blob[];
  sessionId: string;
  source: CaptureSource;
  mimeType: string;
  startedAt: number;
  timer: ReturnType<typeof setInterval>;
  stopping: Promise<RecordedAudio | null> | null;
}

function newSessionId(): string {
  return typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function toError(error: unknown): Error {
  return error instanceof Error ? error : new Error(String(error));
}

export function useRecorder(options: UseRecorderOptions = {}): UseRecorder {
  const store = useMemo(() => options.store ?? openChunkStore(), [options.store]);
  const openCapture = options.openCapture ?? defaultOpenCapture;
  const createRecorder =
    options.createRecorder ??
    ((stream: MediaStream, mimeType: string | undefined) =>
      new MediaRecorder(stream, mimeType ? { mimeType } : undefined));
  const choose = options.pickMimeType ?? (() => defaultPick());
  const timesliceMs = options.timesliceMs ?? 5000;
  const now = options.now ?? Date.now;

  const [state, setState] = useState<RecorderState>("idle");
  const [error, setError] = useState<Error | null>(null);
  const [elapsedMs, setElapsedMs] = useState(0);
  const [level, setLevel] = useState(0);
  const [recording, setRecording] = useState<RecordedAudio | null>(null);
  const [recoverable, setRecoverable] = useState<SessionInfo[]>([]);
  const active = useRef<ActiveRecording | null>(null);

  useEffect(() => {
    let cancelled = false;
    store
      .sessions()
      .then((sessions) => {
        if (!cancelled) setRecoverable(sessions);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [store]);

  useEffect(
    () => () => {
      // Unmounting mid-recording: release the mic; chunks already saved stay recoverable.
      const current = active.current;
      if (current) {
        clearInterval(current.timer);
        current.capture.stop();
        active.current = null;
      }
    },
    [],
  );

  const stop = useCallback((): Promise<RecordedAudio | null> => {
    const current = active.current;
    if (!current) return Promise.resolve(null);
    if (current.stopping) return current.stopping;
    setState("stopping");
    current.stopping = new Promise((resolve) => {
      current.recorder.onstop = () => {
        clearInterval(current.timer);
        current.capture.stop();
        const result: RecordedAudio = {
          blob: new Blob(current.chunks, { type: current.mimeType }),
          mimeType: current.mimeType,
          source: current.source,
          sessionId: current.sessionId,
          durationMs: now() - current.startedAt,
        };
        active.current = null;
        setLevel(0);
        setElapsedMs(result.durationMs);
        setRecording(result);
        setState("stopped");
        resolve(result);
      };
      current.recorder.stop();
    });
    return current.stopping;
  }, [now]);

  const start = useCallback(
    async (source: CaptureSource) => {
      if (active.current) return;
      setError(null);
      setRecording(null);
      setElapsedMs(0);
      setState("requesting");
      let capture: Capture;
      try {
        capture = await openCapture(source);
      } catch (e) {
        setError(toError(e));
        setState("error");
        return;
      }
      const preferred = choose();
      let recorder: MediaRecorder;
      try {
        recorder = createRecorder(capture.stream, preferred);
      } catch {
        capture.stop();
        setError(new Error("This browser cannot record in a supported audio format."));
        setState("error");
        return;
      }
      const sessionId = newSessionId();
      const mimeType = recorder.mimeType || preferred || "audio/webm";
      const chunks: Blob[] = [];
      let seq = 0;
      recorder.ondataavailable = (event: BlobEvent) => {
        if (event.data.size > 0) {
          chunks.push(event.data);
          void store.put(sessionId, seq++, event.data, mimeType, source).catch(() => {});
        }
      };
      const startedAt = now();
      const buffer = capture.analyser ? new Uint8Array(capture.analyser.fftSize) : null;
      const timer = setInterval(() => {
        setElapsedMs(now() - startedAt);
        if (capture.analyser && buffer) {
          capture.analyser.getByteTimeDomainData(buffer);
          let sum = 0;
          for (const value of buffer) {
            const x = (value - 128) / 128;
            sum += x * x;
          }
          setLevel(Math.min(1, Math.sqrt(sum / buffer.length) * 4));
        }
      }, 200);
      active.current = { recorder, capture, chunks, sessionId, source, mimeType, startedAt, timer, stopping: null };
      capture.onEnded(() => {
        void stop();
      });
      recorder.start(timesliceMs);
      setState("recording");
    },
    [openCapture, choose, createRecorder, store, timesliceMs, now, stop],
  );

  const reset = useCallback(() => {
    setRecording(null);
    setError(null);
    setElapsedMs(0);
    setState("idle");
  }, []);

  const recover = useCallback(
    async (sessionId: string): Promise<RecordedAudio | null> => {
      const info = recoverable.find((s) => s.sessionId === sessionId);
      const blob = await store.assemble(sessionId);
      if (!blob || !info) return null;
      return { blob, mimeType: blob.type, source: info.source, sessionId, durationMs: 0 };
    },
    [store, recoverable],
  );

  const discard = useCallback(
    async (sessionId: string) => {
      await store.remove(sessionId);
      setRecoverable((sessions) => sessions.filter((s) => s.sessionId !== sessionId));
    },
    [store],
  );

  return { state, error, elapsedMs, level, recording, recoverable, start, stop, reset, recover, discard };
}
```

Add to `packages/react/src/index.ts`:
```ts
export {
  CaptureError,
  type CaptureErrorCode,
  type CaptureSource,
  supportsTabCapture,
} from "./recorder/capture";
export { type ChunkStore, type SessionInfo, memoryChunkStore, openChunkStore } from "./recorder/chunkStore";
export {
  type RecordedAudio,
  type RecorderState,
  type UseRecorder,
  type UseRecorderOptions,
  useRecorder,
} from "./recorder/useRecorder";
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `pnpm --filter @saydo/react test && pnpm --filter @saydo/react typecheck`
Expected: all pass; no type errors.

- [ ] **Step 7: Commit**

```bash
git add packages/react
git commit -m "feat(react): browser capture with overlap-friendly constraints, crash-safe chunk store, useRecorder"
```

---

### Task 16: `<MeetingRecorder>` component (M1)

**Files:**
- Create: `packages/react/src/components/MeetingRecorder.tsx`
- Create: `packages/react/src/test/clientMock.ts`
- Modify: `packages/react/src/index.ts`
- Test: `packages/react/src/components/MeetingRecorder.test.tsx`

**Interfaces:**
- Consumes: `useSaydoClient`, `SaydoApiError` (Task 14); `useRecorder`, `UseRecorderOptions`, `supportsTabCapture` and `CaptureSource` (Task 15); `formatTimestamp` and `cx` (Task 14).
- Produces:
  - `<MeetingRecorder onUploaded?(recordingId: string) sources?: CaptureSource[] recorderOptions?: UseRecorderOptions className? />`.
  - Accessible names the end-to-end test relies on:
    - text field `Title`;
    - radios `Microphone`, `Browser tab` and `Microphone + browser tab`;
    - buttons `Record`, `Stop`, `Upload & transcribe`, `Discard` and `Upload it`;
    - file input `Or upload a file`;
    - number inputs `Minimum speakers` and `Maximum speakers`.
  - Test helper `mockClient(overrides?) -> SaydoClient` (every method is a `vi.fn()`).

- [ ] **Step 1: Write the client mock and failing tests**

`packages/react/src/test/clientMock.ts`:
```ts
import { vi } from "vitest";
import type { SaydoClient } from "../client/client";

export function mockClient(overrides: Partial<Record<keyof SaydoClient, unknown>> = {}): SaydoClient {
  const base: Record<keyof SaydoClient, unknown> = {
    uploadRecording: vi.fn(async () => ({ id: "rec-1", status: "queued" })),
    listRecordings: vi.fn(async () => []),
    getRecording: vi.fn(),
    getStatus: vi.fn(),
    retryRecording: vi.fn(async () => ({ id: "rec-1", status: "queued" })),
    deleteRecording: vi.fn(async () => undefined),
    getAudioUrl: vi.fn(async () => "http://api.test/audio.wav"),
    renameSpeaker: vi.fn(),
    createSpeaker: vi.fn(),
    mergeSpeaker: vi.fn(),
    reassignSegment: vi.fn(),
    markReviewed: vi.fn(),
  };
  return { ...base, ...overrides } as SaydoClient;
}
```

`packages/react/src/components/MeetingRecorder.test.tsx`:
```tsx
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { SaydoApiError } from "../client/client";
import { SaydoProvider } from "../context";
import { CaptureError } from "../recorder/capture";
import { memoryChunkStore } from "../recorder/chunkStore";
import { mockClient } from "../test/clientMock";
import { fakeCapture, recorderOptions } from "../test/recorderFakes";
import { MeetingRecorder } from "./MeetingRecorder";

function setup(client = mockClient(), options = {}) {
  const onUploaded = vi.fn();
  const store = memoryChunkStore();
  const { capture } = fakeCapture();
  render(
    <SaydoProvider client={client}>
      <MeetingRecorder onUploaded={onUploaded} recorderOptions={{ store, ...recorderOptions(capture), ...options }} />
    </SaydoProvider>,
  );
  return { client, onUploaded, store, user: userEvent.setup() };
}

describe("MeetingRecorder", () => {
  it("records, stops and uploads", async () => {
    const { client, onUploaded, user } = setup();
    await user.type(screen.getByLabelText("Title"), "Sprint planning");
    await user.click(screen.getByRole("button", { name: "Record" }));
    await user.click(await screen.findByRole("button", { name: "Stop" }));
    await user.click(await screen.findByRole("button", { name: "Upload & transcribe" }));
    await waitFor(() => expect(onUploaded).toHaveBeenCalledWith("rec-1"));
    expect(client.uploadRecording).toHaveBeenCalledWith(
      expect.objectContaining({
        source: "mic",
        title: "Sprint planning",
        filename: expect.stringMatching(/^meeting-\d{8}-\d{4}\.webm$/),
      }),
    );
    expect(await screen.findByRole("button", { name: "Record" })).toBeEnabled();
  });

  it("uploads a chosen file", async () => {
    const { client, onUploaded, user } = setup();
    const file = new File(["x"], "standup.wav", { type: "audio/wav" });
    await user.upload(screen.getByLabelText("Or upload a file"), file);
    await waitFor(() => expect(onUploaded).toHaveBeenCalledWith("rec-1"));
    expect(client.uploadRecording).toHaveBeenCalledWith(
      expect.objectContaining({ source: "upload", filename: "standup.wav", file }),
    );
  });

  it("passes speaker-count hints", async () => {
    const { client, user } = setup();
    await user.type(screen.getByLabelText("Minimum speakers"), "2");
    await user.type(screen.getByLabelText("Maximum speakers"), "6");
    await user.upload(screen.getByLabelText("Or upload a file"), new File(["x"], "a.wav", { type: "audio/wav" }));
    await waitFor(() =>
      expect(client.uploadRecording).toHaveBeenCalledWith(expect.objectContaining({ minSpeakers: 2, maxSpeakers: 6 })),
    );
  });

  it("shows upload errors", async () => {
    const client = mockClient({
      uploadRecording: vi.fn(async () => {
        throw new SaydoApiError(413, "payload_too_large", "The file exceeds the upload size limit");
      }),
    });
    const { user } = setup(client);
    await user.upload(screen.getByLabelText("Or upload a file"), new File(["x"], "a.wav", { type: "audio/wav" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("The file exceeds the upload size limit");
  });

  it("shows capture errors", async () => {
    const { user } = setup(mockClient(), {
      openCapture: vi.fn(async () => {
        throw new CaptureError("permission_denied", "Microphone access was blocked.");
      }),
    });
    await user.click(screen.getByRole("button", { name: "Record" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Microphone access was blocked.");
  });

  it("offers to upload an unfinished recording", async () => {
    const client = mockClient();
    const store = memoryChunkStore();
    await store.put("old", 0, new Blob(["xy"], { type: "audio/webm" }), "audio/webm", "mic");
    const onUploaded = vi.fn();
    render(
      <SaydoProvider client={client}>
        <MeetingRecorder onUploaded={onUploaded} recorderOptions={{ store, ...recorderOptions(fakeCapture().capture) }} />
      </SaydoProvider>,
    );
    await userEvent.click(await screen.findByRole("button", { name: "Upload it" }));
    await waitFor(() => expect(onUploaded).toHaveBeenCalledWith("rec-1"));
    expect(await store.sessions()).toHaveLength(0);
  });

  it("disables tab sources when the browser cannot capture tabs", () => {
    setup();
    expect(screen.getByLabelText("Microphone")).toBeEnabled();
    expect(screen.getByLabelText("Browser tab")).toBeDisabled();
  });
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pnpm --filter @saydo/react test MeetingRecorder`
Expected: FAIL with `Failed to resolve import "./MeetingRecorder"`.

- [ ] **Step 3: Implement**

`packages/react/src/components/MeetingRecorder.tsx`:
```tsx
import { type ChangeEvent, useEffect, useState } from "react";
import type { RecordingSource } from "../client/client";
import { useSaydoClient } from "../context";
import { cx, formatTimestamp } from "../format";
import { type CaptureSource, supportsTabCapture } from "../recorder/capture";
import { type UseRecorderOptions, useRecorder } from "../recorder/useRecorder";

const SOURCE_LABELS: Record<CaptureSource, string> = {
  mic: "Microphone",
  tab: "Browser tab",
  "mic+tab": "Microphone + browser tab",
};

export interface MeetingRecorderProps {
  onUploaded?: (recordingId: string) => void;
  sources?: CaptureSource[];
  recorderOptions?: UseRecorderOptions;
  className?: string;
}

function recordingFilename(mimeType: string, date = new Date()): string {
  const pad = (n: number) => String(n).padStart(2, "0");
  const stamp = `${date.getFullYear()}${pad(date.getMonth() + 1)}${pad(date.getDate())}-${pad(date.getHours())}${pad(date.getMinutes())}`;
  return `meeting-${stamp}.${mimeType.includes("mp4") ? "m4a" : "webm"}`;
}

function parseCount(value: string): number | undefined {
  const n = Number.parseInt(value, 10);
  return Number.isFinite(n) && n > 0 ? n : undefined;
}

export function MeetingRecorder({
  onUploaded,
  sources = ["mic", "tab", "mic+tab"],
  recorderOptions,
  className,
}: MeetingRecorderProps) {
  const client = useSaydoClient();
  const recorder = useRecorder(recorderOptions);
  const [source, setSource] = useState<CaptureSource>("mic");
  const [title, setTitle] = useState("");
  const [minSpeakers, setMinSpeakers] = useState("");
  const [maxSpeakers, setMaxSpeakers] = useState("");
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [tabSupported, setTabSupported] = useState(false);

  useEffect(() => setTabSupported(supportsTabCapture()), []);

  const isRecording = recorder.state === "recording" || recorder.state === "stopping";
  const busy = isRecording || uploading || recorder.state === "requesting";

  async function upload(file: Blob, filename: string, uploadSource: RecordingSource, sessionId?: string) {
    setUploading(true);
    setUploadError(null);
    try {
      const created = await client.uploadRecording({
        file,
        filename,
        source: uploadSource,
        title: title.trim() || undefined,
        minSpeakers: parseCount(minSpeakers),
        maxSpeakers: parseCount(maxSpeakers),
      });
      if (sessionId) await recorder.discard(sessionId);
      recorder.reset();
      onUploaded?.(created.id);
    } catch (e) {
      setUploadError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  }

  async function uploadRecorded() {
    const recorded = recorder.recording;
    if (recorded) {
      await upload(recorded.blob, recordingFilename(recorded.mimeType), recorded.source, recorded.sessionId);
    }
  }

  async function discardRecorded() {
    if (recorder.recording) await recorder.discard(recorder.recording.sessionId);
    recorder.reset();
  }

  async function recoverAndUpload(sessionId: string) {
    const recovered = await recorder.recover(sessionId);
    if (recovered) {
      await upload(recovered.blob, recordingFilename(recovered.mimeType), recovered.source, sessionId);
    }
  }

  function onFile(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (file) void upload(file, file.name, "upload");
  }

  const errorMessage = uploadError ?? recorder.error?.message ?? null;

  return (
    <section className={cx("saydo-recorder", className)} aria-label="Meeting recorder">
      {recorder.state === "idle" && recorder.recoverable.length > 0 && (
        <div className="saydo-banner" role="status">
          {recorder.recoverable.map((session) => (
            <p key={session.sessionId}>
              An unfinished recording from {new Date(session.createdAt).toLocaleString()} was found.{" "}
              <button type="button" className="saydo-link" disabled={uploading} onClick={() => void recoverAndUpload(session.sessionId)}>
                Upload it
              </button>{" "}
              <button type="button" className="saydo-link" disabled={uploading} onClick={() => void recorder.discard(session.sessionId)}>
                Discard
              </button>
            </p>
          ))}
        </div>
      )}

      <label>
        Title{" "}
        <input value={title} maxLength={200} placeholder="Sprint planning" onChange={(e) => setTitle(e.target.value)} />
      </label>

      <fieldset disabled={busy}>
        <legend>Audio source</legend>
        {sources.map((option) => (
          <label key={option}>
            <input
              type="radio"
              name="saydo-source"
              value={option}
              checked={source === option}
              disabled={option !== "mic" && !tabSupported}
              onChange={() => setSource(option)}
            />{" "}
            {SOURCE_LABELS[option]}
          </label>
        ))}
      </fieldset>

      <details>
        <summary>Speaker count (optional)</summary>
        <label>
          Minimum speakers{" "}
          <input type="number" min={1} max={20} value={minSpeakers} onChange={(e) => setMinSpeakers(e.target.value)} />
        </label>{" "}
        <label>
          Maximum speakers{" "}
          <input type="number" min={1} max={20} value={maxSpeakers} onChange={(e) => setMaxSpeakers(e.target.value)} />
        </label>
      </details>

      <div className="saydo-recorder-controls">
        {isRecording ? (
          <>
            <button
              type="button"
              className="saydo-btn saydo-btn-stop"
              disabled={recorder.state === "stopping"}
              onClick={() => void recorder.stop()}
            >
              Stop
            </button>
            <span className="saydo-timer">{formatTimestamp(recorder.elapsedMs / 1000)}</span>
            <meter min={0} max={1} value={recorder.level} aria-label="Input level" />
          </>
        ) : recorder.state === "stopped" && recorder.recording ? (
          <>
            <span>Recorded {formatTimestamp(recorder.recording.durationMs / 1000)}</span>
            <button type="button" className="saydo-btn saydo-btn-primary" disabled={uploading} onClick={() => void uploadRecorded()}>
              {uploading ? "Uploading…" : "Upload & transcribe"}
            </button>
            <button type="button" className="saydo-btn" disabled={uploading} onClick={() => void discardRecorded()}>
              Discard
            </button>
          </>
        ) : (
          <button
            type="button"
            className="saydo-btn saydo-btn-record"
            disabled={recorder.state === "requesting" || uploading}
            onClick={() => void recorder.start(source)}
          >
            Record
          </button>
        )}
      </div>

      <label className="saydo-file">
        Or upload a file{" "}
        <input type="file" accept="audio/*,video/webm,video/mp4" disabled={busy} onChange={onFile} />
      </label>

      {uploading && <p className="saydo-progress">Uploading…</p>}
      {errorMessage && (
        <p role="alert" className="saydo-error">
          {errorMessage}
        </p>
      )}
    </section>
  );
}
```

Add to `packages/react/src/index.ts`:
```ts
export { MeetingRecorder, type MeetingRecorderProps } from "./components/MeetingRecorder";
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pnpm --filter @saydo/react test && pnpm --filter @saydo/react typecheck`
Expected: all pass. If the "Upload & transcribe" button label is matched by two elements while uploading, the test still passes: it only queries the button before the click.

- [ ] **Step 5: Commit**

```bash
git add packages/react
git commit -m "feat(react): MeetingRecorder with record/stop/upload, file upload and crash recovery"
```

---

### Task 17: `<TranscriptView>`: polling, overlap groups, playback, review filter, retry (M3)

**Files:**
- Create: `packages/react/src/transcript/model.ts`, `packages/react/src/hooks/useRecording.ts`
- Create: `packages/react/src/components/SegmentRow.tsx`, `packages/react/src/components/OverlapGroup.tsx`, `packages/react/src/components/TranscriptView.tsx`
- Create: `packages/react/src/test/fixtures.ts`
- Modify: `packages/react/src/index.ts`
- Test: `packages/react/src/transcript/model.test.ts`, `packages/react/src/components/TranscriptView.test.tsx`

**Interfaces:**
- Consumes: the client types and `useSaydoClient` (Task 14); `formatTimestamp` and `cx` (Task 14).
- Produces:
  - `model.ts`:
    - `TranscriptItem = { kind: "segment"; key; segment } | { kind: "overlap"; key; region; columns: { speakerId; segments }[] }`
    - `buildTranscriptItems(detail, { onlyNeedsReview? }) -> TranscriptItem[]`
    - `needsAttention(segment) -> boolean`, which is `needs_review && !reviewed`
    - `reviewCount(detail) -> number`
    - `STAGE_LABELS: Record<string, string>`
    - `speakerColor(index) -> string`
  - `useRecording(recordingId, { pollMs? }) -> { detail, status, error, setDetail, refresh, retry }`.
  - `TranscriptActions` with `renameSpeaker(speakerId, name: string | null)`, `reassign(segmentId, speakerId)`, `reassignToNew(segmentId, name?)`, `merge(speakerId, intoSpeakerId)` and `markReviewed(segmentId)`, each returning `Promise<void>`. It is exported from `SegmentRow.tsx`.
  - `<SegmentRow segment speakers actions onSeek />`, `<OverlapGroup region columns speakers actions onSeek />` and `<TranscriptView recordingId pollMs? className? />`.
  - Text the end-to-end test relies on:
    - headings: the recording title (`h2`);
    - regions named `Overlapping speech · mm:ss–mm:ss`;
    - badges `Needs review`, `Corrected` and `Reviewed`;
    - the button `Mark reviewed`;
    - the checkbox `Show only items needing review (N)`;
    - the messages `No speech detected.`, `Processing failed` and `Retry`;
    - the placeholder text `[overlapping speech – not transcribed]`.
  - Test fixtures `makeDetail(overrides?) -> RecordingDetail` with `SPEAKER_1 = "spk-1"` and `SPEAKER_2 = "spk-2"`.

- [ ] **Step 1: Write fixtures and failing tests**

`packages/react/src/test/fixtures.ts`:
```ts
import type { RecordingDetail, Segment } from "../client/client";

export const SPEAKER_1 = "spk-1";
export const SPEAKER_2 = "spk-2";

function segment(partial: Partial<Segment> & Pick<Segment, "id" | "speaker_id" | "start_s" | "end_s">): Segment {
  return {
    original_speaker_id: partial.speaker_id,
    text: "",
    words: [],
    origin: "mixture",
    is_overlap: false,
    overlap_region_id: null,
    confidence: 0.9,
    confidence_parts: { asr: 0.9, margin: 0.9, length: 1, penalty: 1 },
    needs_review: false,
    user_corrected: false,
    reviewed: false,
    ...partial,
  };
}

export function makeDetail(overrides: Partial<RecordingDetail["recording"]> = {}): RecordingDetail {
  return {
    recording: {
      id: "rec-1",
      title: "Sprint planning",
      status: "ready",
      source: "mic",
      created_at: "2026-10-09T10:00:00Z",
      duration_s: 10,
      stage: null,
      error: null,
      channels: 1,
      language: null,
      min_speakers: null,
      max_speakers: null,
      transcript_version: 1,
      raw_audio_deleted_at: null,
      retain_until: null,
      playback_available: true,
      ...overrides,
    },
    speakers: [
      { id: SPEAKER_1, label: "Speaker 1", display_name: null, name: "Speaker 1", color_index: 0 },
      { id: SPEAKER_2, label: "Speaker 2", display_name: null, name: "Speaker 2", color_index: 1 },
    ],
    overlap_regions: [{ id: "ov-1", start_s: 3, end_s: 4, n_speakers: 2 }],
    segments: [
      segment({ id: "seg-1", speaker_id: SPEAKER_1, start_s: 0, end_s: 3, text: "Hello team" }),
      segment({
        id: "seg-2", speaker_id: SPEAKER_1, start_s: 3, end_s: 4, text: "I can take it",
        is_overlap: true, overlap_region_id: "ov-1", needs_review: true, confidence: 0.4,
      }),
      segment({
        id: "seg-3", speaker_id: SPEAKER_2, start_s: 3, end_s: 4, origin: "placeholder",
        is_overlap: true, overlap_region_id: "ov-1", needs_review: true, confidence: 0,
        confidence_parts: { asr: 0, margin: 0, length: 0, penalty: 0 },
      }),
      segment({ id: "seg-4", speaker_id: SPEAKER_2, start_s: 4, end_s: 6, text: "Sounds good" }),
    ],
  };
}
```

`packages/react/src/transcript/model.test.ts`:
```ts
import { describe, expect, it } from "vitest";
import { SPEAKER_1, SPEAKER_2, makeDetail } from "../test/fixtures";
import { buildTranscriptItems, reviewCount } from "./model";

describe("buildTranscriptItems", () => {
  it("groups overlapping segments into one item with a column per speaker", () => {
    const items = buildTranscriptItems(makeDetail());
    expect(items.map((i) => i.kind)).toEqual(["segment", "overlap", "segment"]);
    const overlap = items[1];
    if (overlap?.kind !== "overlap") throw new Error("expected overlap");
    expect(overlap.columns.map((c) => c.speakerId)).toEqual([SPEAKER_1, SPEAKER_2]);
  });

  it("filters to items needing review, keeping whole overlap groups", () => {
    const items = buildTranscriptItems(makeDetail(), { onlyNeedsReview: true });
    expect(items.map((i) => i.key)).toEqual(["ov-1"]);
  });

  it("counts unreviewed flags", () => {
    const detail = makeDetail();
    expect(reviewCount(detail)).toBe(2);
    detail.segments[1]!.reviewed = true;
    expect(reviewCount(detail)).toBe(1);
  });
});
```

`packages/react/src/components/TranscriptView.test.tsx`:
```tsx
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { SaydoClient } from "../client/client";
import { SaydoProvider } from "../context";
import { mockClient } from "../test/clientMock";
import { makeDetail } from "../test/fixtures";
import { TranscriptView } from "./TranscriptView";

function renderView(client: SaydoClient) {
  render(
    <SaydoProvider client={client}>
      <TranscriptView recordingId="rec-1" pollMs={10} />
    </SaydoProvider>,
  );
  return userEvent.setup();
}

let seekedTo: number | null = null;
beforeEach(() => {
  seekedTo = null;
  vi.spyOn(window.HTMLMediaElement.prototype, "play").mockResolvedValue(undefined);
  Object.defineProperty(window.HTMLMediaElement.prototype, "currentTime", {
    configurable: true,
    get: () => 0,
    set: (value: number) => {
      seekedTo = value;
    },
  });
});

describe("TranscriptView", () => {
  it("shows progress while processing, then the transcript", async () => {
    const processing = makeDetail({ status: "processing", stage: "transcribe", playback_available: false });
    const client = mockClient({
      getRecording: vi.fn().mockResolvedValueOnce(processing).mockResolvedValue(makeDetail()),
      getStatus: vi
        .fn()
        .mockResolvedValueOnce({ status: "processing", stage: "transcribe", error: null })
        .mockResolvedValue({ status: "ready", stage: null, error: null }),
    });
    renderView(client);
    expect(await screen.findByText("Transcribing…")).toBeInTheDocument();
    expect(await screen.findByRole("heading", { name: "Sprint planning" })).toBeInTheDocument();
  });

  it("renders overlapping speech side by side with a placeholder and review badges", async () => {
    renderView(mockClient({ getRecording: vi.fn(async () => makeDetail()) }));
    const group = await screen.findByRole("region", { name: "Overlapping speech · 00:03–00:04" });
    expect(within(group).getByText("I can take it")).toBeInTheDocument();
    expect(within(group).getByText("[overlapping speech – not transcribed]")).toBeInTheDocument();
    expect(within(group).getAllByText("Needs review")).toHaveLength(2);
    expect(screen.getByText(/not a probability/)).toBeInTheDocument();
  });

  it("filters to items needing review", async () => {
    const user = renderView(mockClient({ getRecording: vi.fn(async () => makeDetail()) }));
    await user.click(await screen.findByLabelText("Show only items needing review (2)"));
    expect(screen.queryByText("Sounds good")).toBeNull();
    expect(screen.getByText("I can take it")).toBeInTheDocument();
  });

  it('shows "No speech detected" for an empty transcript', async () => {
    const empty = { ...makeDetail(), speakers: [], overlap_regions: [], segments: [] };
    renderView(mockClient({ getRecording: vi.fn(async () => empty) }));
    expect(await screen.findByText("No speech detected.")).toBeInTheDocument();
  });

  it("shows failures with a retry button", async () => {
    const failed = makeDetail({ status: "failed", stage: "normalize", error: "Could not decode audio: bad" });
    const client = mockClient({
      getRecording: vi.fn().mockResolvedValueOnce(failed).mockResolvedValue(makeDetail()),
    });
    const user = renderView(client);
    expect(await screen.findByRole("alert")).toHaveTextContent("Could not decode audio: bad");
    await user.click(screen.getByRole("button", { name: "Retry" }));
    expect(client.retryRecording).toHaveBeenCalledWith("rec-1");
  });

  it("seeks the audio when a timestamp is clicked", async () => {
    const user = renderView(mockClient({ getRecording: vi.fn(async () => makeDetail()) }));
    await waitFor(() => expect(document.querySelector("audio")).not.toBeNull());
    await user.click(screen.getByRole("button", { name: "Play from 00:04" }));
    expect(seekedTo).toBe(4);
  });

  it("marks a segment reviewed", async () => {
    const reviewed = makeDetail();
    reviewed.segments[1]!.reviewed = true;
    const client = mockClient({
      getRecording: vi.fn(async () => makeDetail()),
      markReviewed: vi.fn(async () => reviewed),
    });
    const user = renderView(client);
    const group = await screen.findByRole("region", { name: /Overlapping speech/ });
    await user.click(within(group).getAllByRole("button", { name: "Mark reviewed" })[0]!);
    expect(client.markReviewed).toHaveBeenCalledWith("seg-2");
    await waitFor(() => expect(within(group).getAllByText("Needs review")).toHaveLength(1));
  });
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pnpm --filter @saydo/react test`
Expected: FAIL with `Failed to resolve import "./model"` and `"./TranscriptView"`.

- [ ] **Step 3: Implement the model and hook**

`packages/react/src/transcript/model.ts`:
```ts
import type { OverlapRegion, RecordingDetail, Segment } from "../client/client";

export type TranscriptItem =
  | { kind: "segment"; key: string; segment: Segment }
  | { kind: "overlap"; key: string; region: OverlapRegion; columns: { speakerId: string; segments: Segment[] }[] };

export const STAGE_LABELS: Record<string, string> = {
  normalize: "Preparing audio",
  diarize: "Identifying speakers",
  transcribe: "Transcribing",
  merge: "Aligning words to speakers",
  score: "Scoring confidence",
  persist: "Saving",
};

export function needsAttention(segment: Segment): boolean {
  return segment.needs_review && !segment.reviewed;
}

export function reviewCount(detail: Pick<RecordingDetail, "segments">): number {
  return detail.segments.filter(needsAttention).length;
}

export function speakerColor(index: number): string {
  return `var(--saydo-speaker-${((index % 12) + 12) % 12})`;
}

function toColumns(segments: Segment[]): { speakerId: string; segments: Segment[] }[] {
  const columns = new Map<string, Segment[]>();
  for (const segment of segments) {
    const column = columns.get(segment.speaker_id) ?? [];
    column.push(segment);
    columns.set(segment.speaker_id, column);
  }
  return [...columns].map(([speakerId, items]) => ({ speakerId, segments: items }));
}

export function buildTranscriptItems(
  detail: Pick<RecordingDetail, "segments" | "overlap_regions">,
  options: { onlyNeedsReview?: boolean } = {},
): TranscriptItem[] {
  const regions = new Map(detail.overlap_regions.map((r) => [r.id, r]));
  const groups = new Map<string, Segment[]>();
  const items: TranscriptItem[] = [];
  for (const segment of detail.segments) {
    const region = segment.overlap_region_id ? regions.get(segment.overlap_region_id) : undefined;
    if (!region) {
      items.push({ kind: "segment", key: segment.id, segment });
      continue;
    }
    let group = groups.get(region.id);
    if (!group) {
      group = [];
      groups.set(region.id, group);
      items.push({ kind: "overlap", key: region.id, region, columns: [] });
    }
    group.push(segment);
  }
  for (const item of items) {
    if (item.kind === "overlap") item.columns = toColumns(groups.get(item.key) ?? []);
  }
  if (!options.onlyNeedsReview) return items;
  return items.filter((item) =>
    item.kind === "segment"
      ? needsAttention(item.segment)
      : item.columns.some((column) => column.segments.some(needsAttention)),
  );
}
```

`packages/react/src/hooks/useRecording.ts`:
```ts
import { useCallback, useEffect, useState } from "react";
import type { RecordingDetail, StatusOut } from "../client/client";
import { useSaydoClient } from "../context";

function toError(error: unknown): Error {
  return error instanceof Error ? error : new Error(String(error));
}

export interface UseRecordingResult {
  detail: RecordingDetail | null;
  status: StatusOut | null;
  error: Error | null;
  setDetail(detail: RecordingDetail): void;
  refresh(): Promise<void>;
  retry(): Promise<void>;
}

export function useRecording(recordingId: string, { pollMs = 2000 }: { pollMs?: number } = {}): UseRecordingResult {
  const client = useSaydoClient();
  const [detail, setDetailState] = useState<RecordingDetail | null>(null);
  const [status, setStatus] = useState<StatusOut | null>(null);
  const [error, setError] = useState<Error | null>(null);

  const setDetail = useCallback((next: RecordingDetail) => {
    setDetailState(next);
    setStatus({ status: next.recording.status, stage: next.recording.stage, error: next.recording.error });
  }, []);

  const refresh = useCallback(async () => {
    try {
      setDetail(await client.getRecording(recordingId));
      setError(null);
    } catch (e) {
      setError(toError(e));
    }
  }, [client, recordingId, setDetail]);

  useEffect(() => {
    setDetailState(null);
    setStatus(null);
    void refresh();
  }, [refresh]);

  const current = status?.status;
  const settled = current === "ready" || current === "failed";
  useEffect(() => {
    if (!current || settled) return;
    const timer = setInterval(async () => {
      try {
        const next = await client.getStatus(recordingId);
        setStatus(next);
        if (next.status === "ready" || next.status === "failed") await refresh();
      } catch (e) {
        setError(toError(e));
      }
    }, pollMs);
    return () => clearInterval(timer);
  }, [client, recordingId, current, settled, pollMs, refresh]);

  const retry = useCallback(async () => {
    try {
      await client.retryRecording(recordingId);
      await refresh();
    } catch (e) {
      setError(toError(e));
    }
  }, [client, recordingId, refresh]);

  return { detail, status, error, setDetail, refresh, retry };
}
```

- [ ] **Step 4: Implement the components**

`packages/react/src/components/SegmentRow.tsx`:
```tsx
import type { CSSProperties } from "react";
import type { Segment, Speaker } from "../client/client";
import { cx, formatTimestamp } from "../format";
import { needsAttention, speakerColor } from "../transcript/model";

export interface TranscriptActions {
  renameSpeaker(speakerId: string, name: string | null): Promise<void>;
  reassign(segmentId: string, speakerId: string): Promise<void>;
  reassignToNew(segmentId: string, name?: string): Promise<void>;
  merge(speakerId: string, intoSpeakerId: string): Promise<void>;
  markReviewed(segmentId: string): Promise<void>;
}

export interface SegmentRowProps {
  segment: Segment;
  speakers: Speaker[];
  actions: TranscriptActions;
  onSeek(seconds: number): void;
}

function confidenceTitle(segment: Segment): string {
  const p = segment.confidence_parts;
  const f = (value: number | undefined) => (value ?? 0).toFixed(2);
  return `Confidence ${segment.confidence.toFixed(2)}: speech recognition ${f(p.asr)}, voice match ${f(p.margin)}, length ${f(p.length)}, penalty ×${f(p.penalty)}`;
}

export function SegmentRow({ segment, speakers, actions, onSeek }: SegmentRowProps) {
  const speaker = speakers.find((s) => s.id === segment.speaker_id);
  const placeholder = segment.origin === "placeholder";
  const attention = needsAttention(segment);
  const color = { "--saydo-speaker-color": speakerColor(speaker?.color_index ?? 0) } as CSSProperties;
  return (
    <div
      className={cx("saydo-segment", attention && "saydo-segment-review", placeholder && "saydo-segment-placeholder")}
      data-segment-id={segment.id}
    >
      <button
        type="button"
        className="saydo-ts"
        aria-label={`Play from ${formatTimestamp(segment.start_s)}`}
        onClick={() => onSeek(segment.start_s)}
      >
        {formatTimestamp(segment.start_s)}
      </button>
      <span className="saydo-speaker" style={color}>
        {speaker?.name ?? "Unknown speaker"}
      </span>
      <p className="saydo-text">{placeholder ? <em>[overlapping speech – not transcribed]</em> : segment.text}</p>
      <span className="saydo-badges">
        {attention && (
          <span className="saydo-badge saydo-badge-review" title={confidenceTitle(segment)}>
            Needs review
          </span>
        )}
        {segment.user_corrected && <span className="saydo-badge">Corrected</span>}
        {segment.reviewed && !segment.user_corrected && segment.needs_review && (
          <span className="saydo-badge">Reviewed</span>
        )}
        {attention && (
          <button type="button" className="saydo-link" onClick={() => void actions.markReviewed(segment.id)}>
            Mark reviewed
          </button>
        )}
      </span>
    </div>
  );
}
```

`packages/react/src/components/OverlapGroup.tsx`:
```tsx
import type { OverlapRegion, Segment, Speaker } from "../client/client";
import { formatTimestamp } from "../format";
import { SegmentRow, type TranscriptActions } from "./SegmentRow";

export interface OverlapGroupProps {
  region: OverlapRegion;
  columns: { speakerId: string; segments: Segment[] }[];
  speakers: Speaker[];
  actions: TranscriptActions;
  onSeek(seconds: number): void;
}

export function OverlapGroup({ region, columns, speakers, actions, onSeek }: OverlapGroupProps) {
  const label = `Overlapping speech · ${formatTimestamp(region.start_s)}–${formatTimestamp(region.end_s)}`;
  return (
    <section className="saydo-overlap" aria-label={label}>
      <h3 className="saydo-overlap-title">{label}</h3>
      <div className="saydo-overlap-columns">
        {columns.map((column) => (
          <div key={column.speakerId} className="saydo-overlap-column">
            {column.segments.map((segment) => (
              <SegmentRow key={segment.id} segment={segment} speakers={speakers} actions={actions} onSeek={onSeek} />
            ))}
          </div>
        ))}
      </div>
    </section>
  );
}
```

`packages/react/src/components/TranscriptView.tsx`:
```tsx
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { RecordingDetail } from "../client/client";
import { useSaydoClient } from "../context";
import { cx } from "../format";
import { useRecording } from "../hooks/useRecording";
import { STAGE_LABELS, buildTranscriptItems, reviewCount } from "../transcript/model";
import { OverlapGroup } from "./OverlapGroup";
import { SegmentRow, type TranscriptActions } from "./SegmentRow";

export interface TranscriptViewProps {
  recordingId: string;
  pollMs?: number;
  className?: string;
}

export function TranscriptView({ recordingId, pollMs, className }: TranscriptViewProps) {
  const client = useSaydoClient();
  const { detail, status, error, setDetail, retry } = useRecording(recordingId, { pollMs });
  const [onlyReview, setOnlyReview] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement>(null);

  const playable = detail?.recording.status === "ready" && detail.recording.playback_available;
  useEffect(() => {
    if (!playable) return;
    let cancelled = false;
    client
      .getAudioUrl(recordingId)
      .then((url) => {
        if (!cancelled) setAudioUrl(url);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [client, recordingId, playable]);

  const seek = useCallback((seconds: number) => {
    const audio = audioRef.current;
    if (!audio) return;
    audio.currentTime = seconds;
    void audio.play().catch(() => {});
  }, []);

  const run = useCallback(
    async (operation: () => Promise<RecordingDetail>) => {
      setActionError(null);
      try {
        setDetail(await operation());
      } catch (e) {
        setActionError(e instanceof Error ? e.message : "The change could not be saved");
      }
    },
    [setDetail],
  );

  const actions = useMemo<TranscriptActions>(
    () => ({
      renameSpeaker: (speakerId, name) => run(() => client.renameSpeaker(speakerId, name)),
      reassign: (segmentId, speakerId) => run(() => client.reassignSegment(segmentId, speakerId)),
      reassignToNew: (segmentId, name) =>
        run(async () => {
          const created = await client.createSpeaker(recordingId, name);
          return client.reassignSegment(segmentId, created.speaker.id);
        }),
      merge: (speakerId, intoSpeakerId) => run(() => client.mergeSpeaker(speakerId, intoSpeakerId)),
      markReviewed: (segmentId) => run(() => client.markReviewed(segmentId)),
    }),
    [client, recordingId, run],
  );

  if (error && !detail) {
    return (
      <p role="alert" className="saydo-error">
        {error.message}
      </p>
    );
  }
  if (!detail || !status) return <p className="saydo-muted">Loading…</p>;

  if (status.status === "failed") {
    const stage = detail.recording.stage;
    return (
      <div role="alert" className={cx("saydo-error", className)}>
        <p>
          Processing failed{stage ? ` (${STAGE_LABELS[stage] ?? stage})` : ""}: {status.error ?? detail.recording.error}
        </p>
        <button type="button" className="saydo-btn" onClick={() => void retry()}>
          Retry
        </button>
      </div>
    );
  }
  if (status.status !== "ready") {
    const label = status.status === "processing" && status.stage ? STAGE_LABELS[status.stage] : undefined;
    return (
      <p role="status" className="saydo-progress">
        {status.status === "queued" ? "Waiting to start…" : `${label ?? "Processing"}…`}
      </p>
    );
  }

  const items = buildTranscriptItems(detail, { onlyNeedsReview: onlyReview });
  const pending = reviewCount(detail);
  return (
    <section className={cx("saydo-transcript", className)} aria-label="Transcript">
      <header className="saydo-transcript-header">
        <h2>{detail.recording.title}</h2>
        {audioUrl && <audio ref={audioRef} src={audioUrl} controls preload="metadata" />}
        <label>
          <input type="checkbox" checked={onlyReview} onChange={(e) => setOnlyReview(e.target.checked)} /> Show only
          items needing review ({pending})
        </label>
        <p className="saydo-muted">
          Confidence is a heuristic score for ranking what to check, not a probability.
        </p>
      </header>
      {actionError && (
        <p role="alert" className="saydo-error">
          {actionError}
        </p>
      )}
      {detail.segments.length === 0 ? (
        <p className="saydo-muted">No speech detected.</p>
      ) : items.length === 0 ? (
        <p className="saydo-muted">Nothing needs review.</p>
      ) : (
        <ol className="saydo-items">
          {items.map((item) => (
            <li key={item.key}>
              {item.kind === "segment" ? (
                <SegmentRow segment={item.segment} speakers={detail.speakers} actions={actions} onSeek={seek} />
              ) : (
                <OverlapGroup
                  region={item.region}
                  columns={item.columns}
                  speakers={detail.speakers}
                  actions={actions}
                  onSeek={seek}
                />
              )}
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}
```

Add to `packages/react/src/index.ts`:
```ts
export { TranscriptView, type TranscriptViewProps } from "./components/TranscriptView";
export { type TranscriptActions } from "./components/SegmentRow";
export { type UseRecordingResult, useRecording } from "./hooks/useRecording";
export { type TranscriptItem, buildTranscriptItems, reviewCount } from "./transcript/model";
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pnpm --filter @saydo/react test && pnpm --filter @saydo/react typecheck`
Expected: all pass. In the checkbox test, the label text spans a line break in the JSX, but React renders it as one text run: "Show only items needing review (2)".

- [ ] **Step 6: Commit**

```bash
git add packages/react
git commit -m "feat(react): TranscriptView with overlap groups, review filter, playback and retry"
```

---

### Task 18: Speaker chip menu: rename, reassign, new speaker, merge (M4)

**Files:**
- Create: `packages/react/src/components/SpeakerChip.tsx`
- Modify: `packages/react/src/components/SegmentRow.tsx` (swap the static label for the chip)
- Test: `packages/react/src/components/SpeakerChip.test.tsx`

**Interfaces:**
- Consumes: `TranscriptActions`, `SegmentRow` (Task 17); the client types (Task 14).
- Produces:
  - `<SpeakerChip speaker segment speakers actions />`.
  - Accessible names:
    - the chip is a button named after the speaker;
    - menu items: `Rename <name>…`, one item per other speaker (labelled with that speaker's name), `New speaker…` and `Merge into <other name>`;
    - form fields: `Speaker name` (rename) and `New speaker name`;
    - buttons: `Save` and `Cancel`.

- [ ] **Step 1: Write the failing tests**

`packages/react/src/components/SpeakerChip.test.tsx`:
```tsx
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { SaydoClient } from "../client/client";
import { SaydoProvider } from "../context";
import { mockClient } from "../test/clientMock";
import { SPEAKER_1, SPEAKER_2, makeDetail } from "../test/fixtures";
import { TranscriptView } from "./TranscriptView";

function renderView(client: SaydoClient) {
  render(
    <SaydoProvider client={client}>
      <TranscriptView recordingId="rec-1" />
    </SaydoProvider>,
  );
  return userEvent.setup();
}

async function openFirstChip(user: ReturnType<typeof userEvent.setup>, name = "Speaker 1") {
  const chips = await screen.findAllByRole("button", { name });
  await user.click(chips[0]!);
}

describe("SpeakerChip", () => {
  it("renames a speaker everywhere", async () => {
    const renamed = makeDetail();
    renamed.speakers[0] = { ...renamed.speakers[0]!, display_name: "Priya", name: "Priya" };
    const client = mockClient({
      getRecording: vi.fn(async () => makeDetail()),
      renameSpeaker: vi.fn(async () => renamed),
    });
    const user = renderView(client);
    await openFirstChip(user);
    await user.click(screen.getByRole("menuitem", { name: "Rename Speaker 1…" }));
    await user.type(screen.getByLabelText("Speaker name"), "Priya");
    await user.click(screen.getByRole("button", { name: "Save" }));
    expect(client.renameSpeaker).toHaveBeenCalledWith(SPEAKER_1, "Priya");
    await waitFor(() => expect(screen.getAllByRole("button", { name: "Priya" })).toHaveLength(2));
    expect(screen.queryByRole("button", { name: "Speaker 1" })).toBeNull();
  });

  it("reassigns a segment to another speaker", async () => {
    const client = mockClient({
      getRecording: vi.fn(async () => makeDetail()),
      reassignSegment: vi.fn(async () => makeDetail()),
    });
    const user = renderView(client);
    await openFirstChip(user);
    await user.click(screen.getByRole("menuitem", { name: "Speaker 2" }));
    expect(client.reassignSegment).toHaveBeenCalledWith("seg-1", SPEAKER_2);
  });

  it("reassigns a segment to a new speaker", async () => {
    const detail = makeDetail();
    const client = mockClient({
      getRecording: vi.fn(async () => detail),
      createSpeaker: vi.fn(async () => ({
        speaker: { id: "spk-3", label: "Speaker 3", display_name: "Sam", name: "Sam", color_index: 2 },
        detail,
      })),
      reassignSegment: vi.fn(async () => detail),
    });
    const user = renderView(client);
    await openFirstChip(user);
    await user.click(screen.getByRole("menuitem", { name: "New speaker…" }));
    await user.type(screen.getByLabelText("New speaker name"), "Sam");
    await user.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() => expect(client.reassignSegment).toHaveBeenCalledWith("seg-1", "spk-3"));
    expect(client.createSpeaker).toHaveBeenCalledWith("rec-1", "Sam");
  });

  it("merges a speaker into another", async () => {
    const client = mockClient({
      getRecording: vi.fn(async () => makeDetail()),
      mergeSpeaker: vi.fn(async () => makeDetail()),
    });
    const user = renderView(client);
    await openFirstChip(user);
    await user.click(screen.getByRole("menuitem", { name: "Merge into Speaker 2" }));
    expect(client.mergeSpeaker).toHaveBeenCalledWith(SPEAKER_1, SPEAKER_2);
  });

  it("closes the menu on Escape", async () => {
    const user = renderView(mockClient({ getRecording: vi.fn(async () => makeDetail()) }));
    await openFirstChip(user);
    expect(screen.getByRole("menu")).toBeInTheDocument();
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("menu")).toBeNull();
  });

  it("shows an error when a correction fails", async () => {
    const client = mockClient({
      getRecording: vi.fn(async () => makeDetail()),
      mergeSpeaker: vi.fn(async () => {
        throw new Error("Edits are possible once processing has finished");
      }),
    });
    const user = renderView(client);
    await openFirstChip(user);
    await user.click(screen.getByRole("menuitem", { name: "Merge into Speaker 2" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Edits are possible once processing has finished");
  });
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pnpm --filter @saydo/react test SpeakerChip`
Expected: FAIL. `findAllByRole("button", { name: "Speaker 1" })` finds nothing, because the speaker label is still a `<span>`.

- [ ] **Step 3: Implement**

`packages/react/src/components/SpeakerChip.tsx`:
```tsx
import { type CSSProperties, type FormEvent, useEffect, useId, useRef, useState } from "react";
import type { Segment, Speaker } from "../client/client";
import { speakerColor } from "../transcript/model";
import type { TranscriptActions } from "./SegmentRow";

type Mode = "closed" | "menu" | "rename" | "new";

export interface SpeakerChipProps {
  speaker: Speaker | undefined;
  segment: Segment;
  speakers: Speaker[];
  actions: TranscriptActions;
}

export function SpeakerChip({ speaker, segment, speakers, actions }: SpeakerChipProps) {
  const [mode, setMode] = useState<Mode>("closed");
  const [name, setName] = useState("");
  const rootRef = useRef<HTMLSpanElement>(null);
  const menuId = useId();
  const close = () => setMode("closed");

  useEffect(() => {
    if (mode === "closed") return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setMode("closed");
    };
    const onPointer = (event: MouseEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) setMode("closed");
    };
    document.addEventListener("keydown", onKey);
    document.addEventListener("mousedown", onPointer);
    return () => {
      document.removeEventListener("keydown", onKey);
      document.removeEventListener("mousedown", onPointer);
    };
  }, [mode]);

  const others = speakers.filter((s) => s.id !== segment.speaker_id);
  const style = { "--saydo-speaker-color": speakerColor(speaker?.color_index ?? 0) } as CSSProperties;

  function submit(event: FormEvent) {
    event.preventDefault();
    const value = name.trim();
    close();
    if (mode === "rename" && speaker) void actions.renameSpeaker(speaker.id, value || null);
    if (mode === "new") void actions.reassignToNew(segment.id, value || undefined);
  }

  return (
    <span className="saydo-chip-root" ref={rootRef}>
      <button
        type="button"
        className="saydo-speaker"
        style={style}
        aria-haspopup="menu"
        aria-expanded={mode !== "closed"}
        aria-controls={menuId}
        onClick={() => setMode(mode === "closed" ? "menu" : "closed")}
      >
        {speaker?.name ?? "Unknown speaker"}
      </button>

      {mode === "menu" && (
        <div role="menu" id={menuId} className="saydo-menu" aria-label={`Actions for ${speaker?.name ?? "speaker"}`}>
          {speaker && (
            <button
              type="button"
              role="menuitem"
              onClick={() => {
                setName(speaker.display_name ?? "");
                setMode("rename");
              }}
            >
              Rename {speaker.name}…
            </button>
          )}
          {others.length > 0 && <p className="saydo-menu-heading">Assign this segment to</p>}
          {others.map((other) => (
            <button
              key={other.id}
              type="button"
              role="menuitem"
              onClick={() => {
                close();
                void actions.reassign(segment.id, other.id);
              }}
            >
              {other.name}
            </button>
          ))}
          <button
            type="button"
            role="menuitem"
            onClick={() => {
              setName("");
              setMode("new");
            }}
          >
            New speaker…
          </button>
          {speaker && others.length > 0 && <p className="saydo-menu-heading">Same person as…</p>}
          {speaker &&
            others.map((other) => (
              <button
                key={`merge-${other.id}`}
                type="button"
                role="menuitem"
                onClick={() => {
                  close();
                  void actions.merge(speaker.id, other.id);
                }}
              >
                {`Merge into ${other.name}`}
              </button>
            ))}
        </div>
      )}

      {(mode === "rename" || mode === "new") && (
        <form className="saydo-menu" onSubmit={submit}>
          <label>
            {mode === "rename" ? "Speaker name" : "New speaker name"}{" "}
            {/* biome-ignore lint/a11y/noAutofocus: focus moves into the form the user just opened */}
            <input autoFocus value={name} maxLength={80} onChange={(e) => setName(e.target.value)} />
          </label>
          <span>
            <button type="submit" className="saydo-btn saydo-btn-primary">
              Save
            </button>{" "}
            <button type="button" className="saydo-btn" onClick={close}>
              Cancel
            </button>
          </span>
        </form>
      )}
    </span>
  );
}
```

In `packages/react/src/components/SegmentRow.tsx`:
- Replace the import line `import type { CSSProperties } from "react";` with nothing (delete it).
- Delete the line `const color = { "--saydo-speaker-color": speakerColor(speaker?.color_index ?? 0) } as CSSProperties;`.
- Change `import { needsAttention, speakerColor } from "../transcript/model";` to `import { needsAttention } from "../transcript/model";`.
- Add `import { SpeakerChip } from "./SpeakerChip";`.
- Replace this block:
```tsx
      <span className="saydo-speaker" style={color}>
        {speaker?.name ?? "Unknown speaker"}
      </span>
```
with:
```tsx
      <SpeakerChip speaker={speaker} segment={segment} speakers={speakers} actions={actions} />
```

`SpeakerChip.tsx` imports `TranscriptActions` from `SegmentRow.tsx`, and `SegmentRow.tsx` imports `SpeakerChip`. This import cycle is type-only on the `SpeakerChip` side (`import type`), so it is erased at runtime and safe.

- [ ] **Step 4: Run all package tests**

Run: `pnpm --filter @saydo/react test && pnpm --filter @saydo/react typecheck && pnpm --filter @saydo/react build`
Expected: all pass, including the Task 17 tests. Those tests don't query the speaker name, so the chip swap doesn't affect them.

- [ ] **Step 5: Commit**

```bash
git add packages/react
git commit -m "feat(react): speaker chip menu for rename, reassign, new speaker and merge"
```

---

### Task 19: Next.js reference app, Playwright end-to-end test, web Docker image, CI web jobs (M1–M4)

**Files:**
- Create: `web/package.json`, `web/tsconfig.json`, `web/next.config.ts`, `web/next-env.d.ts`
- Create: `web/app/layout.tsx`, `web/app/providers.tsx`, `web/app/globals.css`, `web/app/page.tsx`, `web/app/recording-list.tsx`, `web/app/recordings/[id]/page.tsx`, `web/app/recordings/[id]/recording-screen.tsx`
- Create: `web/playwright.config.ts`, `web/e2e/record-and-correct.spec.ts`
- Create: `docker/web.Dockerfile`
- Modify: `docker-compose.yml` (add the `web` service), `.github/workflows/ci.yml` (add the `web` and `e2e` jobs)

**Interfaces:**
- Consumes:
  - from `@saydo/react`: `SaydoProvider`, `MeetingRecorder`, `TranscriptView`, `useSaydoClient` and `formatTimestamp`;
  - the API started with fake plugins.
- Produces:
  - the app on port 3000;
  - the environment variable `SAYDO_PUBLIC_API_URL`, read at request time;
  - the routes `/` and `/recordings/[id]`;
  - a `Delete recording` button.

- [ ] **Step 1: Create the app**

`web/package.json`:
```json
{
  "name": "saydo-web",
  "private": true,
  "scripts": {
    "dev": "next dev --port 3000",
    "build": "next build",
    "start": "next start --port 3000",
    "typecheck": "tsc --noEmit",
    "e2e": "playwright test"
  },
  "dependencies": {
    "@saydo/react": "workspace:*",
    "next": "^16.4.0",
    "react": "^19.3.0",
    "react-dom": "^19.3.0"
  },
  "devDependencies": {
    "@playwright/test": "^1.64.0",
    "@types/node": "^24.0.0",
    "@types/react": "^19.3.0",
    "@types/react-dom": "^19.3.0",
    "typescript": "^5.9.3"
  }
}
```

`web/tsconfig.json`:
```json
{
  "extends": "../tsconfig.base.json",
  "compilerOptions": {
    "noEmit": true,
    "allowJs": false,
    "incremental": true,
    "plugins": [{ "name": "next" }]
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx", ".next/types/**/*.ts"],
  "exclude": ["node_modules"]
}
```

`web/next-env.d.ts`:
```ts
/// <reference types="next" />
/// <reference types="next/image-types/global" />
```

`web/next.config.ts`:
```ts
import path from "node:path";
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  outputFileTracingRoot: path.resolve(process.cwd(), ".."),
};

export default nextConfig;
```

`web/app/layout.tsx`:
```tsx
import type { Metadata } from "next";
import Link from "next/link";
import type { ReactNode } from "react";
import "@saydo/react/styles.css";
import "./globals.css";
import { Providers } from "./providers";

export const metadata: Metadata = {
  title: "SayDo",
  description: "Meeting transcripts with honest overlap handling",
};

// Read SAYDO_PUBLIC_API_URL per request so one image works in any deployment.
export const dynamic = "force-dynamic";

export default function RootLayout({ children }: { children: ReactNode }) {
  const apiBaseUrl = process.env.SAYDO_PUBLIC_API_URL ?? "http://localhost:8000";
  return (
    <html lang="en">
      <body>
        <header className="app-header">
          <Link href="/">SayDo</Link>
        </header>
        <Providers apiBaseUrl={apiBaseUrl}>{children}</Providers>
      </body>
    </html>
  );
}
```

`web/app/providers.tsx`:
```tsx
"use client";

import { SaydoProvider } from "@saydo/react";
import type { ReactNode } from "react";

export function Providers({ apiBaseUrl, children }: { apiBaseUrl: string; children: ReactNode }) {
  return <SaydoProvider apiBaseUrl={apiBaseUrl}>{children}</SaydoProvider>;
}
```

`web/app/globals.css`:
```css
body { margin: 0; font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
.app-header { padding: 12px 24px; border-bottom: 1px solid #d1d9e0; font-weight: 700; }
.app-header a { color: inherit; text-decoration: none; }
.page { max-width: 960px; margin: 0 auto; padding: 24px; display: grid; gap: 24px; }
.recording-list { list-style: none; padding: 0; display: grid; gap: 8px; }
.recording-list a { font-weight: 600; }
.danger { color: #cf222e; }
```

`web/app/page.tsx`:
```tsx
"use client";

import { MeetingRecorder } from "@saydo/react";
import { useRouter } from "next/navigation";
import { RecordingList } from "./recording-list";

export default function HomePage() {
  const router = useRouter();
  return (
    <main className="page">
      <section>
        <h1>New meeting</h1>
        <MeetingRecorder onUploaded={(id) => router.push(`/recordings/${id}`)} />
      </section>
      <section>
        <h2>Recent meetings</h2>
        <RecordingList />
      </section>
    </main>
  );
}
```

`web/app/recording-list.tsx`:
```tsx
"use client";

import { type RecordingSummary, formatTimestamp, useSaydoClient } from "@saydo/react";
import Link from "next/link";
import { useEffect, useState } from "react";

export function RecordingList() {
  const client = useSaydoClient();
  const [recordings, setRecordings] = useState<RecordingSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    client
      .listRecordings()
      .then(setRecordings)
      .catch((e: Error) => setError(e.message));
  }, [client]);

  if (error) return <p role="alert">{error}</p>;
  if (!recordings) return <p>Loading…</p>;
  if (recordings.length === 0) return <p>No meetings yet.</p>;
  return (
    <ul className="recording-list">
      {recordings.map((r) => (
        <li key={r.id}>
          <Link href={`/recordings/${r.id}`}>{r.title}</Link> · {new Date(r.created_at).toLocaleString()} ·{" "}
          {r.duration_s ? formatTimestamp(r.duration_s) : "—"} · {r.status}
        </li>
      ))}
    </ul>
  );
}
```

`web/app/recordings/[id]/page.tsx`:
```tsx
import { RecordingScreen } from "./recording-screen";

export default async function RecordingPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return <RecordingScreen id={id} />;
}
```

`web/app/recordings/[id]/recording-screen.tsx`:
```tsx
"use client";

import { TranscriptView, useSaydoClient } from "@saydo/react";
import { useRouter } from "next/navigation";
import { useState } from "react";

export function RecordingScreen({ id }: { id: string }) {
  const client = useSaydoClient();
  const router = useRouter();
  const [error, setError] = useState<string | null>(null);

  async function remove() {
    if (!window.confirm("Delete this recording, its audio and its transcript permanently?")) return;
    try {
      await client.deleteRecording(id);
      router.push("/");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Delete failed");
    }
  }

  return (
    <main className="page">
      <TranscriptView recordingId={id} />
      <p>
        <button type="button" className="danger" onClick={() => void remove()}>
          Delete recording
        </button>
      </p>
      {error && <p role="alert">{error}</p>}
    </main>
  );
}
```

- [ ] **Step 2: Build and run it manually against the fake-plugin API**

```bash
pnpm install
pnpm --filter @saydo/react build
pnpm --filter saydo-web typecheck && pnpm --filter saydo-web build
```
Expected: the build succeeds. Next may rewrite `web/tsconfig.json` on its first build, for example to adjust `jsx`. Keep its changes.

- [ ] **Step 3: Write the end-to-end test**

`web/playwright.config.ts`:
```ts
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { defineConfig, devices } from "@playwright/test";

const API_PORT = 8010;
const WEB_PORT = 3010;
const dataDir = mkdtempSync(path.join(tmpdir(), "saydo-e2e-"));

export default defineConfig({
  testDir: "./e2e",
  timeout: 90_000,
  retries: process.env.CI ? 1 : 0,
  use: {
    baseURL: `http://localhost:${WEB_PORT}`,
    permissions: ["microphone"],
    launchOptions: {
      args: ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"],
    },
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: [
    {
      command: "uv run saydo-api",
      cwd: "../api",
      port: API_PORT,
      timeout: 120_000,
      reuseExistingServer: false,
      env: {
        SAYDO_PORT: String(API_PORT),
        SAYDO_CONFIG: path.join(dataDir, "none.yaml"),
        SAYDO__DATA_DIR: dataDir,
        SAYDO__API__CORS_ORIGINS: JSON.stringify([`http://localhost:${WEB_PORT}`]),
        SAYDO__PIPELINE__DEVICE: "cpu",
        SAYDO__PIPELINE__TRANSCRIBER__NAME: "fake",
        SAYDO__PIPELINE__DIARIZER__NAME: "fake",
        SAYDO__PIPELINE__EMBEDDER__NAME: "fake",
      },
    },
    {
      command: `pnpm build && pnpm exec next start --port ${WEB_PORT}`,
      port: WEB_PORT,
      timeout: 240_000,
      reuseExistingServer: false,
      env: { SAYDO_PUBLIC_API_URL: `http://localhost:${API_PORT}` },
    },
  ],
});
```

`web/e2e/record-and-correct.spec.ts`:
```ts
import { expect, test } from "@playwright/test";

test("record, transcribe, see overlaps, rename a speaker and delete", async ({ page }) => {
  await page.goto("/");
  await page.getByLabel("Title").fill("E2E sprint planning");
  await page.getByRole("button", { name: "Record" }).click();
  await expect(page.getByRole("button", { name: "Stop" })).toBeVisible();
  await page.waitForTimeout(7_500); // fake diarizer overlaps speakers at 3-4 s and 6-7 s
  await page.getByRole("button", { name: "Stop" }).click();
  await page.getByRole("button", { name: "Upload & transcribe" }).click();

  await expect(page).toHaveURL(/\/recordings\/[0-9a-f-]{36}$/);
  await expect(page.getByRole("heading", { name: "E2E sprint planning" })).toBeVisible({ timeout: 30_000 });
  await expect(page.getByRole("region", { name: /Overlapping speech/ }).first()).toBeVisible();
  await expect(page.getByText("[overlapping speech – not transcribed]").first()).toBeVisible();
  await expect(page.getByText("Needs review").first()).toBeVisible();

  await page.getByRole("button", { name: "Speaker 1", exact: true }).first().click();
  await page.getByRole("menuitem", { name: "Rename Speaker 1…" }).click();
  await page.getByLabel("Speaker name").fill("Priya");
  await page.getByRole("button", { name: "Save" }).click();
  await expect(page.getByRole("button", { name: "Speaker 1", exact: true })).toHaveCount(0);
  expect(await page.getByRole("button", { name: "Priya", exact: true }).count()).toBeGreaterThan(1);

  await page.reload();
  await expect(page.getByRole("button", { name: "Priya", exact: true }).first()).toBeVisible();

  page.once("dialog", (dialog) => void dialog.accept());
  await page.getByRole("button", { name: "Delete recording" }).click();
  await expect(page).toHaveURL(/\/$/);
  await expect(page.getByText("E2E sprint planning")).toHaveCount(0);
});
```

- [ ] **Step 4: Run the end-to-end test**

```bash
cd api && uv sync && cd ..
pnpm --filter saydo-web exec playwright install chromium
pnpm --filter saydo-web e2e
```
Expected: `1 passed`. If the overlap region isn't found, Chrome's fake microphone produced audio below the fake plugins' silence threshold. Check with `curl localhost:8010/api/v1/recordings` while the test is paused (`PWDEBUG=1`) before changing anything.

- [ ] **Step 5: Add the web image, compose service and CI jobs**

`docker/web.Dockerfile`:
```dockerfile
# syntax=docker/dockerfile:1
FROM node:24-slim AS build
RUN corepack enable
WORKDIR /repo
COPY package.json pnpm-workspace.yaml pnpm-lock.yaml tsconfig.base.json ./
COPY packages/react/package.json packages/react/
COPY web/package.json web/
RUN pnpm install --frozen-lockfile
COPY packages/react packages/react
COPY web web
RUN pnpm --filter @saydo/react build && pnpm --filter saydo-web build

FROM node:24-slim
WORKDIR /app
ENV NODE_ENV=production PORT=3000 HOSTNAME=0.0.0.0
COPY --from=build /repo/web/.next/standalone ./
COPY --from=build /repo/web/.next/static ./web/.next/static
USER node
EXPOSE 3000
CMD ["node", "web/server.js"]
```

Add to `docker-compose.yml` under `services:` (after `api`):
```yaml
  web:
    build:
      context: .
      dockerfile: docker/web.Dockerfile
    image: saydo-web
    environment:
      SAYDO_PUBLIC_API_URL: http://localhost:8000
    ports:
      - "127.0.0.1:3000:3000"
    depends_on:
      - api
```

Append these jobs to `.github/workflows/ci.yml` under `jobs:`:
```yaml
  web:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: pnpm/action-setup@v4
      - uses: actions/setup-node@v4
        with: { node-version: 24, cache: pnpm }
      - run: pnpm install --frozen-lockfile
      - run: pnpm --filter @saydo/react build
      - run: pnpm typecheck
      - run: pnpm test
      - run: pnpm --filter saydo-web build

  e2e:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: sudo apt-get update && sudo apt-get install -y ffmpeg
      - uses: astral-sh/setup-uv@v6
      - uses: pnpm/action-setup@v4
      - uses: actions/setup-node@v4
        with: { node-version: 24, cache: pnpm }
      - run: pnpm install --frozen-lockfile
      - run: cd api && uv sync --locked
      - name: Generated client is up to date
        run: pnpm gen:client && git diff --exit-code packages/react/openapi.json packages/react/src/client/schema.d.ts
      - run: pnpm --filter @saydo/react build
      - run: pnpm --filter saydo-web exec playwright install --with-deps chromium
      - run: pnpm --filter saydo-web e2e
```

- [ ] **Step 6: Verify the full stack in Docker (demo mode)**

Uncomment the three demo-mode lines in `docker-compose.yml` locally, without committing that change, then run:
```bash
docker compose up --build -d && sleep 15
curl -s localhost:8000/api/v1/health && curl -s -o /dev/null -w "%{http_code}\n" localhost:3000/
docker compose down
git checkout docker-compose.yml
```
Expected: `{"status":"ok","version":"0.1.0"}` and `200`.

- [ ] **Step 7: Commit**

```bash
git add web docker/web.Dockerfile docker-compose.yml .github/workflows/ci.yml pnpm-lock.yaml
git commit -m "feat(web): Next.js reference app, Playwright e2e, web image and CI jobs"
```

---

### Task 20: Evaluation harness: DER, JER, overlap detection, cpWER, error-catch rate (M3)

**Files:**
- Create: `api/src/saydo/evaluation/__init__.py` (empty), `api/src/saydo/evaluation/rttm.py`, `api/src/saydo/evaluation/manifest.py`, `api/src/saydo/evaluation/metrics.py`, `api/src/saydo/evaluation/runner.py`
- Create: `eval/run_eval.py`, `eval/prepare_ami.py`, `eval/README.md`, `eval/reports/.gitkeep`
- Modify: `.github/workflows/ci.yml` (the api job installs the eval extra)
- Test: `api/tests/unit/test_evaluation.py`

**Interfaces:**
- Consumes: `Turn` (Task 5); `find_overlap_regions` (Task 6); `analyze`, `Plugins` (Task 9); `normalize` (Task 3); `load_audio` (Task 5); `Settings`, `ReviewSettings` (Task 1).
- Produces:
  - `rttm.py`:
    - `read_rttm(path) -> dict[str, list[Turn]]`
    - `write_rttm(path, uri, turns) -> None`
    - `read_uem(path) -> dict[str, tuple[float, float]]`
  - `manifest.py`:
    - `EvalItem(uri, audio, rttm, uem=None, transcript=None, min_speakers=None, max_speakers=None)`
    - `load_manifest(path) -> list[EvalItem]` (JSONL; paths are relative to the manifest)
    - `load_transcript(path) -> dict[str, str]`, built from a JSON list of `{speaker, start_s, end_s, text}`
  - `metrics.py`:
    - `to_annotation(turns, uri="item")`
    - `diarization_scores(reference, hypothesis, *, uem=None) -> {"der", "missed", "false_alarm", "confusion", "jer"}`
    - `der_in_regions(reference, hypothesis, regions) -> float | None`
    - `overlap_detection(reference_regions, hypothesis_regions, *, duration_s, frame_s=0.01) -> {"precision", "recall", "f1"}`
    - `speaker_mapping(reference, hypothesis) -> dict[str, str]`, mapping hypothesis labels to reference labels
    - `ScoredSegment(speaker_key, start_s, end_s, confidence, origin)`
    - `segment_is_wrong(segment, reference, mapping) -> bool`
    - `error_catch_counts(reference, segments, mapping, thresholds=DEFAULT_THRESHOLDS) -> list[dict]`
    - `catch_rates(rows) -> list[dict]`
    - `normalize_words(text) -> list[str]`
    - `cpwer(reference: dict[str, str], hypothesis: dict[str, str]) -> {"cpwer", "errors", "ref_words"}`
  - `runner.py`:
    - `ItemResult`
    - `overlap_ratio(reference) -> float`
    - `evaluate_item(item, plugins, review, workdir) -> ItemResult`
    - `format_report(results, *, settings, device, created) -> str`
- Definitions:
  - **Wrong segment:** a segment is *wrong* if the reference speaker mapped from its hypothesis speaker covers less than 30 % of the segment's span.
  - **Flagged segment:** a segment is *flagged* at threshold `t` if it is a placeholder or its confidence is below `t`.
  - **M3 target:** catch rate ≥ 70 % at a flag rate ≤ 25 %.

- [ ] **Step 1: Write the failing tests**

`api/tests/unit/test_evaluation.py`:
```python
import json
import math

import pytest

pytest.importorskip("pyannote.metrics", reason="install with: uv sync --extra eval")
pytest.importorskip("rapidfuzz", reason="install with: uv sync --extra eval")

from saydo.evaluation.manifest import load_manifest, load_transcript  # noqa: E402
from saydo.evaluation.metrics import (  # noqa: E402
    ScoredSegment,
    catch_rates,
    cpwer,
    der_in_regions,
    diarization_scores,
    error_catch_counts,
    overlap_detection,
    speaker_mapping,
)
from saydo.evaluation.rttm import read_rttm, read_uem, write_rttm  # noqa: E402
from saydo.evaluation.runner import ItemResult, format_report, overlap_ratio  # noqa: E402
from saydo.pipeline.types import Turn  # noqa: E402

REF = [Turn("A", 0, 4), Turn("B", 3, 7), Turn("A", 7, 10)]


def test_rttm_and_uem_roundtrip(tmp_path):
    path = tmp_path / "x.rttm"
    write_rttm(path, "meet", REF)
    assert read_rttm(path) == {"meet": REF}
    (tmp_path / "x.uem").write_text("meet 1 0.000 9.500\n")
    assert read_uem(tmp_path / "x.uem") == {"meet": (0.0, 9.5)}


def test_manifest_paths_are_relative_to_manifest(tmp_path):
    (tmp_path / "m.jsonl").write_text(json.dumps({"uri": "a", "audio": "audio/a.wav", "rttm": "a.rttm"}) + "\n\n")
    (item,) = load_manifest(tmp_path / "m.jsonl")
    assert item.audio == (tmp_path / "audio/a.wav").resolve()
    assert item.uem is None


def test_manifest_requires_audio_and_rttm(tmp_path):
    (tmp_path / "m.jsonl").write_text(json.dumps({"uri": "a"}) + "\n")
    with pytest.raises(ValueError, match="line 1"):
        load_manifest(tmp_path / "m.jsonl")


def test_load_transcript_joins_per_speaker_in_time_order(tmp_path):
    path = tmp_path / "t.json"
    path.write_text(json.dumps([
        {"speaker": "A", "start_s": 5, "end_s": 6, "text": "world"},
        {"speaker": "A", "start_s": 0, "end_s": 1, "text": "hello"},
        {"speaker": "B", "start_s": 2, "end_s": 3, "text": "hi"},
    ]))
    assert load_transcript(path) == {"A": "hello world", "B": "hi"}


def test_perfect_and_relabelled_hypotheses_score_zero():
    assert diarization_scores(REF, REF)["der"] == pytest.approx(0.0)
    relabelled = [Turn({"A": "x", "B": "y"}[t.speaker_key], t.start_s, t.end_s) for t in REF]
    assert diarization_scores(REF, relabelled)["der"] == pytest.approx(0.0)
    assert speaker_mapping(REF, relabelled) == {"x": "A", "y": "B"}


def test_missing_speech_counts_as_missed():
    scores = diarization_scores([Turn("A", 0, 10)], [Turn("x", 0, 5)])
    assert scores["missed"] == pytest.approx(0.5)
    assert scores["der"] == pytest.approx(0.5)


def test_der_in_regions():
    assert der_in_regions(REF, REF, []) is None
    assert der_in_regions(REF, [Turn("x", 0, 10)], [(3, 4)]) == pytest.approx(0.5)


def test_overlap_detection():
    scores = overlap_detection([(1, 2)], [(1.5, 2.5)], duration_s=3)
    assert scores == pytest.approx({"precision": 0.5, "recall": 0.5, "f1": 0.5})
    empty = overlap_detection([], [], duration_s=3)
    assert math.isnan(empty["precision"]) and math.isnan(empty["recall"])


def test_error_catch_counts_and_rates():
    mapping = {"x": "A", "y": "B"}
    segments = [
        ScoredSegment("x", 0, 3, 0.9, "mixture"),  # correct, confident
        ScoredSegment("y", 7.5, 9.5, 0.4, "mixture"),  # wrong (A spoke), low confidence
        ScoredSegment("x", 4.5, 6.5, 0.8, "mixture"),  # wrong (B spoke), confident -> missed
        ScoredSegment("y", 3, 4, 0.0, "placeholder"),  # correct, always flagged
    ]
    (row,) = error_catch_counts(REF, segments, mapping, thresholds=(0.6,))
    assert row == {"threshold": 0.6, "segments": 4, "flagged": 2, "wrong": 2, "caught": 1}
    (rate,) = catch_rates([row])
    assert rate["flag_rate"] == pytest.approx(0.5)
    assert rate["catch_rate"] == pytest.approx(0.5)


def test_cpwer_uses_best_speaker_permutation():
    ref = {"A": "Hello there.", "B": "Good morning"}
    assert cpwer(ref, {"s1": "good morning", "s2": "hello their"})["cpwer"] == pytest.approx(0.25)
    extra = cpwer(ref, {"s1": "good morning", "s2": "hello their", "s3": "extra"})
    assert extra["errors"] == 2


def test_overlap_ratio():
    assert overlap_ratio(REF) == pytest.approx(1 / 10)


def test_format_report_lists_items_and_target(settings):
    from datetime import UTC, datetime

    result = ItemResult(
        uri="ES2004a", duration_s=600.0, processing_s=60.0, ref_overlap_ratio=0.12,
        diarization={"der": 0.2, "missed": 0.1, "false_alarm": 0.05, "confusion": 0.05, "jer": 0.3},
        der_overlap=0.5, overlap={"precision": 0.6, "recall": 0.5, "f1": 0.55},
        catch_counts=[{"threshold": 0.6, "segments": 100, "flagged": 20, "wrong": 10, "caught": 8}],
        cpwer=None, n_speakers_ref=4, n_speakers_hyp=4,
    )
    report = format_report([result], settings=settings, device="cpu", created=datetime(2026, 10, 9, tzinfo=UTC))
    assert "ES2004a" in report
    assert "| 0.60 | 20.0% | 80.0% |" in report
    assert "M3 target met at threshold 0.60" in report
```

- [ ] **Step 2: Install the eval extra and run the tests to verify they fail**

Run: `cd api && uv sync --extra eval && uv run pytest tests/unit/test_evaluation.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'saydo.evaluation'`.

- [ ] **Step 3: Implement**

`api/src/saydo/evaluation/rttm.py`:
```python
"""RTTM (reference/hypothesis turns) and UEM (scored region) file helpers."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from saydo.pipeline.types import Turn


def read_rttm(path: Path) -> dict[str, list[Turn]]:
    turns: dict[str, list[Turn]] = defaultdict(list)
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 8 or parts[0] != "SPEAKER":
            continue
        start, duration = float(parts[3]), float(parts[4])
        turns[parts[1]].append(Turn(parts[7], start, round(start + duration, 6)))
    return {uri: sorted(items, key=lambda t: (t.start_s, t.speaker_key)) for uri, items in turns.items()}


def write_rttm(path: Path, uri: str, turns: Sequence[Turn]) -> None:
    lines = [
        f"SPEAKER {uri} 1 {t.start_s:.3f} {t.end_s - t.start_s:.3f} <NA> <NA> {t.speaker_key} <NA> <NA>"
        for t in turns
    ]
    path.write_text("\n".join(lines) + "\n")


def read_uem(path: Path) -> dict[str, tuple[float, float]]:
    regions: dict[str, tuple[float, float]] = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 4:
            regions[parts[0]] = (float(parts[2]), float(parts[3]))
    return regions
```

`api/src/saydo/evaluation/manifest.py`:
```python
"""Evaluation manifests: one JSON object per line.

{"uri": "ES2004a", "audio": "audio/ES2004a.wav", "rttm": "rttm/ES2004a.rttm",
 "uem": "uem/ES2004a.uem", "transcript": "text/ES2004a.json", "min_speakers": 2, "max_speakers": 6}

Only uri, audio and rttm are required. Paths are relative to the manifest file.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvalItem:
    uri: str
    audio: Path
    rttm: Path
    uem: Path | None = None
    transcript: Path | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None


def load_manifest(path: Path) -> list[EvalItem]:
    base = path.parent
    items: list[EvalItem] = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)

        def resolve(key: str, raw: dict = raw) -> Path | None:
            value = raw.get(key)
            return (base / value).resolve() if value else None

        audio, rttm = resolve("audio"), resolve("rttm")
        if "uri" not in raw or audio is None or rttm is None:
            raise ValueError(f"{path}: line {number}: 'uri', 'audio' and 'rttm' are required")
        items.append(
            EvalItem(
                uri=raw["uri"], audio=audio, rttm=rttm, uem=resolve("uem"),
                transcript=resolve("transcript"),
                min_speakers=raw.get("min_speakers"), max_speakers=raw.get("max_speakers"),
            )
        )
    return items


def load_transcript(path: Path) -> dict[str, str]:
    """Reference transcript: JSON list of {"speaker", "start_s", "end_s", "text"}."""
    by_speaker: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for entry in json.loads(path.read_text()):
        by_speaker[entry["speaker"]].append((float(entry["start_s"]), entry["text"]))
    return {speaker: " ".join(text for _, text in sorted(parts)) for speaker, parts in by_speaker.items()}
```

`api/src/saydo/evaluation/metrics.py`:
```python
"""Evaluation metrics. DER/JER via pyannote.metrics (collar 0, overlap scored); cpWER is the
minimum over speaker assignments of total word edit distance, divided by reference words."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate

from saydo.pipeline.types import Turn

DEFAULT_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
WRONG_COVERAGE = 0.3


def to_annotation(turns: Iterable[Turn], uri: str = "item") -> Annotation:
    annotation = Annotation(uri=uri)
    for track, turn in enumerate(turns):
        if turn.end_s > turn.start_s:
            annotation[Segment(turn.start_s, turn.end_s), track] = turn.speaker_key
    return annotation


def _timeline(spans: Sequence[tuple[float, float]]) -> Timeline:
    return Timeline([Segment(s, e) for s, e in spans if e > s]).support()


def diarization_scores(
    reference: Sequence[Turn], hypothesis: Sequence[Turn], *, uem: tuple[float, float] | None = None
) -> dict[str, float]:
    ref, hyp = to_annotation(reference), to_annotation(hypothesis)
    scored = _timeline([uem]) if uem else None
    details = DiarizationErrorRate(collar=0.0, skip_overlap=False)(ref, hyp, uem=scored, detailed=True)
    total = details["total"] or 1.0
    jer = JaccardErrorRate(collar=0.0)(ref, hyp, uem=scored)
    return {
        "der": float(details["diarization error rate"]),
        "missed": details["missed detection"] / total,
        "false_alarm": details["false alarm"] / total,
        "confusion": details["confusion"] / total,
        "jer": float(jer),
    }


def der_in_regions(
    reference: Sequence[Turn], hypothesis: Sequence[Turn], regions: Sequence[tuple[float, float]]
) -> float | None:
    if not regions:
        return None
    metric = DiarizationErrorRate(collar=0.0, skip_overlap=False)
    return float(metric(to_annotation(reference), to_annotation(hypothesis), uem=_timeline(regions)))


def _frames(spans: Sequence[tuple[float, float]], n: int, frame_s: float) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    for start, end in spans:
        mask[int(round(start / frame_s)) : int(round(end / frame_s))] = True
    return mask


def overlap_detection(
    reference_regions: Sequence[tuple[float, float]],
    hypothesis_regions: Sequence[tuple[float, float]],
    *,
    duration_s: float,
    frame_s: float = 0.01,
) -> dict[str, float]:
    n = int(math.ceil(duration_s / frame_s))
    ref = _frames(reference_regions, n, frame_s)
    hyp = _frames(hypothesis_regions, n, frame_s)
    tp = int((ref & hyp).sum())
    fp = int((~ref & hyp).sum())
    fn = int((ref & ~hyp).sum())
    precision = tp / (tp + fp) if tp + fp else math.nan
    recall = tp / (tp + fn) if tp + fn else math.nan
    if math.isnan(precision) or math.isnan(recall):
        f1 = math.nan
    else:
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def speaker_mapping(reference: Sequence[Turn], hypothesis: Sequence[Turn]) -> dict[str, str]:
    if not reference or not hypothesis:
        return {}
    mapping = DiarizationErrorRate().optimal_mapping(to_annotation(reference), to_annotation(hypothesis))
    return {str(h): str(r) for h, r in mapping.items()}


@dataclass(frozen=True)
class ScoredSegment:
    speaker_key: str
    start_s: float
    end_s: float
    confidence: float
    origin: str


def segment_is_wrong(segment: ScoredSegment, reference: Sequence[Turn], mapping: dict[str, str]) -> bool:
    duration = segment.end_s - segment.start_s
    ref_speaker = mapping.get(segment.speaker_key)
    if duration <= 0 or ref_speaker is None:
        return True
    covered = sum(
        max(0.0, min(t.end_s, segment.end_s) - max(t.start_s, segment.start_s))
        for t in reference
        if t.speaker_key == ref_speaker
    )
    return covered < WRONG_COVERAGE * duration


def error_catch_counts(
    reference: Sequence[Turn],
    segments: Sequence[ScoredSegment],
    mapping: dict[str, str],
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> list[dict[str, float]]:
    wrong = [segment_is_wrong(s, reference, mapping) for s in segments]
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        flagged = [s.origin == "placeholder" or s.confidence < threshold for s in segments]
        rows.append(
            {
                "threshold": threshold,
                "segments": len(segments),
                "flagged": sum(flagged),
                "wrong": sum(wrong),
                "caught": sum(f and w for f, w in zip(flagged, wrong, strict=True)),
            }
        )
    return rows


def catch_rates(rows: Iterable[dict[str, float]]) -> list[dict[str, float]]:
    """Pool counts across recordings per threshold, then compute rates."""
    pooled: dict[float, dict[str, float]] = {}
    for row in rows:
        acc = pooled.setdefault(row["threshold"], {"segments": 0, "flagged": 0, "wrong": 0, "caught": 0})
        for key in acc:
            acc[key] += row[key]
    out = []
    for threshold in sorted(pooled):
        acc = pooled[threshold]
        out.append(
            {
                "threshold": threshold,
                "flag_rate": acc["flagged"] / acc["segments"] if acc["segments"] else math.nan,
                "catch_rate": acc["caught"] / acc["wrong"] if acc["wrong"] else math.nan,
                "wrong": acc["wrong"],
            }
        )
    return out


_NON_WORD = re.compile(r"[^\w\s']")


def normalize_words(text: str) -> list[str]:
    return _NON_WORD.sub(" ", text.lower()).split()


def cpwer(reference: dict[str, str], hypothesis: dict[str, str]) -> dict[str, float]:
    from rapidfuzz.distance import Levenshtein
    from scipy.optimize import linear_sum_assignment

    ref = [normalize_words(t) for t in reference.values()]
    hyp = [normalize_words(t) for t in hypothesis.values()]
    size = max(len(ref), len(hyp))
    ref_words = sum(len(words) for words in ref)
    if size == 0:
        return {"cpwer": math.nan, "errors": 0.0, "ref_words": 0}
    ref += [[] for _ in range(size - len(ref))]
    hyp += [[] for _ in range(size - len(hyp))]
    cost = np.array([[Levenshtein.distance(r, h) for h in hyp] for r in ref], dtype=float)
    rows, cols = linear_sum_assignment(cost)
    errors = float(cost[rows, cols].sum())
    return {"cpwer": errors / ref_words if ref_words else math.nan, "errors": errors, "ref_words": ref_words}
```

`api/src/saydo/evaluation/runner.py`:
```python
"""Evaluate the pipeline on one manifest item and render a Markdown report."""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from saydo import __version__
from saydo.config import ReviewSettings, Settings
from saydo.evaluation.manifest import EvalItem, load_transcript
from saydo.evaluation.metrics import (
    ScoredSegment,
    catch_rates,
    cpwer,
    der_in_regions,
    diarization_scores,
    error_catch_counts,
    overlap_detection,
    speaker_mapping,
    to_annotation,
)
from saydo.evaluation.rttm import read_rttm, read_uem
from saydo.pipeline.normalize import normalize
from saydo.pipeline.overlap import find_overlap_regions
from saydo.pipeline.run import Plugins, analyze
from saydo.pipeline.types import Turn, load_audio

M3_CATCH_TARGET = 0.70
M3_FLAG_LIMIT = 0.25


@dataclass
class ItemResult:
    uri: str
    duration_s: float
    processing_s: float
    ref_overlap_ratio: float
    diarization: dict[str, float]
    der_overlap: float | None
    overlap: dict[str, float]
    catch_counts: list[dict[str, float]]
    cpwer: dict[str, float] | None
    n_speakers_ref: int
    n_speakers_hyp: int


def _true_overlaps(reference: list[Turn]) -> list[tuple[float, float]]:
    return [(r.start_s, r.end_s) for r in find_overlap_regions(reference, min_duration_s=0.0, merge_gap_s=0.0)]


def overlap_ratio(reference: list[Turn]) -> float:
    speech = to_annotation(reference).get_timeline().support().duration()
    overlapped = sum(end - start for start, end in _true_overlaps(reference))
    return overlapped / speech if speech else 0.0


def evaluate_item(item: EvalItem, plugins: Plugins, review: ReviewSettings, workdir: Path) -> ItemResult:
    normalized = normalize(
        item.audio, workdir / f"{item.uri}.wav", keep_channels=plugins.diarizer.wants_multichannel
    )
    audio = load_audio(normalized.path)
    started = time.perf_counter()
    analysis = analyze(
        audio, plugins, review, min_speakers=item.min_speakers, max_speakers=item.max_speakers
    )
    processing_s = time.perf_counter() - started

    reference = read_rttm(item.rttm).get(item.uri)
    if reference is None:
        raise ValueError(f"{item.rttm} has no turns for uri '{item.uri}'")
    uem = read_uem(item.uem).get(item.uri) if item.uem else None
    hypothesis = list(analysis.turns)
    mapping = speaker_mapping(reference, hypothesis)
    scored = [
        ScoredSegment(d.speaker_key, d.start_s, d.end_s, s.confidence, d.origin)
        for d, s in zip(analysis.drafts, analysis.scores, strict=True)
    ]

    cp = None
    if item.transcript:
        spoken: dict[str, list[str]] = defaultdict(list)
        for draft in analysis.drafts:
            if draft.text:
                spoken[draft.speaker_key].append(draft.text)
        cp = cpwer(load_transcript(item.transcript), {k: " ".join(v) for k, v in spoken.items()})

    return ItemResult(
        uri=item.uri,
        duration_s=audio.duration_s,
        processing_s=processing_s,
        ref_overlap_ratio=overlap_ratio(reference),
        diarization=diarization_scores(reference, hypothesis, uem=uem),
        der_overlap=der_in_regions(reference, hypothesis, _true_overlaps(reference)),
        overlap=overlap_detection(
            _true_overlaps(reference),
            [(r.start_s, r.end_s) for r in analysis.regions],
            duration_s=audio.duration_s,
        ),
        catch_counts=error_catch_counts(reference, scored, mapping),
        cpwer=cp,
        n_speakers_ref=len({t.speaker_key for t in reference}),
        n_speakers_hyp=len({t.speaker_key for t in hypothesis}),
    )


def _pct(value: float | None) -> str:
    return "n/a" if value is None or math.isnan(value) else f"{value * 100:.1f}%"


def _weighted(results: list[ItemResult], get) -> float | None:  # type: ignore[no-untyped-def]
    pairs = [(get(r), r.duration_s) for r in results]
    pairs = [(v, w) for v, w in pairs if v is not None and not math.isnan(v)]
    total = sum(w for _, w in pairs)
    return sum(v * w for v, w in pairs) / total if total else None


def format_report(results: list[ItemResult], *, settings: Settings, device: str, created: datetime) -> str:
    p = settings.pipeline
    lines = [
        "# SayDo evaluation report",
        "",
        f"- Date: {created:%Y-%m-%d %H:%M} UTC",
        f"- SayDo: {__version__} · device: {device}",
        f"- Transcriber: {p.transcriber.name} {p.transcriber.options}",
        f"- Diarizer: {p.diarizer.name} {p.diarizer.options}",
        f"- Embedder: {p.embedder.name} {p.embedder.options}",
        f"- Review threshold: {settings.review.threshold}",
        "- DER/JER: collar 0 s, overlapping speech scored. Overlap metrics use reference overlaps.",
        "",
        "## Per recording",
        "",
        "| Recording | Duration | Ref overlap | DER | DER in overlaps | JER | Overlap P / R / F1 | cpWER | Speakers ref/hyp | RTF |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        o = r.overlap
        lines.append(
            f"| {r.uri} | {r.duration_s / 60:.1f} min | {_pct(r.ref_overlap_ratio)} | "
            f"{_pct(r.diarization['der'])} | {_pct(r.der_overlap)} | {_pct(r.diarization['jer'])} | "
            f"{_pct(o['precision'])} / {_pct(o['recall'])} / {_pct(o['f1'])} | "
            f"{_pct(r.cpwer['cpwer']) if r.cpwer else 'n/a'} | {r.n_speakers_ref}/{r.n_speakers_hyp} | "
            f"{r.processing_s / r.duration_s:.2f} |"
        )
    pooled_cp = [r.cpwer for r in results if r.cpwer]
    cp_errors = sum(c["errors"] for c in pooled_cp)
    cp_words = sum(c["ref_words"] for c in pooled_cp)
    total_duration = sum(r.duration_s for r in results)
    lines += [
        "",
        "## Totals (duration-weighted)",
        "",
        f"- DER: {_pct(_weighted(results, lambda r: r.diarization['der']))}",
        f"- DER inside overlaps: {_pct(_weighted(results, lambda r: r.der_overlap))}",
        f"- JER: {_pct(_weighted(results, lambda r: r.diarization['jer']))}",
        f"- Overlap F1: {_pct(_weighted(results, lambda r: r.overlap['f1']))}",
        f"- cpWER (pooled): {_pct(cp_errors / cp_words) if cp_words else 'n/a'}",
        f"- Real-time factor: {sum(r.processing_s for r in results) / total_duration:.2f}"
        if total_duration
        else "- Real-time factor: n/a",
        "",
        "## Review flags: error-catch rate",
        "",
        "A segment is wrong when its mapped reference speaker covers < 30 % of it. Flagged = placeholder "
        "or confidence below the threshold.",
        "",
        "| Threshold | Flag rate | Error-catch rate | Wrong segments |",
        "|---|---|---|---|",
    ]
    rates = catch_rates(row for r in results for row in r.catch_counts)
    for row in rates:
        lines.append(
            f"| {row['threshold']:.2f} | {_pct(row['flag_rate'])} | {_pct(row['catch_rate'])} | {int(row['wrong'])} |"
        )
    met = [
        row for row in rates
        if row["flag_rate"] <= M3_FLAG_LIMIT and not math.isnan(row["catch_rate"])
        and row["catch_rate"] >= M3_CATCH_TARGET
    ]
    lines.append("")
    lines.append(
        f"M3 target met at threshold {met[0]['threshold']:.2f}"
        if met
        else f"M3 target (catch ≥ {M3_CATCH_TARGET:.0%} at flag rate ≤ {M3_FLAG_LIMIT:.0%}) not met"
    )
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Write the CLI scripts and docs**

`eval/run_eval.py`:
```python
"""Run SayDo's pipeline on an evaluation manifest and write a Markdown report to eval/reports/.

  cd api && HF_TOKEN=hf_xxx uv run --extra ml --extra eval \
      python ../eval/run_eval.py --manifest ../eval/data/ami/manifest.jsonl

The pipeline configuration comes from saydo.yaml / SAYDO__* env vars, as for the server.
"""

import argparse
import hashlib
import json
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from saydo.config import Settings
from saydo.evaluation.manifest import load_manifest
from saydo.evaluation.runner import evaluate_item, format_report
from saydo.pipeline.run import build_plugins


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "reports")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N items")
    args = parser.parse_args()

    settings = Settings()
    items = load_manifest(args.manifest)[: args.limit]
    plugins = build_plugins(settings)
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for index, item in enumerate(items, start=1):
            print(f"[{index}/{len(items)}] {item.uri}", file=sys.stderr, flush=True)
            results.append(evaluate_item(item, plugins, settings.review, Path(tmp)))

    created = datetime.now(UTC)
    config = {"pipeline": settings.pipeline.model_dump(), "review": settings.review.model_dump()}
    digest = hashlib.sha1(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]  # noqa: S324
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{created:%Y-%m-%d}-{digest}.md"
    out.write_text(format_report(results, settings=settings, device=plugins.device, created=created))
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

`eval/prepare_ami.py`:
```python
"""Download the AMI test set (single distant microphone: Array1-01) with reference RTTM/UEM
from BUTSpeechFIT/AMI-diarization-setup ("only_words" references). AMI is CC-BY-4.0.
About 0.8 GB for all 16 test meetings.

  python eval/prepare_ami.py --out eval/data/ami [--meetings ES2004a IS1009a]
"""

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

TEST_MEETINGS = [
    "IS1009a", "IS1009b", "IS1009c", "IS1009d", "ES2004a", "ES2004b", "ES2004c", "ES2004d",
    "TS3003a", "TS3003b", "TS3003c", "TS3003d", "EN2002a", "EN2002b", "EN2002c", "EN2002d",
]
AUDIO_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{m}/audio/{m}.Array1-01.wav"
SETUP_URL = "https://raw.githubusercontent.com/BUTSpeechFIT/AMI-diarization-setup/main"
RTTM_URL = SETUP_URL + "/only_words/rttms/test/{m}.rttm"
UEM_URL = SETUP_URL + "/uems/test/{m}.uem"


def fetch(url: str, target: Path) -> bool:
    if target.exists():
        return True
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, target.with_suffix(".part"))  # noqa: S310
    except urllib.error.HTTPError as error:
        print(f"  skip {url}: HTTP {error.code}", file=sys.stderr)
        return False
    target.with_suffix(".part").rename(target)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("eval/data/ami"))
    parser.add_argument("--meetings", nargs="*", default=TEST_MEETINGS)
    args = parser.parse_args()

    lines = []
    for meeting in args.meetings:
        print(meeting, file=sys.stderr, flush=True)
        ok = all(
            [
                fetch(AUDIO_URL.format(m=meeting), args.out / "audio" / f"{meeting}.wav"),
                fetch(RTTM_URL.format(m=meeting), args.out / "rttm" / f"{meeting}.rttm"),
                fetch(UEM_URL.format(m=meeting), args.out / "uem" / f"{meeting}.uem"),
            ]
        )
        if ok:
            lines.append(json.dumps({
                "uri": meeting, "audio": f"audio/{meeting}.wav",
                "rttm": f"rttm/{meeting}.rttm", "uem": f"uem/{meeting}.uem",
            }))
    (args.out / "manifest.jsonl").write_text("\n".join(lines) + "\n")
    print(args.out / "manifest.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

`eval/README.md`:
````markdown
# Evaluating SayDo

Measures diarization, overlap handling and review-flag honesty on recordings with known
speakers. Reports land in `eval/reports/` (committed); audio lives in `eval/data/` (ignored).

## Datasets
- **AMI test set** (16 meetings, single distant microphone, CC-BY-4.0):
  `python eval/prepare_ami.py --out eval/data/ami`
- **Your own meetings**: record 3–5 sprint meetings **with everyone's consent**, label who
  spoke when (RTTM; e.g. with Audacity labels converted to RTTM), optionally add a reference
  transcript, and write a manifest:
  ```json
  {"uri": "sprint-01", "audio": "sprint-01.webm", "rttm": "sprint-01.rttm", "transcript": "sprint-01.json"}
  ```
  `transcript` is a JSON list of `{"speaker", "start_s", "end_s", "text"}`.
- **LibriCSS** (controlled overlap ratios): planned for M5; any dataset works once it has a manifest.

## Run
```bash
cd api
HF_TOKEN=hf_xxx uv run --extra ml --extra eval python ../eval/run_eval.py --manifest ../eval/data/ami/manifest.jsonl
```
Use `--limit 2` for a quick check. The pipeline configuration comes from `saydo.yaml` / env vars.

## Metrics
- **DER** (diarization error rate): missed speech + false alarm + speaker confusion, divided by
  reference speech time; collar 0, overlapping speech scored. Also reported **inside reference
  overlap regions only**.
- **JER** (Jaccard error rate): per-speaker error, less dominated by talkative speakers.
- **Overlap P / R / F1**: frame-level (10 ms) detection of overlapping speech.
- **cpWER**: word error rate after the best speaker assignment (needs reference transcripts).
- **Error-catch rate**: of the segments attributed to the wrong speaker, the share flagged
  "Needs review", reported against the share of all segments flagged. M3 target: ≥ 70 % caught
  while flagging ≤ 25 %.
- **RTF** (real-time factor): processing time ÷ audio duration (excludes model loading).
````

`eval/reports/.gitkeep`: empty file.

In `.github/workflows/ci.yml`, change the api job's `- run: uv sync --locked` to `- run: uv sync --locked --extra eval`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd api && uv run pytest -q && uv run ruff check . ../eval && uv run mypy src`
Expected: all pass, including `tests/unit/test_evaluation.py`, which is not skipped because the eval extra is installed.

- [ ] **Step 6: Smoke-run the harness on one AMI meeting**

```bash
python3 eval/prepare_ami.py --out eval/data/ami --meetings ES2004a
cd api && HF_TOKEN=hf_xxx uv run --extra ml --extra eval python ../eval/run_eval.py --manifest ../eval/data/ami/manifest.jsonl
```
Expected: a report path is printed, and the report shows a DER for ES2004a in the 10–35 % range. Don't commit this one-meeting report; Task 21 produces the real one.

- [ ] **Step 7: Commit**

```bash
git add api/src/saydo/evaluation api/tests/unit/test_evaluation.py eval .github/workflows/ci.yml
git commit -m "feat(eval): DER/JER, overlap detection, cpWER and error-catch evaluation harness"
```

---

### Task 21: Milestone acceptance runs and Phase 1 report (M0–M4 sign-off)

This task runs the spec §18 acceptance tests for M0–M4 on real hardware and records the evidence. No new product code. If a criterion fails, open a fix task instead of weakening the criterion. The exception is M3's flag threshold, which the spec explicitly allows to be recalibrated, provided the change is documented.

**Files:**
- Create: `docs/acceptance/phase1.md`
- Create: `eval/reports/<date>-<hash>.md` (generated)
- Modify (only if M3 recalibration is needed): `saydo.example.yaml` (`review.threshold`), `api/src/saydo/config.py` (`ReviewSettings.threshold` default) and `api/tests/unit/test_config.py` (expected default)

- [ ] **Step 1: M0: stack starts**

```bash
HF_TOKEN=hf_xxx docker compose up --build -d && sleep 20
curl -s localhost:8000/api/v1/health
curl -s -o /dev/null -w "%{http_code}\n" localhost:3000/
```
Expected: `{"status":"ok","version":"0.1.0"}` and `200`. If the repository is pushed to GitHub, also confirm the latest CI run is green.

- [ ] **Step 2: M1: recording works in all three browsers, normalization and deletion are clean**

For each of Chrome, Firefox and Safari, at http://localhost:3000:
1. Choose the microphone source and record 10 minutes of real conversation.
2. Stop, then upload.
3. Wait until the transcript appears.
4. Check the normalized audio, using the recording ID from the URL:
```bash
docker compose exec api ffprobe -v error -show_entries stream=sample_rate,channels -of csv=p=0 /data/recordings/<id>/audio.wav
```
   Expected: `16000,1`.
5. Click **Delete recording**, then:
```bash
docker compose exec api sh -c 'ls /data/recordings/<id> 2>&1; python -c "import sqlite3;print(sqlite3.connect(\"/data/saydo.db\").execute(\"select count(*) from recording where id=?\",(\"<id>\",)).fetchone())"'
```
   Expected: `No such file or directory` and `(0,)`.

Also run `pnpm --filter saydo-web e2e` and confirm it passes.

- [ ] **Step 3: M2: diarization quality on AMI and a one-hour meeting**

```bash
python3 eval/prepare_ami.py --out eval/data/ami
cd api && HF_TOKEN=hf_xxx uv run --extra ml --extra eval python ../eval/run_eval.py --manifest ../eval/data/ami/manifest.jsonl
```
Expected: the total DER in the report is ≤ 21.9 %. The published `community-1` AMI SDM DER is 19.9 %, and the acceptance margin is 2 points. Note that our references are `only_words` RTTMs with collar 0, which matches pyannote's benchmark setup.

Then build and process a one-hour file on the reference GPU machine, timing it:
```bash
cd eval/data/ami/audio
ffmpeg -loglevel error -y -i ES2004a.wav -i ES2004b.wav -i ES2004c.wav -i ES2004d.wav \
  -filter_complex concat=n=4:v=0:a=1 -t 3600 /tmp/one-hour.wav
START=$(date +%s)
ID=$(curl -s -F file=@/tmp/one-hour.wav -F source=upload localhost:8000/api/v1/recordings | python3 -c 'import sys,json;print(json.load(sys.stdin)["id"])')
until curl -s localhost:8000/api/v1/recordings/$ID/status | grep -qE '"(ready|failed)"'; do sleep 10; done
echo "status: $(curl -s localhost:8000/api/v1/recordings/$ID/status) elapsed: $(( $(date +%s) - START ))s"
```
Expected: the status is `ready`. Record the elapsed seconds and the GPU model. Repeat once on the CPU-only development Mac and record that time too, for the README hardware table.

- [ ] **Step 4: M3: overlap surfaced and review flags honest**

- **Overlap regions are surfaced:** covered by `test_end_to_end_with_fake_plugins`, by the end-to-end test, and by opening one AMI meeting in the UI. To open it, upload `eval/data/ami/audio/ES2004a.wav` and confirm that "Overlapping speech" blocks appear.
- **Error-catch target:** read the "Review flags" table in the Step 3 report.
  - If the report says `M3 target met at threshold X` and X equals the configured threshold (0.6), the criterion passes.
  - If it is met only at another threshold, set `review.threshold` to the lowest threshold that meets the target. Update the default in `ReviewSettings`, `saydo.example.yaml` and the expected value in `test_config.py`. Then re-run `uv run pytest -q` and the evaluation.
  - If it isn't met at any threshold, keep 0.6 and record the best achievable trade-off. Add a line to the README's "Known limitations" section with the measured catch rate and flag rate, and open an M5 follow-up to improve the confidence signals.

- [ ] **Step 5: M4: corrections**

Covered by `api/tests/api/test_corrections.py`, `SpeakerChip.test.tsx` and the end-to-end rename. Also check by hand on a real AMI transcript:
- rename a speaker;
- reassign one flagged segment;
- merge two speakers;
- reload the page and confirm that all three changes persisted.

- [ ] **Step 6: Write the acceptance record**

`docs/acceptance/phase1.md` must contain a table with one row per spec §18 criterion for M0–M4, with these columns:
- milestone;
- criterion;
- evidence: the command or report path;
- measured value;
- pass/fail;
- notes.

Fill every value from Steps 1–5. Include:
- the hardware used (GPU model, CPU or Mac model);
- the one-hour processing times;
- the AMI total DER and the DER inside overlaps;
- the chosen review threshold and its catch and flag rates.

Then add the measured numbers to the README's "Accuracy" section, replacing the link-only text with a three-line summary plus the link.

- [ ] **Step 7: Commit**

```bash
git add docs/acceptance/phase1.md eval/reports README.md saydo.example.yaml api/src/saydo/config.py api/tests/unit/test_config.py
git commit -m "docs: phase 1 acceptance results and evaluation report"
```
