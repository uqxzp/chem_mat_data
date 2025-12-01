from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, Tuple, Optional

from chem_mat_data.agent.opencode_client import send_message

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_LINKS_PATH = ARTIFACTS_DIR / "download_links.json"
DEFAULT_SCRIPT_PATH = ARTIFACTS_DIR / "generated_script.py"
DEFAULT_OPENCODE_CONFIG = Path.home() / ".config" / "opencode" / "config.json"
REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
DEFAULT_BASE_EXPERIMENT = SCRIPTS_DIR / "create_graph_datasets__from_xyz.py"

DEFAULT_LINK_PROMPT = """
You are a research assistant who specializes in chemistry and materials science datasets.
Given the URL of a publication, find direct download URLs for the dataset released with the paper.
Look for supplementary information, GitHub/Zenodo/Figshare/OSF releases, institutional or lab
repositories, or any other location that hosts the dataset. Only keep links that directly download
files or archives containing molecular structure data (SMILES, SDF, XYZ, CIF, or archives of those).

Return a pure JSON object with this shape and nothing else:
{
  "download_links": ["https://direct-download-1", "..."],
  "notes": "short bullet points about where the links were found or why none were found"
}

If no dataset links are available, respond with an empty list for "download_links" and explain
briefly in "notes" what you checked.
"""

DEFAULT_SCRIPT_PROMPT = """
You are an autonomous coding agent extending the ChemMatData project. Using the supplied download
links, produce a Python conversion script similar to the modules in chem_mat_data/scripts.
Requirements:
- The dataset contains molecules (SMILES or XYZ files are expected).
- Prefer the existing style: use pycomex Experiment for the workflow, store artifacts in the
  experiment path, and emit graphs that fit ChemMatData conventions.
- Include code to download the dataset from the provided links, extract archives, and iterate over
  the molecule files.
- Assume the agent has access to an MCP file server (abs222222/mcp-file-downloader) to inspect
  files when needed; describe which paths or file samples should be fetched via MCP for validation,
  but DO NOT call the MCP tool in this chat.
- Do NOT call load_xyz_dataset or ensure_dataset (they rely on remote file-share metadata). Instead,
  parse the extracted SMILES/XYZ files directly in this script with a local helper (e.g.,
  load_local_xyz_dataset using load_xyz_as_mol).
- Define BASE_EXPERIMENT as an absolute path and pass it directly to Experiment.extend; do NOT use
  just a filename. If the file does not exist, fall back to a standalone Experiment.
- Include required imports (e.g., pandas, shutil) when you use them.
- Set BASE_EXPERIMENT to the absolute path of chem_mat_data/scripts/create_graph_datasets__from_xyz.py,
  and if that file is missing, fall back to defining a self-contained Experiment without extend().
- Keep dependencies minimal and use placeholders where data-specific logic is unknown.

Return only the Python script text. Do not wrap it in Markdown.
"""


@dataclass
class LinkDiscoveryResult:
    """
    Structured result for dataset download link discovery.
    """

    publication_url: str
    download_links: list[str]
    notes: str
    raw_response: str

    def save(self, path: Path | str | None = None) -> Path:
        """
        Saves the discovery result JSON to ``path``.

        :param path: Optional output path. Defaults to ``DEFAULT_LINKS_PATH``.

        :returns: The resolved path that was written.
        """
        target = _resolve_path(path, DEFAULT_LINKS_PATH)
        payload = {
            "publication_url": self.publication_url,
            "download_links": self.download_links,
            "notes": self.notes,
            "raw_response": self.raw_response,
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return target


@dataclass
class ScriptGenerationResult:
    """
    Structured result for dataset script generation.
    """

    publication_url: str | None
    download_links: list[str]
    script_text: str
    raw_response: str
    output_path: Path

    def save(self, path: Path | str | None = None) -> Path:
        """
        Saves ``script_text`` to ``path``.

        :param path: Optional output path. Defaults to ``DEFAULT_SCRIPT_PATH``.

        :returns: The resolved path that was written.
        """
        target = _resolve_path(path, DEFAULT_SCRIPT_PATH)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            handle.write(self.script_text)
        self.output_path = target
        return target


def discover_links(publication_url: str, prompt_override: str | None = None) -> LinkDiscoveryResult:
    """
    Runs the first LLM prompt to discover dataset download links for ``publication_url``.

    :param publication_url: The publication URL to inspect for dataset links.
    :param prompt_override: Optional custom prompt to override the default template.

    :returns: The structured link discovery result.
    """
    prompt = build_link_prompt(publication_url, prompt_override)
    raw_response = send_message(prompt)
    result = parse_link_response(raw_response, publication_url)
    return result


def generate_script_from_links(
        download_links: Sequence[str],
        publication_url: str | None = None,
        mcp_endpoint: str | None = None,
        prompt_override: str | None = None,
        output_path: Path | str | None = None,
) -> ScriptGenerationResult:
    """
    Runs the second LLM prompt to generate a dataset conversion script for ``download_links``.

    :param download_links: The download URLs to include in the script prompt.
    :param publication_url: Optional publication URL for additional context.
    :param mcp_endpoint: Optional MCP file server endpoint to mention in the prompt.
    :param prompt_override: Optional custom prompt to override the default template.
    :param output_path: Optional path where the generated script will be written.

    :returns: The structured script generation result with the saved file path.
    """
    if mcp_endpoint is None:
        mcp_endpoint = detect_mcp_hint()

    prompt = build_script_prompt(download_links, publication_url, mcp_endpoint, prompt_override)
    raw_response = send_message(prompt)
    script_result = ScriptGenerationResult(
        publication_url=publication_url,
        download_links=list(download_links),
        script_text=raw_response,
        raw_response=raw_response,
        output_path=_resolve_path(output_path, DEFAULT_SCRIPT_PATH),
    )
    script_result.save(output_path)
    return script_result


def load_links(path: Path | str | None = None) -> LinkDiscoveryResult:
    """
    Loads a ``LinkDiscoveryResult`` from disk.

    :param path: Optional custom path. Defaults to ``DEFAULT_LINKS_PATH``.

    :returns: The loaded link discovery result.
    """
    target = _resolve_path(path, DEFAULT_LINKS_PATH)
    if not target.exists():
        raise FileNotFoundError(f"No stored download links found at {target}")

    with target.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    links = _normalize_links(payload.get("download_links", []))
    notes = str(payload.get("notes", "")).strip()
    publication_url = str(payload.get("publication_url", "")).strip()
    raw_response = str(payload.get("raw_response", "")).strip()
    return LinkDiscoveryResult(
        publication_url=publication_url,
        download_links=links,
        notes=notes,
        raw_response=raw_response,
    )


def build_link_prompt(publication_url: str, prompt_override: str | None = None) -> str:
    """
    Builds the prompt text for discovering dataset download links.

    :param publication_url: The publication URL to append to the prompt.
    :param prompt_override: Optional custom prompt text.

    :returns: The final prompt string.
    """
    base_prompt = (prompt_override or os.environ.get("OPENCODE_LINK_PROMPT", DEFAULT_LINK_PROMPT)).strip()
    return f"{base_prompt}\n\nPublication: {publication_url}".strip()


def build_script_prompt(
        download_links: Sequence[str],
        publication_url: str | None = None,
        mcp_endpoint: str | None = None,
        prompt_override: str | None = None,
) -> str:
    """
    Builds the prompt text for generating a dataset conversion script.

    :param download_links: The download URLs to pass to the model.
    :param publication_url: Optional publication URL to include for context.
    :param mcp_endpoint: Optional MCP file server endpoint used for inspection.
    :param prompt_override: Optional custom prompt text.

    :returns: The final prompt string.
    """
    base_prompt = (prompt_override or os.environ.get("OPENCODE_SCRIPT_PROMPT", DEFAULT_SCRIPT_PROMPT)).strip()
    link_lines = "\n".join(f"- {link}" for link in download_links)
    details: list[str] = [
        "Use these dataset download URLs:",
        link_lines or "- (no links supplied)",
    ]
    if publication_url:
        details.append(f"Publication URL: {publication_url}")
    if mcp_endpoint:
        details.append(f"MCP file server info: {mcp_endpoint}")
    context = [
        "Project context:",
        f"- Repo root: {REPO_ROOT}",
        f"- Scripts dir: {SCRIPTS_DIR}",
        f"- Preferred base experiment: {DEFAULT_BASE_EXPERIMENT}",
        "- Avoid remote file-share dependencies; implement local file parsing.",
        "- If file names are uncertain, add small detection logic and sensible defaults.",
    ]
    prompt = f"{base_prompt}\n\n" + "\n".join(details) + "\n\n" + "\n".join(context)
    return prompt.strip()


def parse_link_response(raw_response: str, publication_url: str) -> LinkDiscoveryResult:
    """
    Parses the raw LLM response and extracts download links.

    :param raw_response: The raw text returned by the LLM.
    :param publication_url: The publication URL used for the query.

    :returns: The structured link discovery result.
    """
    download_links, notes = _parse_links_from_json(raw_response)
    if not download_links:
        download_links, fallback_notes = _parse_links_with_regex(raw_response)
        if fallback_notes:
            notes = notes or fallback_notes

    return LinkDiscoveryResult(
        publication_url=publication_url,
        download_links=download_links,
        notes=notes,
        raw_response=raw_response,
    )


def _parse_links_from_json(raw_response: str) -> Tuple[list[str], str]:
    """
    Attempts to parse download links from a JSON string.
    """
    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError:
        return [], ""

    links = _normalize_links(payload.get("download_links", []))
    notes = str(payload.get("notes", "")).strip()
    return links, notes


def _parse_links_with_regex(raw_response: str) -> Tuple[list[str], str]:
    """
    Extracts URLs from a free-form text response as a fallback.
    """
    pattern = re.compile(r"https?://[^\s\\\"]+")
    links = [match.rstrip(".,);") for match in pattern.findall(raw_response)]
    notes = "Links extracted from free-form response."
    return _unique_preserve_order(links), notes


def _normalize_links(links: Iterable[str]) -> list[str]:
    """
    Normalizes and de-duplicates download links.
    """
    cleaned = [str(link).strip() for link in links if str(link).strip()]
    return _unique_preserve_order(cleaned)


def _unique_preserve_order(items: Iterable[str]) -> list[str]:
    """
    Returns unique values from ``items`` while preserving input order.
    """
    seen = set()
    unique_items: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        unique_items.append(item)
    return unique_items


def _resolve_path(path: Path | str | None, default: Path) -> Path:
    """
    Resolves ``path`` or returns ``default``.
    """
    if path is None:
        return default
    return Path(path)


def detect_mcp_hint(config_path: Optional[Path] = None) -> Optional[str]:
    """
    Attempts to build a short MCP hint string from the Opencode config file.

    :param config_path: Optional path to an Opencode config JSON file.

    :returns: A hint string describing the MCP servers or ``None`` if unavailable.
    """
    path = config_path or DEFAULT_OPENCODE_CONFIG
    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None

    mcp = config.get("mcp")
    if not isinstance(mcp, dict):
        return None

    parts: list[str] = []
    for name, entry in mcp.items():
        if not isinstance(entry, dict):
            continue
        enabled = entry.get("enabled", False)
        mcp_type = entry.get("type", "unknown")
        command = entry.get("command")
        cmd_str = " ".join(command) if isinstance(command, list) else str(command) if command else None
        label = f"{name} (type={mcp_type}"
        if cmd_str:
            label += f", command='{cmd_str}'"
        label += f", enabled={enabled})"
        parts.append(label)

    if not parts:
        return None

    return "Available MCP servers: " + "; ".join(parts)
