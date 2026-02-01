from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from chem_mat_data.agent.opencode_client import send_message

ARTIFACTS_DIR: Path = Path(__file__).resolve().parent / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# link discovery
DOWNLOADS_DIR = ARTIFACTS_DIR / "downloads"
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
LINKS_PATH: Path = ARTIFACTS_DIR / "download_links.json"
LINK_PROMPT: str = """
You are a research assistant specializing in chemistry and materials science datasets.

Goal:
Given a publication URL, find the direct download URL(s) for the molecular dataset released with the paper.

Return ONLY a single JSON object (no prose/markdown) with exactly:
{
  "download_links": ["https://direct-download-1", "..."],
  "downloaded_files": ["chem_mat_data/agent/artifacts/downloads/FILE1", "..."],
  "notes": "..."
}

Rules:
- Return ONLY direct file/archive download URLs that immediately trigger a browser download.
- Do NOT return publication/landing/home/supplement pages unless they are direct file downloads.
- Search supporting/supplementary materials and external repositories (GitHub/Zenodo/Figshare/OSF), institutional/lab repos, or any other linked host.
- Ignore PDB/CIF datasets (treat as not found).

Selecting what to download (avoid duplicates):
- Download the dataset only ONCE per representation (do NOT download the same dataset again in different formats).
- If the dataset is available in multiple formats, always prefer tabular files (.csv/.tsv/.xlsx) over archives (.zip/.tar.gz). If a tabular file exists, do NOT download an archive that appears to bundle the same data.
- Download multiple files only if they are distinct non-overlapping parts of the dataset (e.g., split tables/parts), not alternate formats.

Tool use:
- Do not ask for permissions or mention access limitations. You are allowed to fetch any needed pages.
- When reading any webpage, ALWAYS use `crawlfetch` (NOT `webfetch`).

Downloading (MCP tool):
- After identifying the best dataset file(s), download them using `file_downloader_download_file` with args: `url` (required), `filename` (required), `use_browser` (optional).
- Save ONLY inside `chem_mat_data/agent/artifacts/downloads` and never outside it.
- Choose a clear filename based on dataset name + format (e.g., `AqSolDB.csv`).
- "downloaded_files" MUST contain the exact `filename` values used, in the same order as "download_links" (1:1 correspondence).
- Always pass `filename` as an ABSOLUTE path (not relative).
- "downloaded_files" MUST contain the exact local `filename` paths used in the tool call (not URLs).
- "download_links" and "downloaded_files" must have the same length and correspond by index.

Notes:
- If you found dataset links and downloaded them, set "notes" to "".
- If you found no dataset links, set "download_links" and "downloaded_files" to [] and set "notes" to 1–2 sentences describing what you checked.
"""



# script generation
SCRIPTS_ARTIFACTS_DIR: Path = ARTIFACTS_DIR / "scripts"
SCRIPTS_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
EXAMPLE_SCRIPTS_DIR: Path = Path(__file__).resolve().parent / "example_scripts"
BASE_TEMPLATE_PATH: Path = (
    Path(__file__).resolve().parents[1] / "scripts" / "create_graph_datasets.py"
)
SCRIPT_GENERATION_PROMPT: str = """
You are a Python developer who specializes in writing dataset processing scripts. You have file tools enabled (list, glob, read, grep). Use them to inspect the downloaded dataset file and existing scripts before writing code.

Inputs for this task:
- Downloaded dataset file (may be an archive with subfolders): {dataset_path}
- Example scripts for different datasets to mimic: {example_scripts_dir}
- Base template to extend (use this exact path): {base_template_path}

Goal:
- Inspect the dataset (and, if archived, its contents) and generate a processing script that follows the style and structure of the existing scripts in {example_scripts_dir}.
- When looking at example scripts, inspect at most 5 files that seem most similar; do not enumerate the entire directory.
- Extend the base experiment at {base_template_path} using its absolute path (e.g., Experiment.extend(str({base_template_path}))) so the script runs without missing-path errors.
- The script should be ready to run locally against the downloaded data.
- Return only the complete Python script content; do not wrap it in markdown fences or add prose.

Note: Once you have a script with working code, you may review it once and only once. Then you must respond. Do not keep reviewing. There is a strict time limit of 3 minutes. Once it is reached, your response will no longer be accepted! Therefore I repeat, do not keep reviewing once you have working code!
"""


@dataclass
class LinkDiscoveryResult:
    # structured result for dataset download link discovery
    publication_url: str
    download_links: list[str]
    notes: str
    raw_response: str

    def save(self) -> Path:
        target = LINKS_PATH
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


def discover_links(publication_url: str) -> LinkDiscoveryResult:
    prompt_with_link = (
        f"{LINK_PROMPT}\n\nHere is the link to the publication: {publication_url}"
    )
    raw_response = send_message(prompt_with_link, timeout=180, agent="dataset-links")
    result = parse_link_response(raw_response, publication_url)
    return result


def parse_link_response(raw_response: str, publication_url: str) -> LinkDiscoveryResult:
    download_links, notes = parse_links_from_json(raw_response)
    return LinkDiscoveryResult(
        publication_url=publication_url,
        download_links=download_links,
        notes=notes,
        raw_response=raw_response,
    )


def parse_links_from_json(raw_response: str) -> tuple[list[str], str]:
    payload = coerce_json_payload(raw_response)
    links = [str(link).strip() for link in payload.get("download_links", [])]
    notes = str(payload.get("notes", ""))
    return links, notes


def coerce_json_payload(raw_response: str) -> dict:
    cleaned = raw_response.strip()
    if cleaned:
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            print(raw_response)
            raise ValueError("Invalid JSON response from the LLM.") from exc
    print(raw_response)
    raise ValueError("Invalid JSON response from the LLM.")


def discover_and_download(
    publication_url: str,
) -> tuple[LinkDiscoveryResult, list[Path]]:
    discovery = discover_links(publication_url)
    discovery.save()
    if not discovery.download_links:
        if discovery.notes.strip():
            print(f"Notes: {discovery.notes}")
        raise ValueError("No download links found")

    downloaded_paths = [Path(p) for p in discovery.download_links]

    if not downloaded_paths:
        raise ValueError("Agent returned download_links but not downloaded_files.")

    return discovery, downloaded_paths


# script generation


def generate_processing_script() -> Path:
    """
    Generates a dataset processing script for the first downloaded dataset.

    :returns: Path to the generated script
    :raises FileNotFoundError: If no downloaded file exists
    """
    dataset_path = get_downloaded_file()
    return generate_processing_script_for_dataset(dataset_path)


def generate_processing_script_for_dataset(
    dataset_path: Path,
    target_dir: Path | None = None,
) -> Path:
    """
    Generates a dataset processing script for the provided dataset path.

    :param dataset_path: Path to the dataset file or archive
    :param target_dir: Optional target directory for generated scripts
    :returns: Path to the generated script
    :raises FileNotFoundError: If the dataset path does not exist
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    prompt = SCRIPT_GENERATION_PROMPT.format(
        dataset_path=dataset_path,
        example_scripts_dir=EXAMPLE_SCRIPTS_DIR,
        base_template_path=BASE_TEMPLATE_PATH,
    )
    response = send_message(prompt, 180)

    output_dir = target_dir or SCRIPTS_ARTIFACTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    target_name = f"{dataset_path.stem}_generated.py"
    target_path = output_dir / target_name
    with target_path.open("w", encoding="utf-8") as f:
        f.write(response)

    return target_path


def get_downloaded_file() -> Path:
    """
    Returns the first downloaded file from the downloads artifacts folder.

    :returns: Path to the downloaded file
    :raises FileNotFoundError: If no files exist in the downloads folder
    """
    # returns first file in folder; for now assumes there is only one
    for path in sorted(DOWNLOADS_DIR.rglob("*")):
        if path.is_file():
            return path
    raise FileNotFoundError(f"No files in {DOWNLOADS_DIR}")
