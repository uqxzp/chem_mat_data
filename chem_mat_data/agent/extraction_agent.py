from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from chem_mat_data.agent.opencode_client import send_message
from chem_mat_data.agent.utils import download_file

# TODO: limit processing to smiles/xyz

ARTIFACTS_DIR: Path = Path(__file__).resolve().parent / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# link discovery
DOWNLOADS_DIR = ARTIFACTS_DIR / "downloads"
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
LINKS_PATH: Path = ARTIFACTS_DIR / "download_links.json"


LINK_PROMPT: str = """
You are a research assistant who specializes in chemistry and materials science datasets.
Given the URL of a publication, find the direct download URL(s) for the molecular dataset released with the paper.
Return only direct file or archive download links that trigger a download immediately in a browser (i.e., a link that starts a file download dialog).
Do NOT return publication pages, landing pages, dataset home pages, or supplementary information pages unless they are direct file downloads.
Look for supporting information, supplementary information and materials, GitHub/Zenodo/Figshare/OSF releases, institutional or lab repositories, or any other location that hosts the dataset.
If the dataset is available in multiple formats, prefer tabular formats such as .xlsx or .csv over other formats.
Ignore datasets that are PDB or CIF files; treat these cases as if no dataset was found.
Do not ask for permissions or mention access limitations. You are allowed to fetch any needed pages.
When you need to read a webpage (papers, publisher pages, Figshare/Zenodo/OSF/GitHub, supplemental pages), always use the `crawlfetch` tool instead of `webfetch`.
If you found the dataset links, leave "notes" as an empty string.
If you didn't find dataset links, respond with an empty list for "download_links" and explain in 1-2 sentences in "notes" what you checked.

Return a pure JSON object with this shape and nothing else:
{
  "download_links": ["https://direct-download-1", "..."],
  "notes": "..."
}
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
    prompt_with_link = f"{LINK_PROMPT}\n\Here is the link to the publication: {publication_url}"
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
    """
    Returns the first JSON object found in the LLM response.

    :param raw_response: raw LLM response string
    :returns: Parsed JSON object
    :raises ValueError: If no JSON object can be parsed
    """
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

    downloaded_paths: list[Path] = []
    for link in discovery.download_links:
        filename = Path(urlparse(link).path).name or "downloaded_dataset"
        target_path = DOWNLOADS_DIR / filename
        download_file(link, str(target_path))
        downloaded_paths.append(target_path)

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

def generate_processing_script_for_dataset(dataset_path: Path,
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
