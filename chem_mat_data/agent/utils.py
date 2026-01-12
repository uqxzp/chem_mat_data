import mimetypes
import os
import re
from urllib.parse import parse_qs, unquote, urlparse

import requests


def download_file(url, output_folder):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        filename = None
        if "Content-Disposition" in response.headers:
            content_disposition = response.headers["Content-Disposition"]
            filenames = re.findall('filename="?([^"]+)"?', content_disposition)
            if filenames:
                filename = filenames[0]

        parsed_url = urlparse(url)
        url_basename = os.path.basename(unquote(parsed_url.path))
        if not filename:
            filename = url_basename or "downloaded_file"

        name, ext = os.path.splitext(filename)
        if not ext:
            url_name, url_ext = os.path.splitext(url_basename)
            if url_ext:
                filename = f"{name or url_name}{url_ext}"
            else:
                query = parse_qs(parsed_url.query)
                for key in ("filename", "file", "name"):
                    if key in query and query[key]:
                        candidate = os.path.basename(unquote(query[key][0]))
                        candidate_name, candidate_ext = os.path.splitext(candidate)
                        if candidate_ext:
                            filename = f"{name or candidate_name}{candidate_ext}"
                            break

        name, ext = os.path.splitext(filename)
        if not ext:
            content_type = response.headers.get("Content-Type", "").split(";")[0]
            extension = mimetypes.guess_extension(content_type)
            if extension:
                filename = f"{name}{extension}"

        full_output_path = os.path.join(output_folder, filename)
        os.makedirs(output_folder, exist_ok=True)

        with open(full_output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"Saved file to {full_output_path}")
        return full_output_path

    except Exception as e:
        print(f"Error: {e}")
        return None
