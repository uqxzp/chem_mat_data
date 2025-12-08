import mimetypes
import os
import re

import requests


def download_file(url, output_folder):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # file name
        filename = None
        if "Content-Disposition" in response.headers:
            content_disposition = response.headers["Content-Disposition"]
            filenames = re.findall('filename="?([^"]+)"?', content_disposition)
            if filenames:
                filename = filenames[0]
        if not filename:
            filename = "downloaded_file"

        # ensure file extension
        name, ext = os.path.splitext(filename)
        if not ext:
            content_type = response.headers.get("Content-Type", "").split(";")[0]
            extension = mimetypes.guess_extension(content_type)
            if extension:
                filename = f"{name}{extension}"

        # download file
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
