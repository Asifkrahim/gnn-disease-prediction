
from html.parser import HTMLParser
from pathlib import Path
import requests

BASE_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/opentargets/"
    "platform/26.09/output/"
)

DATASETS = ["clinical_indication", "clinical_target"]
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "opentargets"


class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)


def download_dataset(dataset):
    url = BASE_URL + dataset + "/"
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    parser = LinkParser()
    parser.feed(response.text)

    files = [
        link for link in parser.links
        if link.endswith(".parquet")
        and "/" not in link
    ]

    if not files:
        raise RuntimeError(
            f"No Parquet files found at {url}. "
            "Check the release and directory listing."
        )

    destination = OUTPUT_DIR / dataset
    destination.mkdir(parents=True, exist_ok=True)

    print(f"\n{dataset}: {len(files)} file(s) found")

    for filename in files:
        path = destination / filename

        if path.exists() and path.stat().st_size > 0:
            print(f"Already downloaded: {filename}")
            continue

        file_url = url + filename
        print(f"Downloading {filename}...")

        with requests.get(
            file_url, stream=True, timeout=(20, 180)
        ) as download:
            download.raise_for_status()
            with path.open("wb") as output:
                for chunk in download.iter_content(1024 * 1024):
                    if chunk:
                        output.write(chunk)

        print(f"Saved: {path}")

    print(f"Completed: {dataset}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for dataset in DATASETS:
        download_dataset(dataset)

    print("\nDataset downloads completed.")


if __name__ == "__main__":
    main()
