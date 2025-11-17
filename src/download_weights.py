import argparse
import requests
import os

def download(url, out_path):
    if os.path.exists(out_path):
        print("Destination already exists:", out_path)
        return
    r = requests.get(url, stream=True)
    r.raise_for_status()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("Downloaded model to", out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="Direct URL to the model file (raw link)")
    parser.add_argument("--dest", default="models/best_model.pt", help="Destination path")
    args = parser.parse_args()
    download(args.url, args.dest)

