import hashlib

# based on: https://stackoverflow.com/a/3431838/10550998 by quantumSoup
# License: CC-BY-SA


def sha256sum(path):
    hash = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            hash.update(chunk)
    return hash.hexdigest()
