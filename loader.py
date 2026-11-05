import os
import subprocess
from langchain_core.documents import Document

SKIP_NAMES = {"package-lock.json", "yarn.lock", "LICENSE", "NOTICE","HISTORY.md"}
SKIP_EXT = {".key", ".cnf", ".csr", ".pem", ".crt", ".srl", ".typed"}

def load_repo(repo_path, max_bytes=100_000):
    # Step 1: ask git for the list of real project files
    result = subprocess.run(
        ["git", "-C", repo_path, "ls-files"],
        capture_output=True, text=True, encoding="utf-8"
    )
    file_list = result.stdout.splitlines()

    documents = []
    for rel_path in file_list:
        name = os.path.basename(rel_path)
        full_path = os.path.join(repo_path, rel_path)

        # skip anything that isn't a normal file
        if not os.path.isfile(full_path):
            continue

        # Step 2: skip lock files
        if name in SKIP_NAMES or os.path.splitext(name)[1] in SKIP_EXT:
            continue

        # Step 4: skip huge files and minified files
        if os.path.getsize(full_path) > max_bytes or ".min." in name:
            continue

        # Step 3: skip anything that isn't readable text
        try:
            with open(full_path, encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            continue

        documents.append(Document(
            page_content=text,
            metadata={"source": rel_path, "file_type": os.path.splitext(name)[1]},
        ))
    return documents