from collections import Counter
from loader import load_repo

docs = load_repo("requests")
print(len(docs), "files loaded")

# how many files of each type?
counts = Counter(d.metadata["file_type"] for d in docs)
print(counts.most_common())

# which source files came from src/?
for d in docs:
    if d.metadata["source"].startswith("src/"):
        print(d.metadata["source"])