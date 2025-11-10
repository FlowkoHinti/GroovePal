import random
from pathlib import Path
import json
from GroovePal.Configs import BASE_PATH


def read_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    eval_path = Path(BASE_PATH) / "Data" / "eval"
    files = [eval_path / "test_chunk_1.jsonl", eval_path / "validation_chunk_1.jsonl"]

    # Combine both files
    dnas = []
    for file in files:
        dnas.extend(read_jsonl(file))

    print("Original length:", len(dnas))

    # filtering
    dnas = [
        dna for dna in dnas
        if "foundational" not in dna["DNA_ID"] and len(dna["DNAUnits"]) > 16
    ]

    print("New length:", len(dnas))

    # Randomly sample
    k = 300
    eval_set = random.sample(dnas, k)

    # Save eval set
    with open(eval_path / "eval_chunk.jsonl", 'w', encoding='utf-8') as f:
        for dna in eval_set:
            f.write(json.dumps(dna) + '\n')

    print(f"Eval set created with {k} samples.")


if __name__ == "__main__":
    main()
