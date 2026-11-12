"""Freeze exact rendered-token-length retrieval prompts, without model weights."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_ids(tokenizer, text):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=True,
        add_generation_prompt=True, return_dict=False,
    )


def make_prompt(tokenizer, tokens, index, seed):
    randomizer = random.Random(seed + index * 1000003 + tokens)
    prefix = (
        f"Request {index:03d}. Read the following document of key/value records. "
        "Write a detailed summary of its records and quote the value of key_00000.\n"
    )
    # ASCII records keep token truncation valid UTF-8; an overlong document is
    # deterministically trimmed, then the actual chat-template length checked.
    lines = [prefix]
    for i in range(tokens // 10 + 100):
        value = " ".join(randomizer.choice((
            "amber", "river", "forest", "orbit", "silver", "stone", "valley", "cloud"
        )) for _ in range(8))
        lines.append(f'{{"key_{i:05d}": "{value}"}}\n')
    raw_ids = tokenizer.encode("".join(lines), add_special_tokens=False)
    overhead = len(render_ids(tokenizer, ""))
    count = tokens - overhead
    seen = set()
    for _ in range(64):
        text = tokenizer.decode(raw_ids[:count], skip_special_tokens=False)
        actual = len(render_ids(tokenizer, text))
        if actual == tokens:
            return text
        if count in seen:
            # Resolve rare BPE boundary cycles with a separately verified suffix.
            for padding in range(1, 33):
                candidate = text + " x" * padding
                if len(render_ids(tokenizer, candidate)) == tokens:
                    return candidate
            raise ValueError(f"Cannot construct exact prompt: {tokens}/{index}")
        seen.add(count)
        count += tokens - actual
    raise ValueError("Prompt token count did not converge")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    config = json.loads(args.config.read_text())
    args.output.mkdir(exist_ok=False, parents=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    datasets = []
    for length in config["input_tokens"]:
        path = args.output / f"input-{length}.jsonl"
        records = []
        with path.open("x") as stream:
            for index in range(128):
                text = make_prompt(tokenizer, length, index, config["seed"])
                ids = render_ids(tokenizer, text)
                assert len(ids) == length
                stream.write(json.dumps({"text": text, "output_length": 512}) + "\n")
                records.append({
                    "index": index, "input_tokens": len(ids),
                    "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "rendered_ids_sha256": hashlib.sha256(
                        json.dumps(ids, separators=(",", ":")).encode()
                    ).hexdigest(),
                })
                if index % 16 == 0:
                    print(f"length={length} prepared={index + 1}/128", flush=True)
        datasets.append({"file": path.name, "input_tokens": length,
                         "sha256": digest(path), "records": records})
    manifest = {
        "model_repository": config["model_repository"],
        "model_revision": config["model_revision"], "seed": config["seed"],
        "source_sha256": digest(Path(__file__)), "config_sha256": digest(args.config),
        "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        "tokenizer_files": {p.name: digest(p) for p in args.model.glob("*token*") if p.is_file()},
        "packages": {n: importlib.metadata.version(n) for n in ("transformers", "tokenizers", "huggingface-hub", "jinja2")},
        "datasets": datasets,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
