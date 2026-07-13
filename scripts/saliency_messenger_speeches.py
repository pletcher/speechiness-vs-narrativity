"""Token-level saliency for a fine-tuned sentence classifier.

For each input sentence, prints the model's probability for a target label
and a per-token gradient-saliency score for that label's logit, so you can
inspect *which* tokens are driving a high score. If the salient tokens are
mostly Homeric-dialect spellings / proper names, the model is likely acting
as a dialect detector; if they're independent of dialect-specific spelling
(particles, clause-initial patterns, epithet-like modifiers, tense/person
markers), that's more consistent with a genuine register/discourse signal.

Usage:
    # epic/tragedy classifier (default), scoring P(epic)
    python scripts/saliency_messenger_speeches.py --text "..." --text "..."

    # a different model + target label, e.g. the epic-trained
    # speech/narrative classifier applied to tragedy, scoring P(narrative)
    python scripts/saliency_messenger_speeches.py \\
        --model-dir /path/to/speech-narrative-model \\
        --target-label narrative \\
        --file path/to/sentences.txt
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT_DIR = Path(__file__).parent.parent
DEFAULT_MODEL_DIR = (
    ROOT_DIR
    / "model_output"
    / "grc-epic-tragedy-sentence-classification-no-nestor-no-apologoi-no-soph-electra"
)
DEFAULT_TARGET_LABEL = "epic"


def load_sentences(args: argparse.Namespace) -> list[str]:
    sentences = list(args.text or [])
    if args.file:
        with open(args.file, encoding="utf-8") as f:
            sentences.extend(line.strip() for line in f if line.strip())
    if not sentences:
        raise SystemExit("Provide at least one sentence via --text or --file")
    return sentences


def saliency_for_sentence(model, tokenizer, device, text: str, target_label_id: int):
    inputs = tokenizer(text, truncation=True, max_length=512, return_tensors="pt").to(
        device
    )
    input_ids = inputs["input_ids"]

    embedding_layer = model.get_input_embeddings()
    inputs_embeds = embedding_layer(input_ids).detach().clone()
    inputs_embeds.requires_grad_(True)

    outputs = model(
        inputs_embeds=inputs_embeds, attention_mask=inputs["attention_mask"]
    )
    logits = outputs.logits[0]
    probs = torch.softmax(logits, dim=-1)
    target_prob = probs[target_label_id].item()

    model.zero_grad()
    logits[target_label_id].backward()

    # Per-token saliency: L2 norm of the target logit's gradient w.r.t. that
    # token's input embedding ("simple gradients" attribution).
    saliency = inputs_embeds.grad[0].norm(dim=-1)
    saliency = (saliency / saliency.max()).tolist()

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    return target_prob, list(zip(tokens, saliency))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", action="append", help="A sentence to score")
    parser.add_argument("--file", type=Path, help="Path to a file, one sentence per line")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Path to a fine-tuned sequence classification model directory",
    )
    parser.add_argument(
        "--target-label",
        default=DEFAULT_TARGET_LABEL,
        help="Label name to compute saliency for (must be in the model's id2label)",
    )
    args = parser.parse_args()

    sentences = load_sentences(args)

    device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(args.model_dir))
    model.to(device)
    model.eval()

    label2id = {label: i for i, label in model.config.id2label.items()}
    if args.target_label not in label2id:
        raise SystemExit(
            f"'{args.target_label}' not in model labels: {list(label2id)}"
        )
    target_label_id = label2id[args.target_label]

    for text in sentences:
        target_prob, token_scores = saliency_for_sentence(
            model, tokenizer, device, text, target_label_id
        )
        print(f"\n{text}")
        print(f"P({args.target_label}) = {target_prob:.4f}")
        ranked = sorted(token_scores, key=lambda ts: ts[1], reverse=True)
        top = ", ".join(f"{tok}({score:.2f})" for tok, score in ranked[:10])
        print(f"Top salient tokens: {top}")


if __name__ == "__main__":
    main()
