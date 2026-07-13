"""Use the fine-tuned grc-homeric-speech-narrative-sentence-classification
model (no held-out _apologoi_ or _Iliad_ 11) to test what parts of tragedy
(if any) skew more narrative than others.

Hypothesis: Some messengers' speeches will light up the narrative side,
but so will speeches like Clytaemestra's "Beacon Speech" and Deianira's
report about the tuft of wool.

Since this model was trained only on Homer, the distinctions it learned
shouldn't be affected by dialect — but see saliency_messenger_speeches.py
for spot-checking what tokens are carrying the reported registers.

Uses the tragedies from the Daphne treebanks:

- Medea (skipped because it does not have line refs)
- IT
- Trachiniae
- Antigone
- Ajax
- OT
- Sophocles' Electra
- Philoctetes
- Prometheus Bound
- Agamemnon
- Choephoroi
- Eumenides

Treebanks are available here: https://github.com/francescomambrini/Daphne/tree/master

Uses hand-curated list of messenger speech spans (messenger_speeches.json). In disputed
cases (_Trachiniae_, _Philoctetes_), other authorities were consulted to determine whether or not
to mark a speech as a messenger's speech.
"""

import json
import re
from pathlib import Path

import pandas as pd
import torch

from scipy.stats import fisher_exact, mannwhitneyu, wilcoxon
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT_DIR = Path(__file__).parent.parent
CONLLU_DIR = ROOT_DIR / "conllu"
MESSENGER_SPEECHES_JSON = ROOT_DIR / "messenger_speeches.json"
MODEL = "pletcher/grc-homeric-speech-narrative-sentence-classification"
OUT_CSV = ROOT_DIR / "csv" / "messenger_speech_narrativity.csv"

NARRATIVE_LABEL_ID = 0
BATCH_SIZE = 32

REF_RE = re.compile(r"Ref=(\d+)(?:\.(\d+))?")


def urn_to_stem(urn: str) -> str:
    return urn.removeprefix("urn:cts:greekLit:")


def parse_conllu_sentences(path: Path):
    """Yield (sent_id, text, min_line, max_line) per sentence.

    Returns None for the whole file if it carries no per-token line
    references at all (Ref=N in the MISC column). Tragedy Ref values are
    sometimes dotted (e.g. Ref=161.1, for verse lines split between
    speakers) -- only the integer line number before the dot is used.
    """
    sentences = []
    any_ref_in_file = False
    sent_id = None
    text = None
    lines_seen: list[int] = []

    def flush():
        if sent_id is not None and text is not None and lines_seen:
            sentences.append((sent_id, text, min(lines_seen), max(lines_seen)))

    with open(path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if line.startswith("# sent_id ="):
                flush()
                sent_id = line.split("=", 1)[1].strip()
                text = None
                lines_seen = []
            elif line.startswith("# text ="):
                text = line.split("=", 1)[1].strip()
            elif line and not line.startswith("#"):
                fields = line.split("\t")
                if "." in fields[0] or "-" in fields[0]:
                    continue  # empty node / multiword token
                match = REF_RE.search(fields[-1])
                if match is None:
                    continue
                line_num, _sub_line = match.groups()
                any_ref_in_file = True
                lines_seen.append(int(line_num))
    flush()

    return sentences if any_ref_in_file else None


def load_messenger_speeches() -> list[dict]:
    with open(MESSENGER_SPEECHES_JSON, encoding="utf-8") as f:
        return json.load(f)


def build_sentence_table() -> pd.DataFrame:
    plays = load_messenger_speeches()
    speech_ranges = {urn_to_stem(p["urn"]): p["speeches"] for p in plays}

    rows = []
    skipped = []
    for urn_stem, ranges in speech_ranges.items():
        conllu_path = CONLLU_DIR / f"{urn_stem}.daphne_tb-grc1.conllu"
        if not conllu_path.exists():
            continue

        sentences = parse_conllu_sentences(conllu_path)
        if sentences is None:
            skipped.append(urn_stem)
            continue

        for sent_id, text, min_line, max_line in sentences:
            is_messenger = any(
                min_line <= end and max_line >= start for start, end in ranges
            )
            rows.append(
                {
                    "play": urn_stem,
                    "sent_id": sent_id,
                    "text": text,
                    "is_messenger_speech": is_messenger,
                    # take the "average" line for the speech
                    # for the purposes of plotting
                    "line": (min_line + max_line) / 2,
                }
            )

    if skipped:
        print(f"Skipped plays with no usable line references: {skipped}")

    return pd.DataFrame(rows)


def classify(df: pd.DataFrame) -> pd.DataFrame:
    device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    model.to(device)
    model.eval()

    p_narrative = []
    pred_label = []

    with torch.no_grad():
        for start in range(0, len(df), BATCH_SIZE):
            batch_texts = df["text"].iloc[start : start + BATCH_SIZE].tolist()
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            p_narrative.extend(probs[:, NARRATIVE_LABEL_ID].cpu().tolist())
            pred_label.extend(probs.argmax(dim=-1).cpu().tolist())

    df = df.copy()
    df["p_narrative"] = p_narrative
    df["pred_narrative"] = [label == NARRATIVE_LABEL_ID for label in pred_label]
    return df


def report(df: pd.DataFrame) -> None:
    messenger = df[df["is_messenger_speech"]]
    other = df[~df["is_messenger_speech"]]

    print(
        f"\nMessenger-speech sentences: {len(messenger)}; "
        f"other tragedy sentences: {len(other)}"
    )
    print(
        f"P(narrative) mean: messenger={messenger['p_narrative'].mean():.4f}, "
        f"other={other['p_narrative'].mean():.4f}"
    )
    print(
        f"Narrative classification rate: "
        f"messenger={messenger['pred_narrative'].mean():.4f}, "
        f"other={other['pred_narrative'].mean():.4f}"
    )

    # Mann-Whitney U test tests whether we can reject the null hypothesis
    # that two groups are the same
    u_stat, u_p = mannwhitneyu(
        messenger["p_narrative"], other["p_narrative"], alternative="greater"
    )
    print(
        f"\nMann-Whitney U (H1: messenger P(narrative) > other): "
        f"U={u_stat:.1f}, p={u_p:.4g}"
    )

    table = [
        [messenger["pred_narrative"].sum(), (~messenger["pred_narrative"]).sum()],
        [other["pred_narrative"].sum(), (~other["pred_narrative"]).sum()],
    ]

    # Fisher's exact test gives us a p-value for testing the association
    # between messengers' narrative rates and the rates of other speakers.
    odds_ratio, fisher_p = fisher_exact(table, alternative="greater")
    print(
        f"Fisher's exact (H1: messenger narrative rate > other): "
        f"odds_ratio={odds_ratio:.4f}, p={fisher_p:.4g}"
    )

    print("\nPer-play P(narrative) mean, messenger vs. other:")
    per_play = df.groupby(["play", "is_messenger_speech"])["p_narrative"].mean()
    per_play = per_play.unstack("is_messenger_speech").rename(
        columns={False: "other", True: "messenger"}
    )
    print(per_play)

    # Play-stratified check: the pooled tests above conflate the
    # messenger-speech effect with play-level baseline differences (the
    # "other" mean alone ranges from 0.02 to 0.12 across plays, and
    # messenger-speech sentence counts aren't evenly distributed across
    # plays). A Wilcoxon signed-rank test on the per-play paired differences
    # tests whether the effect holds up play-by-play rather than because a
    # few high-baseline or messenger-heavy plays dominate the pooled sample.
    per_play_diff = per_play["messenger"] - per_play["other"]
    print("\nPer-play (messenger - other) P(narrative) difference:")
    print(per_play_diff)
    w_stat, w_p = wilcoxon(per_play_diff, alternative="greater")
    print(
        f"\nWilcoxon signed-rank across {len(per_play_diff)} plays "
        f"(H1: messenger - other > 0): W={w_stat:.1f}, p={w_p:.4g}"
    )


def main():
    df = build_sentence_table()
    print(
        f"Loaded {len(df)} sentences from {df['play'].nunique()} plays "
        f"({int(df['is_messenger_speech'].sum())} in messenger speeches)"
    )

    df = classify(df)

    OUT_CSV.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote per-sentence results to {OUT_CSV}")

    report(df)


if __name__ == "__main__":
    main()
