import json
import re

from pathlib import Path

import pandas as pd
import torch

from scipy.stats import fisher_exact, mannwhitneyu, wilcoxon
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT_DIR = Path(__file__).parent.parent
JSON_DIR = ROOT_DIR / "json"
MESSENGER_SPEECHES_JSON = ROOT_DIR / "json" / "messenger_speeches.json"
MODEL = "pletcher/grc-homeric-speech-narrative-sentence-classification"
OUT_CSV = ROOT_DIR / "csv" / "ALL_DRAMA_messenger_speech_narrativity.csv"

NARRATIVE_LABEL_ID = 0
BATCH_SIZE = 32

REF_RE = re.compile(r"(\d+)(?:\.(\d+))?")


def as_int(s):
    try:
        return int(s)
    except Exception:
        match = REF_RE.search(s)

        if match:
            line_num, _sub_line = match.groups()

            return int(line_num)

        print(f"NO MATCH FOUND FOR {s}")

        return 0


def urn_to_stem(urn: str) -> str:
    return urn.removeprefix("urn:cts:greekLit:")


def load_messenger_speeches() -> list[dict]:
    with open(MESSENGER_SPEECHES_JSON, encoding="utf-8") as f:
        return json.load(f)


def parse_json_sentences(json_path):
    with json_path.open() as f:
        records = json.load(f)

    return records["sentences"]


def build_sentence_table() -> pd.DataFrame:
    plays = load_messenger_speeches()
    speech_ranges = {urn_to_stem(p["urn"]): p["speeches"] for p in plays}

    rows = []
    for urn_stem, ranges in speech_ranges.items():
        json_path = JSON_DIR / f"{urn_stem}.perseus-grc2.json"
        if not json_path.exists():
            continue

        sentences = parse_json_sentences(json_path)

        sent_id = 1
        for sent in sentences:
            sents = sent["sentences"]

            for sentence in sents["sentences"]:
                first_line = as_int(sentence["first_line"])
                last_line = as_int(sentence["last_line"])
                # is_messenger = True if a messenger speech range contains the lines
                # or if the lines contain a messenger's speech
                is_messenger = any(
                    start <= first_line and end >= last_line for start, end in ranges
                ) or any(
                    first_line <= start and last_line >= end for start, end in ranges
                )

                rows.append(
                    {
                        "sent_id": f"{urn_stem}:{sent_id}",
                        "play": urn_stem,
                        "text": sentence["text"],
                        "is_messenger_speech": is_messenger,
                        # midpoint of the sentence's own line span, not the
                        # whole speech turn's
                        "line": (first_line + last_line) / 2,
                    }
                )

                sent_id += 1

    return pd.DataFrame(rows)


def classify(df: pd.DataFrame) -> pd.DataFrame:
    device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    if tokenizer is None:
        raise ValueError(f"`tokenizer` cannot be None. Check that {MODEL} exists.")

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
    per_play_diff = per_play["messenger"].fillna(0) - per_play["other"]
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
