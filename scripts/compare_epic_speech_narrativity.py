import json
import re

from pathlib import Path

import conllu
import pandas as pd
import torch

from scipy.stats import fisher_exact, mannwhitneyu, wilcoxon
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT_DIR = Path(__file__).parent.parent
CONLLU_DIR = ROOT_DIR / "conllu"
ILIAD_SPEECHES_JSON = ROOT_DIR / "json" / "iliad_speeches.json"
ODYSSEY_SPEECHES_JSON = ROOT_DIR / "json" / "odyssey_speeches.json"
MODEL = "pletcher/grc-homeric-speech-narrative-sentence-classification"
OUT_CSV = ROOT_DIR / "csv" / "epic_speech_narrativity.csv"

NARRATIVE_LABEL_ID = 0
BATCH_SIZE = 32

REF_RE = re.compile(r"(\d+)(?:\.(\d+))?")


def urn_to_stem(urn: str) -> str:
    return urn.removeprefix("urn:cts:greekLit:")


def parse_conllu_sentences(path: Path):
    sentences = []
    sent_id = None
    text = None
    lines_seen: list[tuple[int, int]] = []

    def flush():
        if sent_id is not None and text is not None and lines_seen:
            sentences.append((sent_id, text, lines_seen[0], lines_seen[-1]))

    with open(path, encoding="utf-8") as f:
        conll_sentences = conllu.parse(f.read())

        for sent in conll_sentences:
            flush()
            sent_id = sent.metadata.get("sent_id")
            text = sent.metadata.get("text")
            lines_seen = []

            for tok in sent:
                misc = tok.get("misc")
                if misc:
                    ref = misc.get("Ref")
                    if ref:
                        book, line = ref.split(".")

                        if book and line:
                            lines_seen.append((int(book), int(line)))
    flush()

    return sentences


def book_line_to_ints(raw):
    for item in raw:
        item["l_fi"] = [int(i) for i in item["l_fi"].split(".")]
        item["l_la"] = [int(i) for i in item["l_la"].split(".")]

    return raw


def load_iliad_speeches() -> list[dict]:
    with open(ILIAD_SPEECHES_JSON, encoding="utf-8") as f:
        return book_line_to_ints(json.load(f))


def load_odyssey_speeches() -> list[dict]:
    with open(ODYSSEY_SPEECHES_JSON, encoding="utf-8") as f:
        return book_line_to_ints(json.load(f))


def build_sentence_table() -> pd.DataFrame:
    iliad = load_iliad_speeches()
    odyssey = load_odyssey_speeches()

    rows = []

    iliad_sentences = parse_conllu_sentences(
        ROOT_DIR / "conllu" / "tlg0012.tlg001.daphne_tb-grc1.conllu"
    )
    odyssey_sentences = parse_conllu_sentences(
        ROOT_DIR / "conllu" / "tlg0012.tlg002.daphne_tb-grc1.conllu"
    )

    for sent_id, text, first_line, last_line in iliad_sentences:
        is_direct_speech = any(
            first_line[0] == speech["l_fi"][0]
            and first_line[1] >= speech["l_fi"][1]
            and last_line[0] == speech["l_la"][0]
            and last_line[1] <= speech["l_la"][1]
            for speech in iliad
        )

        rows.append(
            {
                "work": "iliad",
                "sent_id": sent_id,
                "text": text,
                "is_direct_speech": is_direct_speech,
                "book": first_line[0],
                # take the "average" line for the speech
                # for the purposes of plotting
                "line": (first_line[1] + last_line[1]) / 2,
            }
        )

    for sent_id, text, first_line, last_line in odyssey_sentences:
        record = next(
            (
                speech
                for speech in odyssey
                if first_line[0] == speech["l_fi"][0]
                and first_line[1] >= speech["l_fi"][1]
                and last_line[0] == speech["l_la"][0]
                and last_line[1] <= speech["l_la"][1]
            ),
            None,
        )

        is_direct_speech = bool(record)

        speaker_being = None
        speaker_disguise = None
        speaker_gender = None
        speaker_name = None
        speaker_number = None

        if record is not None:
            spkr = record.get("spkr", {})[0]
            speaker_name = spkr.get("name")
            speaker_number = spkr.get("number")
            speaker_being = spkr.get("being")
            speaker_gender = spkr.get("gender")
            speaker_disguise = spkr.get("disguise")

        rows.append(
            {
                "work": "odyssey",
                "sent_id": sent_id,
                "text": text,
                "is_direct_speech": is_direct_speech,
                "book": first_line[0],
                # take the "average" line for the speech
                # for the purposes of plotting
                "line": (first_line[1] + last_line[1]) / 2,
                "speaker_being": speaker_being,
                "speaker_disguise": speaker_disguise,
                "speaker_gender": speaker_gender,
                "speaker_name": speaker_name,
                "speaker_number": speaker_number,
            }
        )

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
    direct = df[df["is_direct_speech"]]
    other = df[~df["is_direct_speech"]]

    print(f"\ndirect speech sentences: {len(direct)}; other sentences: {len(other)}")
    print(
        f"P(narrative) mean: direct={direct['p_narrative'].mean():.4f}, "
        f"other={other['p_narrative'].mean():.4f}"
    )
    print(
        f"Narrative classification rate: "
        f"direct={direct['pred_narrative'].mean():.4f}, "
        f"other={other['pred_narrative'].mean():.4f}"
    )

    # Mann-Whitney U test tests whether we can reject the null hypothesis
    # that two groups are the same
    u_stat, u_p = mannwhitneyu(
        direct["p_narrative"], other["p_narrative"], alternative="greater"
    )
    print(
        f"\nMann-Whitney U (H1: direct P(narrative) > other): "
        f"U={u_stat:.1f}, p={u_p:.4g}"
    )

    table = [
        [direct["pred_narrative"].sum(), (~direct["pred_narrative"]).sum()],
        [other["pred_narrative"].sum(), (~other["pred_narrative"]).sum()],
    ]

    # Fisher's exact test gives us a p-value for testing the association
    # between directs' narrative rates and the rates of other speakers.
    odds_ratio, fisher_p = fisher_exact(table, alternative="greater")
    print(
        f"Fisher's exact (H1: direct narrative rate > other): "
        f"odds_ratio={odds_ratio:.4f}, p={fisher_p:.4g}"
    )

    print("\nPer-work P(narrative) mean, direct vs. other:")
    per_work = df.groupby(["work", "is_direct_speech"])["p_narrative"].mean()
    per_work = per_work.unstack("is_direct_speech").rename(
        columns={False: "other", True: "P(narrative)"}
    )
    print(per_work)

    print("\nPer-speaker P(narrative) mean, direct vs. other:")
    per_speaker = df.groupby(["speaker_name", "is_direct_speech"])["p_narrative"].mean()
    per_speaker = per_speaker.unstack("is_direct_speech").rename(
        columns={False: "other", True: "P(narrative)"}
    )
    print(per_speaker)
    per_speaker.sort_values("P(narrative)", ascending=False).to_csv(
        ROOT_DIR / "csv" / "epic_speakers_narrativity.csv"
    )

    print("\nBy-gender P(narrative) mean, direct vs. other:")
    by_gender = df.groupby(["speaker_gender", "is_direct_speech"])["p_narrative"].mean()
    by_gender = by_gender.unstack("is_direct_speech").rename(
        columns={False: "other", True: "P(narrative)"}
    )
    print(by_gender)
    by_gender.sort_values("P(narrative)", ascending=False).to_csv(
        ROOT_DIR / "csv" / "epic_gender_narrativity.csv"
    )

    per_work_diff = per_work["P(narrative)"] - per_work["other"]
    print("\nPer-work (direct - other) P(narrative) difference:")
    print(per_work_diff)
    w_stat, w_p = wilcoxon(per_work_diff, alternative="greater")
    print(
        f"\nWilcoxon signed-rank across {len(per_work_diff)} plays "
        f"(H1: direct - other > 0): W={w_stat:.1f}, p={w_p:.4g}"
    )


def main():
    df = build_sentence_table()
    print(
        f"Loaded {len(df)} sentences from {df['work'].nunique()} works "
        f"({int(df['is_direct_speech'].sum())} in direct speeches)"
    )

    df = classify(df)

    OUT_CSV.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote per-sentence results to {OUT_CSV}")

    report(df)


if __name__ == "__main__":
    main()
