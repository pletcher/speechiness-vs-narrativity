import json
import re

from pathlib import Path

from lxml import etree  # ty:ignore[unresolved-import]

ROOT_DIR = Path(__file__).parent.parent
JSON_DIR = ROOT_DIR / "json"
XML_DIR = ROOT_DIR / "xml"

NS = {"tei": "http://www.tei-c.org/ns/1.0"}

SENTENCE_SPLIT_RE = re.compile(r"[.·;]")


def lines_to_sentences(lines):
    first_line = lines[0][0]
    last_line = lines[-1][0]

    sentences = []
    buffer = ""
    buffer_start_line = first_line

    for line_n, text in lines:
        buffer = f"{buffer} {text}".strip() if buffer else text

        # everything but the last piece is a complete sentence, terminated
        # somewhere within this line; the last piece continues (or ends,
        # if this line ends with a delimiter) onto the next line
        parts = re.split(SENTENCE_SPLIT_RE, buffer)

        for part in parts[:-1]:
            cleaned = part.replace(",", "").strip()
            if cleaned:
                sentences.append(
                    {
                        "text": cleaned,
                        "first_line": buffer_start_line,
                        "last_line": line_n,
                    }
                )
            buffer_start_line = line_n

        buffer = parts[-1]

    # trailing text with no closing delimiter (e.g. speech ends mid-sentence)
    cleaned = buffer.replace(",", "").strip()
    if cleaned:
        sentences.append(
            {
                "text": cleaned,
                "first_line": buffer_start_line,
                "last_line": last_line,
            }
        )

    return {"first_line": first_line, "last_line": last_line, "sentences": sentences}


def read_xml_files():
    for f in XML_DIR.glob("*.xml"):
        tree = etree.parse(f)

        output = {"urn": f.stem, "sentences": []}

        for sp in tree.iterfind(".//tei:sp", namespaces=NS):
            speaker_name = sp.xpath("./tei:speaker/text()", namespaces=NS)

            if len(speaker_name) == 0:
                continue
            if all([s.strip() for s in speaker_name]) == "":
                continue

            speaker_name = speaker_name[0]
            lines = sp.xpath(".//tei:l", namespaces=NS)
            lines = [
                (
                    line.attrib.get("n"),
                    etree.tostring(line, method="text", encoding="unicode").strip(),
                )
                for line in lines
            ]

            sentences = lines_to_sentences(lines)

            output["sentences"].append(
                {"speaker_name": speaker_name, "sentences": sentences}
            )

        with (JSON_DIR / Path(f.name).with_suffix(".json")).open("w") as g:
            json.dump(output, g, ensure_ascii=False)


def main():
    read_xml_files()


if __name__ == "__main__":
    main()
