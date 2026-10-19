"""Every number in the written sections must come from paper/radiology/numbers.md (or from the
design, which Methods states). Prints each number that appears in neither, with its context.

  python experiments/check_numbers.py [section ...]     # default: results discussion abstract conclusions highlights
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAD = ROOT / "paper" / "radiology"
NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w])")


def numbers(text):
    return set(NUM.findall(text))


def main():
    secs = sys.argv[1:] or ["results", "discussion", "abstract", "conclusions", "highlights"]
    source = numbers((RAD / "numbers.md").read_text()) | numbers((RAD / "sections" / "methods.tex").read_text())
    # design constants and counts every section may state
    source |= {str(i) for i in range(0, 13)} | {"95", "100", "250", "200", "400", "1024", "256", "128", "120"}
    bad = 0
    for sec in secs:
        p = RAD / "sections" / f"{sec}.tex"
        if not p.exists():
            continue
        text = re.sub(r"(?m)^%.*$", "", p.read_text())
        text = re.sub(r"\\(label|ref|cite|includegraphics)(\[[^]]*\])?\{[^}]*\}", "", text)
        for m in NUM.finditer(text):
            v = m.group(0)
            if v in source:
                continue
            bad += 1
            ctx = text[max(0, m.start() - 60):m.end() + 30].replace("\n", " ")
            print(f"[{sec}] {v:>8s}  ...{ctx}...")
    print(f"{bad} number(s) not found in numbers.md or methods.tex")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
