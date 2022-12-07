"""
    Some setnences in the Simple German dataset have encoding error.
    German "Umlaute" are not parsed correctly. This module fixes that.
"""


def text_replace(s, replacements):
    for k, v in replacements.items():
        s = s.replace(v, k)
    return s


def main():
    l = ["Ä", "ä", "Ö", "ö", "Ü", "ü", "ß"]
    replacements = {
        key: str(bytes(key, encoding="utf-8"), encoding="latin-1") for key in l}
    with open("../data/SimpleGerman/spacy_easy.txt") as fp:
        text = fp.read()
    text = text_replace(text, replacements)
    with open("../data/SimpleGerman/fixed_easy.txt", "w") as fp:
        fp.write(text)
    with open("../data/SimpleGerman/spacy_normal.txt") as fp:
        text = fp.read()
    text = text_replace(text, replacements)
    with open("../data/SimpleGerman/fixed_normal.txt", "w") as fp:
        fp.write(text)


if __name__ == "__main__":
    main()
