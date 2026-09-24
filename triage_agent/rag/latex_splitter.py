"""按 \\section 结构切分 LaTeX 源文本。"""

import re

SECTION_PATTERN = re.compile(r"\\(?:sub)*section\*?\{([^}]*)\}")


def split_latex_by_section(text: str) -> list[tuple[str, str]]:
    """输入整份 .tex 文本,输出 [(章节名, 章节内容), ...] 列表。"""
    matches = list(SECTION_PATTERN.finditer(text))

    if not matches:
        return [("(no section)", text)]

    sections: list[tuple[str, str]] = []

    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(("(preamble)", preamble))

    for i, m in enumerate(matches):
        title = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        sections.append((title, body))

    return sections