_LATEX_SPECIAL_CHARS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def escape_latex(value: str | float | None) -> str:
    """Escape LaTeX special characters in candidate-supplied text.

    Resume text is untrusted input rendered directly into a .tex source
    file, so characters like `&`, `%`, or `\\` must be neutralized before
    compilation or they corrupt the document (or, unescaped backslashes
    followed by a command name, execute unintended LaTeX commands).
    """
    if value is None:
        return ""
    text = str(value)
    return "".join(_LATEX_SPECIAL_CHARS.get(char, char) for char in text)
