def clean_data(df, tokenizer, fine_tune):
    old_df = df.copy()

    df[0] = df[0].apply(lambda x: x.replace(" ( [ link ] )", "")
                        .replace("[ link ]", "<link>" if fine_tune else tokenizer.unk_token)
                        .replace("([ link])", "")
                        .replace("[ link]a", "<link>" if fine_tune else tokenizer.unk_token)
                        .replace("[ link]b", "<link>" if fine_tune else tokenizer.unk_token)
                        .replace("[link]b", "<link>" if fine_tune else tokenizer.unk_token)
                        .replace(" ([ link])", "")
                        .replace("([link])", "")
                        .replace("([link], c)", "")
                        .replace("( [ link])", "")
                        .replace("[ link]", "<link>" if fine_tune else tokenizer.unk_token))

    # remove http
    df[0] = df[0].apply(lambda x: re.sub(r"https?:.+(\)|/|(\.pdf)|(\.PDF)|(\.html)|#| - U |aspx?|-[a-zA-z0-9]+|\.htm|\?.+)", "", x))
    df[0] = df[0].apply(lambda x: re.sub(r"https?:.+(\)|/|( \.))", ".", x))
    df[0] = df[0].apply(lambda x: re.sub(r"www.+?( |\))", "", x))

    # remove size
    df[0] = df[0].apply(lambda x: re.sub(r"size .+?{ }", "", x))
    df[0] = df[0].apply(lambda x: re.sub(r"size .+?}", "", x))
    df[0] = df[0].apply(lambda x: re.sub(r"size .+?\. \"", "", x))

    # correct punctuations
    df[0] = df[0].apply(lambda x: x.replace(" ,", ",")
                        .replace(" .", ".")
                        .replace("( ", "(")
                        .replace(" )", ")")
                        .replace("$ ", "$")
                        .replace(" ;", ";")
                        .replace(" :", ":")
                        .replace(" ?", "?")
                        .replace(" ’", "’")
                        .replace("“ ", "“")
                        .replace(" ”", "”")
                        .replace(" - ", "-")
                        .replace(" / ", "/")
                        .replace(" n’t", "n’t")
                        .replace(" '", "'")
                        .replace("_ ", "")
                        .replace(",”", "”")
                        .replace(",’", "’")
                        .replace("‘ ", "‘")
                        .replace(" %", "%")
                        .replace(" + ", "+")
                        .replace(" = ", "=")
                        .replace(" – ", "-"))

    # remove number
    df[0] = df[0].apply(lambda x: re.sub(r"^ \d+\. ", "", x))
    df[0] = df[0].apply(lambda x: re.sub(r"^ \d+\.", ".", x))

    # remove space
    df[0] = df[0].apply(lambda x: x[1:] if x[0] == " " else x)

    # remove (a?b?c?)
    df[0] = df[0].apply(lambda x: re.sub(r"\((a|b|c)\)", "", x))

    # remove (Source: )
    df[0] = df[0].apply(lambda x: re.sub(r"\(Source:.+\) ", "", x))

    # filter { and }
    df[0] = df[0].apply(lambda x: x.replace("{", "").replace("}", ""))

    # remove double space
    df[0] = df[0].apply(lambda x: x.replace("  ", " ").replace("  ", " ").replace("  ", " "))

    # remove again link
    df[0] = df[0].apply(lambda x: x.replace("([link])", ""))

    # remove again punctuation
    df[0] = df[0].apply(lambda x: x.replace(". .", ".")
                                   .replace("..", ".")
                                   .replace(" .", ".")
                                   .replace(",.", ".")
                                   .replace(",  ", "."))

    df[0] = df[0].apply(lambda x: re.sub(r"^\d+\.", ".", x))

    # remove stupid quotes
    df[0] = df[0].apply(lambda x: x.replace("“", "\"").replace("”", "\"").replace("’", "'").replace("‘", "'"))

    # remove stupid greek letters
    greek_alphabet = 'αβγδεζηθικλμνξοπρςστυφχψ'
    greek_alphabet += greek_alphabet.upper()

    for letter in greek_alphabet:
        df[0] = df[0].apply(lambda x: x.replace(letter, tokenizer.unk_token))

    # remove equations
    df[0] = df[0].apply(lambda x: re.sub(r" (([a-zA-Z]|(\d)+)(\+|-|=|⋅))+([a-z[A-Z]|(\d)+) ", " <equation> " if fine_tune else tokenizer.unk_token, x))

    # for row1, row2, col1, col2 in zip(df[0], old_df[0], df[1], old_df[1]):
    #     print(row1, col1)
    #     print(row2, col2)
    #     print()
