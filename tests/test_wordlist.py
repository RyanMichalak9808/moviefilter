import pytest

from moviefilter.wordlist import Matcher, norm


def words(text, prob=0.95, low=()):
    return [{"word": w, "start": i * 0.5, "end": i * 0.5 + 0.4, "prob": 0.3 if w in low else prob}
            for i, w in enumerate(text.split())]


def found(text, levels=("strong", "slurs"), low=(), **kw):
    return [(d.text, d.level, d.kind) for d in Matcher(levels=levels, **kw).find(words(text, low=low))]


@pytest.mark.parametrize("w", ["fuck", "Fuck!", "fucking", "fuckin'", "FUUUCK", "motherfucker", "mother-fucker",
                               "shit,", "bullshit", "shitty", "bitch", "bitches", "asshole", "dumbass",
                               "bastard", "goddamn", "cunt", "dickhead", "pussy", "twat"])
def test_strong_words(w):
    assert [x[1] for x in found(f"you {w} there")] == ["strong"]


@pytest.mark.parametrize("w", ["hello", "shiitake", "class", "assume", "passed", "duck", "shut", "sheet", "hell",
                               "damn", "cocktail", "scunthorpe", "pitch", "Dickens", "bass", "glass", "bitcoin"])
def test_no_false_positives(w):
    assert found(f"the {w} is here") == []


def test_mild_and_religious_levels():
    t = "oh my God what the hell damn it Jesus"
    assert found(t) == []
    lv = {x[1] for x in found(t, levels=("mild",))}
    assert lv == {"mild"}
    kinds = {x[0] for x in found(t, levels=("religious",))}
    assert "oh my God" in kinds and "Jesus" in kinds


def test_phrases():
    res = found("you son of a bitch get out")
    assert ("son of a bitch", "strong", "phrase") in res
    assert ("piece of shit", "strong", "phrase") in found("what a piece of shit")


def test_masked_by_whisper():
    assert found("what the f*** man")[0][2] == "masked"
    assert found("oh s**t")[0][2] == "masked"


def test_fuzzy_only_when_unsure():
    m = Matcher()
    ws = words("well fuk you")
    assert [d.kind for d in m.find(ws)] == ["list"]          # "fuk" is in the list via f+u+c*k+
    ws = words("well biatch off", low=("biatch",))
    assert [d.flags for d in m.find(ws)] == [["fuzzy", "low-confidence"]]
    ws = words("well biatch off")                           # confident -> not fuzzy-matched
    assert m.find(ws) == []


def test_possible_name_flag():
    m = Matcher()
    d = m.find(words("I met Dick yesterday"))[0]
    assert "possible-name" in d.flags
    d = m.find(words("Dick, get out"))[0]                  # sentence start: not a name hint
    assert "possible-name" not in d.flags


def test_slurs_level_on_by_default_and_custom_words():
    assert found("you retard")[0][1] == "slurs"
    assert found("you twerp") == []
    m = Matcher(extra_words=["twerp"])
    assert m.find(words("you twerp"))[0].level == "custom"


def test_allow_overrides():
    m = Matcher(allow=["bastard"])
    assert m.find(words("you bastard")) == []


def test_other_languages():
    m = Matcher(languages=["es", "en"])
    assert {d.text for d in m.find(words("eres un pendejo y mierda fuck"))} == {"pendejo", "mierda", "fuck"}
    m = Matcher(languages=["de"])
    assert m.find(words("so eine Scheiße"))[0].text == "Scheiße"


def test_norm():
    assert norm("Fuck!") == "fuck"
    assert norm("“Shit,”") == "shit"
    assert norm("fuckin'") == "fuckin"


def test_context_marks_word():
    d = Matcher().find(words("one two three shit four five"))[0]
    assert "[shit]" in d.context


def test_unknown_level_rejected():
    with pytest.raises(ValueError):
        Matcher(levels=("nope",))
