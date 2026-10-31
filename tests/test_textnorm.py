"""Check textnorm on small examples where the right answer is known."""
from src.textnorm import clean_text, formatted_text, timed_tokens, verbatim_tokens

def test_verbatim_tokens():
    assert verbatim_tokens("So, umm, I love the colour!") == ["so", "um", "i", "love", "the", "colour"]
    assert verbatim_tokens("So, uhm, I love the colour!") == ["so", "um", "i", "love", "the", "colour"]
    assert verbatim_tokens("So, uh, I love the colour!") == ["so", "uh", "i", "love", "the", "colour"]
    assert verbatim_tokens("So, hmm, I love the colour!") == ["so", "hmm", "i", "love", "the", "colour"]
    assert verbatim_tokens("So, hmmm, I love the colour!") == ["so", "hmm", "i", "love", "the", "colour"]
    assert verbatim_tokens("So, mm, I love the colour!") == ["so", "hmm", "i", "love", "the", "colour"]

def test_verbatim_tokens_with_apostrophes():
    assert verbatim_tokens("It's a good idea.") == ["it's", "a", "good", "idea"]
    assert verbatim_tokens("I don't know.") == ["i", "don't", "know"]

def test_clean_text():
    assert clean_text("So, umm, I love the colour!") == "so i love the color"
    assert clean_text("So, uhm, I love the colour!") == "so i love the color"
    assert clean_text("So, uh, I love the colour!") == "so i love the color"
    assert clean_text("So, hmm, I love the colour!") == "so i love the color"
    assert clean_text("So, hmmm, I love the colour!") == "so i love the color"
    assert clean_text("So, mm, I love the colour!") == "so i love the color"

def test_formatted_text():
    assert formatted_text("So, umm, I love the colour!") == "So, umm, I love the colour!"
    assert formatted_text("So, uhm, I love the colour!") == "So, uhm, I love the colour!"
    assert formatted_text("So, uh, I love the colour!") == "So, uh, I love the colour!"
    assert formatted_text("So, hmm, I love the colour!") == "So, hmm, I love the colour!"
    assert formatted_text("So, hmmm, I love the colour!") == "So, hmmm, I love the colour!"
    assert formatted_text("So, mm, I love the colour!") == "So, mm, I love the colour!"

def test_formatted_keeps_punctuation_and_capitals():
    assert formatted_text("Hello, World!") == "Hello, World!"
    assert formatted_text("This is a test.") == "This is a test."
    assert formatted_text("Python is great!") == "Python is great!"
    assert formatted_text("So, um…\nI  know!") == "So, um… I know!"

def test_timed_tokens():
    assert timed_tokens([["So,", 0.0, 0.2], ["…", 0.2, 0.3], ["well-known", 0.3, 0.8]]) == [("so", 0.0, 0.2), ("well", 0.3, 0.8), ("known", 0.3, 0.8)]
    assert timed_tokens([["Hello,", 1.0, 1.5], ["world!", 1.5, 2.0]]) == [("hello", 1.0, 1.5), ("world", 1.5, 2.0)]
    assert timed_tokens([["This", 2.0, 2.5], ["is", 2.5, 3.0], ["a", 3.0, 3.5], ["test.", 3.5, 4.0]]) == [("this", 2.0, 2.5), ("is", 2.5, 3.0), ("a", 3.0, 3.5), ("test", 3.5, 4.0)]

def test_timed_tokens_drops_punctuation_only_words_and_split_hyphens():
    words = [["So,", 0.0, 0.2], ["…", 0.2, 0.3], ["well-known", 0.3, 0.8]]
    assert timed_tokens(words) == [("so", 0.0, 0.2), ("well", 0.3, 0.8), ("known", 0.3, 0.8)]