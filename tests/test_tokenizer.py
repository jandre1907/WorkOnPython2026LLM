from src.data_preprocessing import CharTokenizer


def test_char_tokenizer_roundtrip():
    s = "The quick brown fox jumps"
    ct = CharTokenizer(s)
    encoded = ct.encode(s)
    decoded = ct.decode(encoded)
    assert decoded == s
