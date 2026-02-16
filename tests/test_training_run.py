import os
from src.data_preprocessing import CharTokenizer, TextDataset, DataLoader
from src.model import Transformer
from src.training import train_model


def test_training_run_smoke(tmp_path, monkeypatch):
    # Tiny synthetic corpus
    s = "hello world " * 8
    tokenizer = CharTokenizer(s)
    data = tokenizer.encode(s)

    seq_length = 16
    dataset = TextDataset(data, seq_length)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    model = Transformer(vocab_size=tokenizer.vocab_size, embed_size=32, num_layers=1, heads=4, max_length=128)

    save_path = str(tmp_path / "models")

    # Prevent thermal guard from sleeping during test
    monkeypatch.setenv('THERMAL_TEMP_THRESHOLD', '1000')
    monkeypatch.setenv('THERMAL_USAGE_THRESHOLD', '1000')

    train_model(model, loader, num_epochs=1, save_path=save_path)

    # Verify checkpoint file was written
    files = os.listdir(save_path)
    assert any(f.endswith('.pkl') for f in files)
