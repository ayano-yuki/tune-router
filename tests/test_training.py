from pathlib import Path
from types import SimpleNamespace

from tunescope.training import _dpo_model, _dpo_tokenizer_name


def test_dpo_tokenizer_uses_adapter_tokenizer_when_available(tmp_path: Path, monkeypatch) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "peft.PeftConfig.from_pretrained",
        lambda model_name: SimpleNamespace(base_model_name_or_path="base/model"),
    )

    assert _dpo_tokenizer_name(str(adapter)) == str(adapter)


def test_dpo_model_loads_peft_adapter_as_trainable(tmp_path: Path, monkeypatch) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    calls = []

    def fake_base_from_pretrained(model_name, **kwargs):
        calls.append(("base", model_name, kwargs))
        return "base-model"

    def fake_peft_from_pretrained(base, model_name, **kwargs):
        calls.append(("peft", base, model_name, kwargs))
        return "peft-model"

    monkeypatch.setattr(
        "peft.PeftConfig.from_pretrained",
        lambda model_name: SimpleNamespace(base_model_name_or_path="base/model"),
    )
    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", fake_base_from_pretrained)
    monkeypatch.setattr("peft.PeftModel.from_pretrained", fake_peft_from_pretrained)

    model, peft_config = _dpo_model(str(adapter), {}, revision=None)

    assert model == "peft-model"
    assert peft_config is None
    assert calls == [
        ("base", "base/model", {"revision": None, "trust_remote_code": True, "device_map": "auto"}),
        ("peft", "base-model", str(adapter), {"is_trainable": True}),
    ]
