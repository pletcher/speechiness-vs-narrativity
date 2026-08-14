import torch
from captum.attr import LayerIntegratedGradients
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = "pletcher/grc-homeric-speech-narrative-sentence-classification"
device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL)

assert tokenizer is not None

model = AutoModelForSequenceClassification.from_pretrained(MODEL)
model.to(device)
model.eval()

TARGET_CLASS = 0


def forward_fn(input_ids, attention_mask):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    return torch.softmax(outputs.logits, dim=-1)[:, TARGET_CLASS]


lig = LayerIntegratedGradients(forward_fn, model.bert.embeddings)


def build_inputs(text, max_length=128):
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )  # ty: ignore[call-non-callable]
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    baseline_ids = input_ids.clone()
    special_mask = torch.tensor(
        tokenizer.get_special_tokens_mask(  # ty: ignore[unresolved-attribute]
            encoded["input_ids"][0].tolist(), already_has_special_tokens=True
        ),
        device=device,
    ).bool()
    baseline_ids[0, ~special_mask] = tokenizer.pad_token_id  # ty: ignore[unresolved-attribute]

    return input_ids, attention_mask, baseline_ids


def explain(text, n_steps=50):
    input_ids, attention_mask, baseline_ids = build_inputs(text)

    attributions, delta = lig.attribute(
        inputs=input_ids,
        baselines=baseline_ids,
        additional_forward_args=(attention_mask,),
        n_steps=n_steps,
        return_convergence_delta=True,
    )

    # attributions shape: (1, seq_len, hidden_dim) -> collapse to per-token score
    token_scores = attributions.sum(dim=-1).squeeze(0)
    token_scores = token_scores / torch.norm(token_scores)  # normalize for readability

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])  # ty: ignore[unresolved-attribute]

    seq_len = attention_mask[0].sum().item()
    tokens = tokens[:seq_len]
    token_scores = token_scores[:seq_len].detach().cpu().numpy()

    with torch.no_grad():
        pred_prob = forward_fn(input_ids, attention_mask).item()

    print(f"Text: {text}")
    print(f"P(class={TARGET_CLASS}) = {pred_prob:.4f}")
    print(f"Convergence delta (should be near 0): {delta.item():.4f}\n")
    for tok, score in zip(tokens, token_scores):
        print(f"{tok:>15s}  {score:+.4f}")

    return tokens, token_scores, pred_prob


if __name__ == "__main__":
    explain(
        """δέκατον μὲν ἔτος τόδʼ ἐπεὶ Πριάμου μέγας ἀντίδικος Μενέλαος ἄναξ ἠδʼ Ἀγαμέμνων, διθρόνου Διόθεν καὶ δισκήπτρου τιμῆς ὀχυρὸν ζεῦγος Ἀτρειδᾶν, στόλον Ἀργείων χιλιοναύτην τῆσδʼ ἀπὸ χώρας ἦραν, στρατιῶτιν ἀρωγήν, μεγάλ’ ἐκ θυμοῦ κλάζοντες Ἄρη, τρόπον αἰγυπιῶν οἵτʼ ἐκπατίοις ἄλγεσι παίδων ὕπατοι λεχέων στροφοδινοῦνται πτερύγων ἐρετμοῖσιν ἐρεσσόμενοι, δεμνιοτήρη πόνον ὀρταλίχων ὀλέσαντες·"""
    )

    explain(
        "εἶτα πῦρ ἂν οὐ παρῆν ἀλλʼ ἐν πέτροισι πέτρον ἐκτρίβων μόλις ἔφηνʼ ἄφαντον φῶς ὃ καὶ σῴζει μʼ ἀεί"
    )
