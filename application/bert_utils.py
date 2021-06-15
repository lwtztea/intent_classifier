def tokenize_text(texts, tokenizer):
    texts = [text.lower() for text in texts]
    tokenized = tokenizer.prepare_seq2seq_batch(texts, return_tensors='pt')
    input_ids = tokenized.input_ids
    mask = tokenized.attention_mask
    return input_ids, mask


def predict_intent(input_ids, mask, model):
    model.eval()
    output = model(input_ids, mask)
    output = output.argmax(dim=-1).cpu().tolist()
    return output
