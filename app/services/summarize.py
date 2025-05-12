import sys

import torch
from transformers import BartForConditionalGeneration, BartTokenizer

from app.config import MODEL_PATH as model_path

model = BartForConditionalGeneration.from_pretrained(model_path)
tokenizer = BartTokenizer.from_pretrained(model_path)


def summarize(text: str, max_length: int = 256, min_length: int = 30) -> str:
    """
    Summarizes the input text using a custom pre-trained BART model.
    :param text: The input text to summarize.
    :param max_length: The maximum length of the summary (max tokens)
    :param min_length: The minimum length of the summary (min tokens)
    :return: The generated summary.
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",    # PyTorch tensors
        truncation=True,        # cut off long texts to fit max length
        padding="longest",      # pad to the longest sequence in the batch
        max_length=1024,        # BART's max length
    )
    input_ids = inputs["input_ids"].to(model.device)

    # generate summary
    with torch.no_grad():
        summary_ids = model.generate(
            input_ids,
            max_length=max_length,
            min_length=min_length,
            length_penalty=1.5,  # penalize >1.0: longer summaries, <1.0: shorter summaries
            num_beams=3,  # beam search (consider 4 candidates at each step)
            early_stopping=False,  # continue until max length or EOS
            repetition_penalty=2.0,  # avoid repeating phrases
            no_repeat_ngram_size=3,
            do_sample=True
        )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


if __name__ == "__main__":
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as f:
            input_text = f.read()
        summary = summarize(input_text)
        print("Original Text:")
        print(input_text)
        print("\nSummary:")
        print(summary)
    else:
        input_text = """
    The transformer is a deep learning architecture that was developed by researchers at Google and is based on the multi-head attention mechanism, which was proposed in the 2017 paper "Attention Is All You Need". Text is converted to numerical representations called tokens, and each token is converted into a vector via lookup from a word embedding table. At each layer, each token is then contextualized within the scope of the context window with other (unmasked) tokens via a parallel multi-head attention mechanism, allowing the signal for key tokens to be amplified and less important tokens to be diminished.
        """

        summary = summarize(input_text)
        print("Original Text:")
        print(input_text)
        print("\nSummary:")
        print(summary)
