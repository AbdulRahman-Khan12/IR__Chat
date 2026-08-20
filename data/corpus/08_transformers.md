# The Transformer Architecture

The transformer was introduced in the 2017 paper "Attention Is All You Need" by Ashish Vaswani and colleagues at Google. Its contribution was subtractive: it removed recurrence and convolution from sequence modelling and kept only attention.

Self-attention lets every position in a sequence look directly at every other position and weight them by relevance. Because there is no recurrence, all positions are computed in parallel, which is what made training on very large corpora affordable. Multi-head attention runs several attention functions side by side so different heads can specialise, and positional encodings restore the word order that removing recurrence discarded.

The original model was an encoder-decoder built for machine translation. Later work split it. BERT, released by Google in 2018, keeps only the encoder and is trained by masking words and predicting them, which makes it strong at understanding tasks such as extractive question answering. The GPT series from OpenAI, which began in 2018, keeps only the decoder and predicts the next token, which makes it strong at generation.
