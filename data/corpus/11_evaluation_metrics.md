# Evaluating Question Answering and Retrieval

A question answering pipeline has to be measured in two places, because it can fail in two ways: the retriever may never surface the right passage, or the extractor may pick the wrong span from a passage that did contain the answer.

Retrieval is scored with precision at k, the fraction of the top k results that are relevant, and recall at k, the fraction of all relevant documents that appear in the top k. Mean reciprocal rank averages one divided by the rank of the first correct result across all questions, so an answer found at rank one scores 1.0 and one found at rank four scores 0.25. Normalised discounted cumulative gain extends this to graded relevance.

Short answers are scored with exact match, which demands a character-for-character hit, and token-level F1, which gives partial credit for overlapping words. SQuAD popularised reporting both together.

Generated text is scored differently. BLEU, introduced by Kishore Papineni and colleagues at IBM in 2002, compares n-gram overlap with reference translations, and ROUGE, published by Chin-Yew Lin in 2004, does the same for summarisation with a recall orientation.
