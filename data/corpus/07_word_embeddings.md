# Word Embeddings

Lexical retrieval fails when the question and the document use different words for the same thing. Embeddings attack this by representing each word as a dense vector whose position encodes meaning.

The idea rests on the distributional hypothesis, associated with Zellig Harris in 1954 and John Rupert Firth in 1957, which holds that words appearing in similar contexts tend to have similar meanings.

Word2vec, released by Tomas Mikolov and colleagues at Google in 2013, made the approach practical. It offers two training objectives: continuous bag of words, which predicts a word from its context, and skip-gram, which predicts the context from the word. GloVe followed in 2014 from Jeffrey Pennington, Richard Socher and Christopher Manning at Stanford University, fitting vectors to global co-occurrence counts rather than local windows. FastText, published by Facebook AI Research in 2016, represents a word as a bag of character n-grams, so it can build a vector for a word it has never seen.

Similarity between two embeddings is measured with cosine similarity. Dense retrieval extends the idea from words to whole passages.
