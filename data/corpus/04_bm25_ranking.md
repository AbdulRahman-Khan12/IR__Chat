# The BM25 Ranking Function

BM25, often called Okapi BM25 after the Okapi retrieval system, was developed by Stephen Robertson, Karen Sparck Jones and colleagues at City University London. It was refined through the Text REtrieval Conference, which the National Institute of Standards and Technology has organised since 1992, and it remains the default lexical baseline three decades later.

BM25 comes from the probabilistic relevance framework, which ranks documents by the estimated probability that they are relevant to the query. It improves on plain term frequency weighting in two ways.

The first is term frequency saturation. Seeing a query word twenty times instead of ten does not make a document twice as relevant, so the contribution of term frequency flattens out. The parameter k1, usually set between 1.2 and 2.0, controls how quickly it flattens.

The second is length normalisation. A long document contains more words and would otherwise win by sheer size. The parameter b, usually 0.75, controls how strongly a document is penalised for being longer than average. Setting b to zero disables normalisation entirely.
