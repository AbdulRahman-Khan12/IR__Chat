# Question Answering Systems

A question answering system returns an answer, not a ranked list of documents. Factoid questions are the classic target: they ask for a short fact such as a name, a date, a place or a quantity.

Two architectures dominate. IR-based question answering retrieves passages likely to contain the answer and then extracts the answer span from the text. Knowledge-based question answering instead converts the question into a structured query against a database or knowledge graph, and returns whatever the query yields.

The knowledge-based approach is older. BASEBALL, built by Bert Green and colleagues in 1961, answered questions about baseball games from a structured record, and LUNAR, built by William Woods around 1971, answered geologists' questions about moon rock samples.

The Text REtrieval Conference ran a dedicated question answering track from 1999, which established the retrieve-then-extract pipeline as standard. IBM Watson beat the champions Ken Jennings and Brad Rutter on the quiz show Jeopardy in February 2011 using a massive ensemble of both approaches. In 2016 Stanford released SQuAD, a dataset of more than a hundred thousand questions written against Wikipedia paragraphs.
