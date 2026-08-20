# WordNet and Lexical Semantics

WordNet is a lexical database of English begun in 1985 at Princeton University under the psychologist George Armitage Miller. It groups words into synsets, sets of synonyms that share one sense, and links those synsets by semantic relation rather than by spelling.

The most important relation is hypernymy, the is-a link that connects a dog to a canine and a canine to an animal. Its inverse is hyponymy. Meronymy records part-of relations, such as a wheel belonging to a car, and antonymy records opposites. Version 3.0 contains roughly one hundred and seventeen thousand synsets covering nouns, verbs, adjectives and adverbs.

Because a word form maps to several synsets, WordNet makes polysemy explicit and provides the sense inventory for word sense disambiguation. The Lesk algorithm, published by Michael Lesk in 1986, picks the sense whose dictionary gloss overlaps most with the words surrounding the target.

For retrieval, WordNet supports query expansion. Adding synonyms and hypernyms of a query term raises recall, at the cost of precision when the wrong sense is expanded.
