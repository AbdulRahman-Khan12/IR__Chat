# Named Entity Recognition

Named entity recognition finds spans of text that name a real world entity and labels each with a type such as person, organisation, location or date. The task was formalised at the sixth Message Understanding Conference in 1995, and the CoNLL-2003 shared task fixed the four-way scheme of person, location, organisation and miscellaneous that many systems still report against.

Because entities are spans rather than single tokens, the standard encoding is BIO tagging. The first token of an entity is labelled B, tokens inside it are labelled I, and everything else is labelled O. This turns entity recognition into ordinary sequence labelling.

Methods have moved from hand-written rules and gazetteer lookup, through conditional random fields with engineered features, to fine-tuned transformer taggers. The spaCy model en_core_web_sm predicts eighteen types trained on OntoNotes, including PERSON, ORG, GPE, DATE, CARDINAL and WORK_OF_ART.

Named entity recognition is the backbone of factoid answer extraction. A question beginning with "who" expects a PERSON, a question beginning with "when" expects a DATE, so the expected answer type maps directly onto entity labels found in a retrieved passage.
