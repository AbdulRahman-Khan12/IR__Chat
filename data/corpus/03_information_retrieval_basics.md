# Foundations of Information Retrieval

Information retrieval is the task of finding, inside a large collection, the documents that satisfy a user's information need. The central data structure is the inverted index, which maps every term to the list of documents containing it, so a query touches only the postings for its own terms instead of scanning every file.

Terms are weighted rather than counted. Term frequency rewards a word that appears often in a document, while inverse document frequency penalises a word that appears in many documents. Karen Sparck Jones introduced inverse document frequency in a 1972 paper in the Journal of Documentation, arguing that the specificity of a term should govern its weight.

Gerard Salton and his group at Cornell University developed the SMART system and the vector space model, in which documents and queries become vectors and relevance becomes the cosine of the angle between them.

Evaluation methodology came from the Cranfield experiments led by Cyril Cleverdon at the College of Aeronautics in Cranfield during the late 1950s and 1960s. Cranfield established the pattern still used today: a fixed document collection, a fixed set of queries, and human relevance judgements, scored with precision and recall.
