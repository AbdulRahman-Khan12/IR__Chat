# Knowledge Graphs and Structured QA

A knowledge graph stores facts as triples of subject, predicate and object, for example that BM25 was developed by Stephen Robertson. Entities are nodes, predicates are edges, and a fact is a path between them.

The Resource Description Framework is the standard triple format on the web, and SPARQL is its query language, which became a World Wide Web Consortium recommendation in 2008. Freebase, launched by the company Metaweb in 2007, was an early large open knowledge base; Google acquired it in 2010 and closed it in 2016, migrating data to Wikidata, which the Wikimedia Foundation launched in October 2012. Google announced its own Knowledge Graph in May 2012.

Neo4j takes a different model. It stores a property graph in which nodes and relationships both carry key-value properties, and it is queried with Cypher, a declarative language introduced in 2011 whose ASCII-art patterns resemble the shape of the graph being matched.

Knowledge-based question answering translates a natural language question into such a query. Answers are precise and traceable, but the system can only answer what the graph contains.
