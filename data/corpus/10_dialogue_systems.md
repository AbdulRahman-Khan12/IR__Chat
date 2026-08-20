# Dialogue Systems and Frames

Dialogue systems split into two families. Task-oriented systems help a user complete something specific, such as booking a flight. Open-domain chatbots aim to keep a conversation going without a task to finish.

The dominant design for task-oriented dialogue is the frame. A frame represents one task and contains slots for the pieces of information the task needs; a flight booking frame has slots for origin, destination and date. The dialogue manager checks which slots are still empty and asks a question to fill the next one. The approach comes from GUS, a travel assistant built by Daniel Bobrow and colleagues at Xerox PARC in 1977, and it is still the backbone of commercial assistants.

Modern frameworks describe user turns with an intent and a set of entities. Rasa, open sourced in 2016, is a widely used implementation. Tracking which slots are filled across a conversation is called dialogue state tracking, and the first Dialog State Tracking Challenge was held in 2013.

Keeping conversation history is what allows follow-up questions, where a pronoun such as "it" refers to an entity mentioned in an earlier turn.
