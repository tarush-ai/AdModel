FinalScore(ad)= w₁⋅Relevance(ad, user) + w₂⋅Quality(ad) + w₃⋅log(1+Stake(ad))

Relevance(ad, user): cosine similarity between ad and user query/profile

Quality(ad): static trust/satisfaction prior (could be CTR, review stars, internal rating).

Stake(ad): the “pot contribution” from the advertiser. Taking the log prevents whales from dominating completely.

w₁, w₂, w₃: tunable weights (initially w₁≈0.6, w₂≈0.2, w₃≈0.2).


User Topic Interest Formula Metrics for Algorithm

1. Frequency Oriented Attention signals
    - Comparative average duration of LLM interaction on topic (time of last llm output - time of first prompt)
    - Comparative average Frequency of prompts on the same topic within a conversation
    - Comparative average number of relevant conversations within a week
    - Time since last prompt: Hawkes burst boost

2. User prompt analysis 
    - Actionable verbage tracking in user prompt ("buy," "book," "compare," "plan") vs ("curious," "what is")
    - Prompt length and connotation on topic
    - Prompted tool calls within result ("search for... ")
    - Satisfaction based on LLM results for user query (this helps, this wasn't interesting)

3. Co-occurence lift: topics that rise together 


