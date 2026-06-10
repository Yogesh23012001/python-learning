# Embeddings

## What an embedding is (my words)
(2 sentences. What did section 1 of the script show you?)
An embedding converts a piece of text into a list of numbers that represents its semantic meaning.The script showed that the sentence "What is karpathy's GitHub score?" became a 384-dimensional vector that can be used for semantic comparison.

## Cosine similarity (my words)
(What does it measure? What range? What did the dog/puppy/database numbers show?)
it measure the same direction of two vector , it range between 1 to -1 , 
dog and puppy have sim 0.8 it mieans they have realtively same semantc meaning. where ase database it is .2 that shows they dont share similar meaning

## The cricket-score trap
(Why did q5 score lower than q2/q3 despite sharing the word "score"?
What does this prove about embeddings vs keyword matching?)
they have simlar words but the overall sematic meaning of the sentace is not same . It contains other dimension which include cricket wherease q2 and q3 share more similar simiantic score using github  , develper etc.

## Semantic cache threshold
(What threshold would I pick and why? What's the failure mode if too high?
Too low?)
I will pick a number 0.8 this will give me enough space to evalute the sematic meaning.  If I choose very high then my same meaning senatce end up in differnt meaning and will not git the cache viceversa we will hit cache for not similar  meaning
## Where this is load-bearing
(Name the things downstream that need embeddings: semantic cache, RAG, search.)
