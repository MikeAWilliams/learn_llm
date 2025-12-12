# Token Expiriment

I am building on what I leanred from Karpathy/build_gpt_from_scratch and Karpathy/tokenization to try and learn more about how tokenization affects the model results. I am taking the v2 model from the gpt and refactoring it into a library. Then I will drive exiriments with the tokens from the main.py.

Expiriments I am interested in running
1. First I will record the result of the base tokenization, raw file. This uses the ascii characters and punctuation, with capitilization directly.
1. Next I will try to minimize the tokens by going to all lower case and removing all punctioation besides space and period. I expect this to reduce the expressivness of the results, but I hope it will make better sentences.
1. I will then try and run a basic tokenization using byte pair encoding, but keeping all the caps and punctuation and compare the results.
1. Maybe I will tokenize the reduced character set after that
