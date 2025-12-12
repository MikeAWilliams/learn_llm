# Token Expiriment

I am building on what I leanred from Karpathy/build_gpt_from_scratch and Karpathy/tokenization to try and learn more about how tokenization affects the model results. I am taking the v2 model from the gpt and refactoring it into a library. Then I will drive exiriments with the tokens from the main.py.

Expiriments I am interested in running
1. First I will record the result of the base tokenization, raw file. This uses the ascii characters and punctuation, with capitilization directly.
1. Next I will try to minimize the tokens by going to all lower case and removing all punctioation besides space and period. I expect this to reduce the expressivness of the results, but I hope it will make better sentences.
1. I will then try and run a basic tokenization using byte pair encoding, but keeping all the caps and punctuation and compare the results.
1. Maybe I will tokenize the reduced character set after that

## tokenization

This implementation provides a **Character Pair Encoding** tokenizer inspired by the Byte Pair Encoding Tokenizer. This one works directly with characters instead of UTF-8 bytes.

### Compression Performance

Based on experiments with the full input.txt dataset (base vocabulary: 65 characters):

| New Tokens | Total Vocab Size | Compression Ratio | Last Merge Occurrences |
|------------|------------------|-------------------|------------------------|
| 50         | 115              | 1.44x             | 2373                   |
| 100        | 165              | 1.62x             | 1267                   |
| 500        | 565              | 2.27x             | 223                    |
| 1000       | 1065             | 2.68x             | 100                    |
| 2000       | 2065             | 3.19x             | 45                     |
| 4000       | 4065             | 3.81x             | 18                     |

![Graph of the abovee](./shakespeare_tokenization.png)

## Scenarios

here are some results

### base
Total vocab size is 65
Time taken on m4 mac pro 43.9 minutes

#### result text

a present of alliberd.

SICINIUS:
Why, the city I beggars will cribe the strength,
And bury threefollowers before the rest.

Senator:
Ah, how above you this senate marchy in prison!
The more shall make my feeb met?
The lady presservation'd and on thement,
And so your very trick my house at home,
The widows itself that I'ld show.
Would that's that? it was even like it faint, I sir,
The entreats of his deeds, contend to const
In the sorps that dieth self most under-spreech'd
Up; and ere he did no baniss'd the ground;
But once his friends to redeep him there do none.

CLIFFORDIFAD:
Off! What maidst thou me! look on on that way
Might bind revenge, for thy cold father, as the
meeting is more than where things love me, thou so art
As too for my head and thy wits, and do so,
And bid him meet here signorant.

First Conspirator:
And what means give Katharina my page?
Enjoy away what this was the noble heart?
I talk no grant that would our tent to know
The three, nother's King was alone,
Put upo

### scrubbed - reduced character set
Total vocab size is 29
allowed_chars = set("abcdefghijklmnopqrstuvwxyz .\n")
Time taken on m4 mac pro 44.2 minutes

#### result text

o standing is the truth to civil him.

king henry vi
then most vicious sreat torchery today
he is great enemies lost the sun of york.

york
but farewell my lord ill fear unto the ground
the trumpets set sorrow his revengement
for shame us i cannot be truth on.

northumberland
ross
i will tell the appear as i hear thee
she runs again
that makes me not you say pity fight
she hath made them shame for scoldiers and
the one day ever had alike althought
is one then by that ratcliff dost thou
even the cause of the death is death
when thinking balm his face against whave
by his poison clifford did spent of mind.
thus thou ish of heavy can life
as make want there corpse against thy fair.
is nothing but the maidships of soyverigone
that yetling to the deet will not stay
a called trusty. comes therefore is foread
and i know it is english.
it now here urse whos name gentlement
thy mislet prefian to come to my just true
to please that word it with do i think
should a prison out the sighty deep
to e
