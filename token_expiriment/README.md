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

### tokenized but not scrubeed using the basic tokenizer and 100 extra tokens

Total vocab size is 165
Time taken on m4 mac pro 42.3 minutes

#### result text

ES OF YORK:

Heyded master, as I like it, they did love it.
Was I been follow'd Henyry's love,
Lest on your first hand men.
But, as it is seasy to dust and beside,
Yet being such in distincts them part
That ne'er bear with their blads in vain remoond
From the hosts that crown mame not on this world
To make a rate his enemies!

DUCHES OF YORK:
Think'st thou that presume, thou not banish'd a fearful hold,
Take him, my heat is foul so tream of truth;
It is fond, none is is his princely fair drinn.
For God, what is he doth he dreams with our interceptres;
The wretch is it most innocent;
His figuring in his captiversialor and his eye.
This like the war of York, this the sworth living,
Made impeder'd and our ministers,
And take him stolding his meet in England;
And with his new kingdom another d and b,
As he is royal king inconsentedancement from him?

DUKE OF AUMERLE:
No; I am the sure of judgment of this safety
Let the slandy of reverence; be if thou love Edward,
In the right royal taunt of shame to let moke.

DUCHESS OF YORK:
Art thou my son Edward, for thy kingly Daughter?
And, tawe, farewell.

QUEEN MARGARET:
Herefore sworn short may pay for Edward, sweet York,
Your rold-past that worthou have kill'd, and Montagued!
Why dost thou not King Henry bundfall'd me hither clips
The glory live, thought, that have uttold mow'd it had,
And these watered-for captize counced fire,
Have I not pass'd from the Duke of Northumberland.
And hire of Lancaster?
Tell my towards London were my inward bushops in arms,
And Clarence; for truth, I do bueseem
Thus, that now revolt upon their heated away.

HENRY BOLIN

### tokenized but not scrubeed using the basic tokenizer and 2000 extra tokens

Total vocab size is 2065
Time taken on m4 mac pro 65.3 minutes

#### result text

wilt dog living,  and cast born,
Are of thy se cure, as from thy law of king,
Were this shine much,
Which son'd to bleedyield to deliver'd to sances hand,
This deadly ly,
Then ever practed touch'd his fin dust from our other's love
Unto our pea of her loyal heart: in paints and his time
With hearts return conspiring daughter,
Unto repent brought a wited the come
UnI will to each whom my country's miseries,
And state much conseal'd off with Ble.

LEONTES:
Whose daughter. Come, Camillo, my sweet Cliffordwell-wixt thy another royal fool, thou ne'er weast:
I shall parthee to the drop and him,
Mel of my ging warrant this my brother king?
Longst thou leaves and are thou not thy children's gth.
The whreslanderous look you before I sple this sight
a crown unat?
Or that reglady's love and death was a min womb of great rees,
For visage makes for heretiment:
Towards Less on my groom,
As iff and part your sight,
Your gentently medied, dead in!
Ratis her in this w herbripast;
For living here, I belike the accuate,
A many likenown that Polix'd inchoice gazing woose my craves it babide my de,
And this deliver of my fending him.
I'll have have her soul.
When he is mistake our'd my lord, thereof, letter
Preputation, and crave our dears.

A Plash my part therein,
And young Please you, fair belike conceive
From Gloucester's same heavens to hear acoler:
His death upon this fair creature is, Say it conduct pardon
To pleasure  mothers.

ANGELO:
Yet hath apparent hour to be mad;
And from this make condemand pardoncounsel
In her counsel and am.
'd, and more advantage: the father's grandy? let me be as fitself--
As is dear somin your love have I love him.

ROMEO:
That's a cup, I crafond myself!
She shallow me for thy ligs.

Nurse:
And so will men not stire.

PARIS:
That ever came from us at all my n.
'Tis dead.
She is done so shall not poor in sleep! afear's eyes?

LADY CAPULET:
Ay, those full of groans fall bounds.
Hark, 'tis much 'tis far butchery: why, ally looks, still you all
The thirthful comfort so strong.
The sun not before I pany'd: is vouch of you say you are--
But with a head!

DUCHESS OF YORK:
Welcome, good friend Hearly at Pain, thou washful sont Pashions, take by my conservice,
To see what I am the safession empesteem withinfection.
Why are the world cannot at my season.

CAMILLO:
Etheside
Sent have so bad of loved
That thee to my nare: but length the world canken, till he take in a goes,
Ladvantage  say at Tuadies, truth bood:
Comfort, as you shall do send you out your man.
Most welcome you be pledged to do mend him together,
And I am the most trumpet by hinturged again to her at the LUrge it.

GLOUCESTER:
They are ady Ann your lord,
Wher bound of York thus reeffements: whose ces,
For I have set my ved:
Since I offices, how I lives are the viewian is the dog
And ceags for mod a cheer your love;
But where is no time to your high pleasure of love,
Your son, gent?

EDWARD:
My lord how but a mother, most Christian's life; his good haviors at my lord, did myself,
It is some merry can betray God gif in a man;
Then am I king my soul to ser than myself,--
Shalt besharm, and incursed by myself unfell in with Matchful hour hearts;
And he be no doubting onighty
