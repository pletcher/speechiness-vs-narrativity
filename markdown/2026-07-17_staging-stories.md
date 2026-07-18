Staging Stories: Echoes of epic narrativity in Athenian Tragedy
------

## Introduction


## Literature review

### Choruses and messengers

As Florence Yoon notes, "Messenger" is a "problematic" label for characters:
"The word more accurately describes a fluid extra-dramatic function that can be
performed by any character" (@Yoon2022 364). While I generally agree with Yoon's
definition and analyses, I have trouble with the notion that the messenger
function is "extra-dramatic," "a concern for the producer rather than a part of
the world of the play" (@Yoon2022 374). The issue is that when a character—and I
agree that any character can perform the messenger function—acts as a messenger,
they signal an epistemic stance not just for the audience or the producer but
for the other character's in the play: they have witnessed an event that others
have not seen, and they stand onstage to describe and interpret that event for
their internal and external audiences at the same time.

Like Yoon, I use the phrase "messenger function" instead of "messenger speech"
(or "messenger's speech") because any character can perform this function.
In Sophocles' _Trachiniae_, for example, nearly every character delivers
what could be considered a "messenger's speech" at some point, but there is
only one character traditionally designated Ἄγγελος, so it is more precise
to say that every character _performs the messenger function_.^[Yoon rightly
points out that Ἄγγελος is likely a late designation and prefers "Old Man"
for this character (@Yoon2022 376).] In addition to Yoon's article (and its
bibliography) and her work on anonymous characters (@Yoon2012), work by Felix
Budelmann and Evert van Emde Boas on a cognitive approach to the messenger
function; work by Eleanor Dickin on how messenger roles developed from bit
parts to star platforms; and seminal monographs by James Barrett and Irene J.
F. de Jong speak to the scholarly interest in tragic messengers over the past
few decades (@Budelmann.vanEmdeBoas2020; Dickin2009; Barrett2002; deJong1991).
Simon Perris and C. W. Marshall have also made valuable contributions to
efforts to define this surprisingly elusive tragic figure/function (@Perris2011,
@Marshall2006). Perris prefers "report-narrative" to describe the type speeches
under investigation, and he usefully sums up the long-supposed epic associations
with these speeches: "... tragic report-narratives have long been considered an
'epic' feature, principally due to the supposedly languid narrative style, the
apparently objective stance, the frequency of direct speech, and the occurence
of 'Homeric' linguistic features (epithets and unaugmented historic verbs) in
some speeches" (@Perris2011 8). Where Perris is "loathe to universalise the epic
quality" of the messenger function, Barrett, on the other hand, and Yoon more
recently seem to take the messenger function's epic tilt as more or less a given
(@Perris2011 8; @Barrett1995; @Barrett2002 PAGE; @Yoon2022 382 n. 62).

If the messenger function's authority derives from its close association
with the epic narrator, we should be able to quantify that influence using
bidirectional encoder representations from transformers (BERT) trained to
classify epic direct speech as opposed to narrative (@Vaswani.etal2023). BERT
models learn to represent textual content as many-dimensional lists of numbers.
By "fine-tuning" a BERT to classify these vectors according to pre-assigned
labels, we can use these abstract representations of textual content to compare
both text and its context. Classifying by epic speech versus epic narrative,
as opposed to classifying according to an epic-tragic axis with the hypothesis
that messengers will be more "epic" overall, avoids the meter conundrum by
restricting the model to one meter (dactylic hexameter) and applying it to
tragic meters (predominantly iambic trimeter, but the experiments apply the
model to anapests and lyric meters as well).

One might wonder whether the messenger function's epic flavor comes from a
preference for metrical resolution compared to other speaker classes. A full
report is outside the scope of this paper, but I have conducted a computational
metrical analysis of the iambic trimeters in tragedy and found that no character
class shows a statistically significant preference for or against resolution.

![FIGURE SHOWING RESOLUTIONS]

Thus, we use a fine-tuned BERT to measure similarities between the learned
vectors representing epic speech and narrative and the vectors representing
sentences in tragedy.

### Calculating influence

The computational methods for this work draw on a rich history of quantitative
research on Homeric epic.^[The bibliography on quantitative approaches to
Homer is voluminous, and I cannot cover all of it here. Tom McConnell's
recent monograph contains the most up to date methodologies alongside
sophisticated linguistic insights and a wide-ranging bibliography in its own
right (@McConnell2025; see esp. ch. 7). Ombretta Cesca and Matteo Romanello
have recently demonstrated the use of the computational tool Passim to detect
repetitions in Homeric characters' speech (@Cesca.Romanello2026). On the
speech-narrative divide in Homer, Jasper Griffin's well-known article remains
essential, if controversial, reading (@Griffin1986); Margalit Finkelberg
provides a more measured but still quantitative approach to Homeric formulae
(@Finkelberg1989). Deborah Beck has further elucidated performance and
performative aspects of Homeric speech, and her digital publications provide
additional tools for the reader to investigate claims made here (@Beck2020,
@Beck2012a, @Beck2012, @Beck2026). Although a network-based approach is not used
in this paper, Jeff Rydberg-Cox's recent work with networks and words in Homer
deserves to be flagged for potential follow-up research (@Rydberg-Cox2026).]
In recent years, teams of scholars have taken various approaches to character
type in drama, such as network analysis to capture the influence of _servi
callidi_ figures in Roman comedy (@Beine.etal2025); sentence embeddings using
SBERT to calculate semantic similarity and capture repetition among speakers
(@Szemes.Nagy2024); and training a classification model to track character
archetypes (Keith.etal2026) or sentiment analysis in the New Testament
(@Kang2025). Little work, however, has been done on quantifying the narrativity
of the messenger function in Athenian tragedy. As described above, this paper
uses a binary classifier trained on speech and narrative in the Homeric epics
as one way of quantifying what sets messenger narratives in tragedy apart from
other types of tragic speeches.

## Methodology

### Computational methods

In a recent paper on character archetypes in Calderón de la Barca, Allison
Keith et al. use the performance of a pretrained automatic classification
model to determine "how differentiable the speech of one archetype is from that
of another" (@Keith.etal2026 2). In a similar vein, this paper does not use
its model to classify speech and narrative _per se_, but rather to measure a
passage's affinity for Homeric narrative style.

To train the model, speeches in the Homeric epics were obtained from the Digital
Initiative for Classics: Epic Speeches (DICES) database (@Forstall.etal2022).
The texts of the epics were obtained from the Perseus Digital Library
treebanks^[https://perseusdl.github.io/treebank_data/], and a script assigned
each sentence—not line—a label of "speech" or "narrative" based on the speech
ranges obtained from DICES. The result was 8633 lines labeled "narrative" and
7125 lines labeled "speech" for the _Iliad_ (15,758 lines total); 3878 lines
labeled "narrative" and 8404 lines labeled "speech" for the _Odyssey_ (12,282
lines total).

![TABLE OF LINES AND LABELS]

The labeled sentences were then used as inputs to fine-tune a language model
to classify a span of text as "speech" or "narrative." The base model is
pnadel/greek-bert, which in turn is a fine-tuned Ancient Greek version of
answerdotai/ModernBERT-base.^[@Nadel2026; @Warner.etal2024] Fine-tuning is
a process whereby a large language model trained on general tasks is further
trained to perform specific tasks with higher accuracy. Fine-tuning thus
significantly reduces energy demands by not requiring the whole model to be
trained again and by enabling training even on relatively small datasets: the
15,148 sentences in this training set are a rounding error compared to the
trillions of tokens on which most large language models have been trained.

In the interest of replicability, the model was trained for 5 epochs on
consumer-grade hardware (a 2021 M1 MacBook Pro), achieving an accuracy of
92.7% and an F1 score (a combined measure of precision and recall) of 0.9268.
As a preliminary experiment, the model was then run against two held-out
datasets: Nestor's long speech at _Iliad_ 11.656–803, and Odysseus' _apologoi_
(9.2–12.453). Both of these speeches contain extended passages of narration,
presenting particular challenges for the model's accuracy.

On these held-out passages, the model performed notably worse than on its
training data. It classified 25 sentences (68.75%) of Nestor's speech as
narrative, and 360 sentences (69.6%) of Odysseus' _apologoi_. These results
suggest that the divide between speech and narrative is less a binary and more
a spectrum, and that we can understand a speech's "narrativity" as the extent
to which its sentences are classified as narrative—in other words, the extent to
which a speech confuses the classification model.

![CONFUSION MATRIX FOR NESTOR'S SPEECH]

![CONFUSION MATRIX FOR APOLOGOI]

These results also suggest that the model can be usefully applied to tragedy:
it will not simply label every sentence as "speech" but will instead provide an
approximate measure of each sentence's narrativity, which can then be aggregated
according to speaker class and plotted for the play as a whole.

### Literary and performance studies methods

Before presenting the quantitative evidence for challenging current definitions
of messenger speech, it will be useful to lay out a methodology drawn from
literary and performance studies as a complement to the computational methods
discussed above.

To understand the messenger's epistemic and hermeneutic roles, I draw on the
Jacques Rancière's ideas on "the distribution of the sensible," which he defines
as "a delimitation of spaces and times, of the visible and the invisible,
of speech and noise, that simultaneously determines the place and stakes of
politics as a form of experience" (@Ranciere2004 13). He continues, "Politics
revolves around what is seen and what can be said about it, around who has the
ability to see and the talent to speak, around the properties of spaces and the
possibilities of time" (@Ranciere2004 13). Rancière's definition in terms of
politics grounds my investigation of the messenger function in the democratic
art of tragedy. It also orients the stakes of our discussion about messenger
speeches and who can deliver them, as the distribution of the sensible entails
the organization of a community, in which the messenger function is one way
of participating.

In "The Aesthetic Dimension," Ranciere clarifies the power imbalance at the
heart of the distribution of the sensible and its connections to the formation
of knowledge:

> A belief is not an illusion in opposition to knowledge. It is the articulation
between two knowledges, the form of balance between those forms of knowledge
and the forms of ignorance they are coupled with. As Plato claimed, articulation
or balance has to be believed. The economy of knowledge has to be predicated on
a story. (@Ranciere2009 16)

The messenger function participates in the economy of knowledge by delivering
news and exercising th power granted by unshared eyewitness experience.
Other speech registers in tragedy—agonistic, emotive, etc.—share knowledge
incidentally; the messenger function shares knowledge as its primary, though not
only, purpose.

## Results

These experiments confirm that the characters we typically call messengers
_do_ have more in common with epic narration than other tragic characters.
At the same time, however, these experiments cast some doubt on that classification by showing that an affinity for epic narrative occurs
across speaker classes and even in many choruses.

## Argument/interpretation

"[A discipline] is a way of defining an idea of the thinkable, an idea of what the objects of knowledge themselves can think and know" (@Ranciere2009 17).

- This applies to the interpretive functions on the stage as well

## Conclusion
