Narrative Suspects: Echoes of epic narrativity in Athenian Tragedy
------

## Abstract

Despite a growing body of innovative approaches to the "messenger function"
in Greek tragedy, scholars continue to wrestle with a clear definition of
what makes a "messenger speech." This paper presents and uses a Bidirectional
Encoder Representations from Transformers (BERT) model to measure speakers'
"narrativity" in tragedy. The model has been trained to classify speech and
narrative in the _Iliad_ and the _Odyssey_, and its classification when applied
to tragedy provides a useful proxy for how narrative a given passage is—that
is, how much it resembles Homeric narrative. By applying this model to the
extant tragic corpus, I show that the messenger function in tragedy is less
a question of "who" and more a question of "when." That is, the results from
these experiments, alongside three close-reading case studies, suggest that
we can productively eschew classification of narrative characters in favor of
identifying narrative _moments_.

## Introduction

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

Like Yoon, I use the phrase "messenger function" instead of "messenger
speech" (or "messenger's speech") because any character can perform this
function.^[Although C. W. Marshall does not use the phrase "messenger
function," he does acknowledge, "It is worth noting that when we speak of the
messenger, we are denoting a function and not a role" (@Marshall2006 207n21).]
In Sophocles' _Trachiniae_, for example, nearly every character delivers what
could be considered a "messenger's speech" at some point, but there is only
one character traditionally designated Ἄγγελος, so it is more precise to say
that every character _performs the messenger function_.^[Yoon rightly points
out that Ἄγγελος is likely a late designation and prefers "Old Man" for this
character (@Yoon2022 376).] In addition to Yoon's article (and its bibliography)
and her work on anonymous characters (@Yoon2012), work by Felix Budelmann and
Evert van Emde Boas on a cognitive approach to the messenger function; work
by Eleanor Dickin on how messenger roles developed from bit parts to star
platforms; and seminal monographs by James Barrett and Irene J. F. de Jong
speak to the scholarly interest in tragic messengers over the past few decades
(@Budelmann.vanEmdeBoas2020; Dickin2009; Barrett2002; deJong1991). Simon Perris
and C. W. Marshall have also made valuable contributions to efforts to define
this surprisingly elusive tragic figure/function (@Perris2011, @Perris2011a,
@Marshall2006). Perris prefers "report-narrative" to describe the type speeches
under investigation, and he usefully sums up the long-supposed epic associations
with these speeches: "... tragic report-narratives have long been considered an
'epic' feature, principally due to the supposedly languid narrative style, the
apparently objective stance, the frequency of direct speech, and the occurence
of 'Homeric' linguistic features (epithets and unaugmented historic verbs) in
some speeches" (@Perris2011 8). Where Perris is "loathe to universalise the epic
quality" of the messenger function, Barrett, on the other hand, and Yoon more
recently seem to take the messenger function's epic heritage as more or less a
given (@Perris2011 8; @Barrett1995; @Barrett2002 72 and _passim_; @Yoon2022 382
n. 62).

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
model to trochaic tetrameters, anapests, and lyric meters as well).

One might wonder whether the messenger function's epic flavor comes from a
preference for metrical resolution compared to other speaker classes. A full
report is outside the scope of this paper, but I have conducted a computational
metrical analysis of the iambic trimeters in tragedy and found that no character
class shows a sufficient preference for or against resolution to distinguish it
from other character classes.

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

The tragic corpus uses a hand-curated list of line ranges that cover speeches
typically classified as performing the messenger function. We use these as a
baseline to compare against the rest of tragedy. The full list can be found >>
LINK REMOVED FOR ANONYMITY <<. The only controversial inclusions are several
main-character speeches in _Trachiniae_ —Jebb provides supporting evidence for
classifying them as messenger-type—and the Merchant in Sophocles' _Philoctetes_.

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
news and exercising the power granted by unshared eyewitness experience.
Other speech registers in tragedy—agonistic, emotive, etc.—share knowledge
incidentally; the messenger function shares knowledge as its primary, though not
only, purpose.

## Results

These experiments confirm that the characters we typically call messengers _do_
have more in common with epic narration than other tragic characters. At the
same time, however, these experiments cast some doubt on that classification by
showing that an affinity for epic narrative occurs across speaker classes and
even in many choruses.

![Violin plot showing messengers (right) versus other speakers (left) in
tragedy.](../figures/ALL_DRAMA_distribution.png)

The model reports the probability that a sentence is epic narrative; a
probability of 1 means that the model thinks the line is definitely epic
narrative, and a probability of 0 means that the model thinks the line is
definitely _not_ epic narrative. We can use these probabilities as a proxy for
how narrative-like a sentence is—that is, to use this paper's shorthand, as a
proxy for its narrativity.

As can be seen in the above violin plot, sentences that appear in speeches
typically labelled "messenger" (using the hand-curated list described above)
show significantly higher narrativity than sentences in other characters'
speech, with a mean of 0.353, which is nearly six times the mean narrativity
of other speakers at 0.061. The plot also shows that many messenger-labeled
sentences have very high or very low narrativity scores, but we should keep in
mind that the relatively thin middle is expected: the individual sentences that
make up this plot are not the speeches themselves. We are looking for trends in
groups of sentences, for which the averages are a much better measure.

For a similar reason, the trendlines on the plots of messenger narrative for
individual tragedies below show a rolling seven-sentence average narrativity,
rather than the per-sentence narrativity scores, as the rolling average makes
the data more interpretable. (The seven-sentence average is somewhat arbitrary,
but I felt that seven was a small enough window to show variation while still
large enough to capture trends for longer speeches.)

In every play in which a character performs the messenger function—so excluding
_Prometheus Bound_, _Choephoroi_, and _Eumenides_ —the average narrativity of
messenger labeled sentences exceeds that of other sentences, with the most
striking differences appearing in Euripides' _Bacchae_, Euripides' _Suppliants_,
and Sophocles' _Electra_. Many plays, including _Bacchae_, have passages showing
high narrativity outside of the traditional messenger function, providing
evidence that we should consider expanding the definition of "messenger
function" to include these other passages.

### Euripides _Bacchae_

![Chart showing per-sentence and rolling 7-sentence average narrativity for
Euripides' _Bacchae_](../figures/ALL_DRAMA_line_position_tlg0006.tlg017.png)

In addition to the Messengers' undeniably narrative speeches in _Bacchae_, the
parodos also contains marked narrative moments. The ode starts to grow narrative
around the description of Dionysus' birth in the first antistrophe at 88–104:

| ὅν
| ποτ' ἔχουσ' ἐν ὠδίνων
| λοχίαις ἀνάγκαισι
| πταμένας Διὸς βροντᾶς νη-
| δύος ἔκβολον μάτηρ
| ἔτεκεν, λιποῦσ' αἰῶ-
| να κερανίῳ πληγᾷ·
| λοχίοις δ' αὐτίκα νιν δέ-
| ξατο θαλάμαις Κρονίδας Ζεύς,
| κατὰ μηρῷ δ`καλύψας
| χρυσέαισιν συνερέιδει
| περόναις κρυπτὸν ἀφ' Ἥρας.
| ἔτεκεν δ', ἁνίκα Μοῖραι
| τέλεσαν, ταυρόκερων θεὸν
| στεφάνωσέν τε δρακόντων
| στεφάνοις, ἔνθεν ἄγραν θη-
| ροτρόφον μαινάδες ἀμφι-
| βάλλονται πλοκάμοις.

| Bromius, whose
| mother bore him cast-out
| in the compulsions of labor pains
| from her belly,
| when the thunder
| of Zeus struck: she died
| with the lightning strike.
| But just then Zeus son of Kronos
| received him in a chamber for childbirth,
| and he covered him in his thigh
| and sewed him up with golden clasps,
| hidden from Hera.
| And when the Fates
| brought it to fulfillment,
| Zeus gave birth to the bull-horned god
| and he crowned him with crowns
| of snakes, which is why the maenads
| throw the wild beast-eater
| about their hair.^[Translations are my own.]

Despite the metrical differences between this lyric verse and epic
hexameter, the model correctly identifies the narrative passage of Dionysus'
birth. This passage's narrative flavor comes partly from the concentration of
third-person main verbs: ἔτεκεν twice, δέξατο, στεφάνωσέν, and ἀμφιβάλλονται.
But if third-person verbs sufficed to identify narrative, then every
third-person passage would be marked by the model—clearly the vector-space
embeddings have identified other markers of epic narrative as well. For
instance, we can also see that the passage's vocabulary and tenses seem to
indicate the remoteness in time and place, rather than events taking place in
the moment.

This high-narrative moment has striking echoes in the first traditional
messenger-type speech, when the Messenger (@Buxton1991 calls this Messenger the
"Cowherd") describes Agave and the maenads' ritual preparations. The maenads
rearrange their fawnskins after the fastenings come loose (696–697), "and they
fasten the spotted hides with snakes that lick their cheeks" (697–698: καὶ
καταστίκτους δορὰς | ὄφεσι κατεζώσαντο λιχμῶσιν γένυν). The chorus narratively
and thematically paved the way for this description already in the parodos.
The Messenger's function here cannot simply depend on his telling his audiences
what he saw, but rather requires retelling what the external audience, at least,
already knows from the chorus. The chorus' etiological account becomes the
Messenger's revelation.

Nancy Worman links the play's focus on attire and appearance to "a growing sense
of coercion and menace" that "attaches to the various costumes" (@Worman2021
100). The narrative punctuations allow this menace to grow not just through what
the characters say to each other but also through what they describe that the
audience cannot see. The first Messenger's "pointed counterfactual assertions
about what Pentheus _would_ have seen" are also counterfactuals for the
audience (@Perris2011a 40). Apprehension builds towards the climax of the second
Messenger's speech through these moments of narration that allow the strange
sights to grow stranger in each audience member's mind precisely because the
audience cannot view these sights collectively—cannot view them at all.

But where the audience is not initiated into the rituals that each time end
in _sparagmos_—first of cattle, second of Pentheus—they witness Pentheus' own
transformation at the hands of Dionysus. This transformation dramatizes the
chorus' and the Cowherd's descriptions of the maenads, turning unseen narrative
into enacted ritual preparation. The dialogue, like the earlier chorus and
monologue, pays particular attention to his hair and fawnskin mantle: "I will
spread your hair out at length upon your head," Dionysus says; and when asked
"What else?" by Pentheus, the god replies, "A thyrsus in your hand and a dappled
fawnskin" (831, 835; cf. 24–25). In a similar way, the chorus later prepares the
audience for the second Messenger's grisly report with their prophetic account
of "justice made manifest" (992=1011: ἴτω δίκα φανερός). As Foley puts it,
"In the fourth stasimon (977–1023) the chorus prophetically imagines Pentheus'
destruction on the mountain soon to be reported by the messenger; the messenger
speech is immediately followed by a choral celebration of the god's victory
over Pentheus (1153–64)" (@Foley1985 221). Although the fourth stasimon does
not register high on the narrativity skill, the previous narrative doubling by
proleptic chorus and the analeptic Messengers joins the chorus and Messengers
as two modes of sustaining the miraculous and profane sights that remain beyond
the external audience's point of view.^[So @Foley1985 221n23: "Theatrical (or
psychological) illusion is the only avenue by which the god can be worshipped
and understood." The Messengers and the chorus seem to represent or even enact,
respectively, theatrical and pyschological illusion.]

Perris also notes the echoes of the fourth stasimon in the second Messenger's
speech, but he does not connect these echoes to the further doubling that
has occurred throughout the plays narrative and dramatic action (@Perris2011a
43). Richard Buxton, on the other hand, connects the Messengers' narratives
with the prologue, Pentheus' opening speech, and the speech by the Servant
who delivers Dionysus (as the Lydian Stranger) to Pentheus—together with the
two Messengers' reports, Buxton writes, "these five narratives may be seen
as constituting a series, the later modifying the earlier; in no two cases
is the relationship between content and narrator identical" (@Buxton1991 40).
While the web of interrelated passages that emerges through the performance
of this play is undeniable, Buxton's throughline does not depend on narrative
per se but rather on a sense of place established by the five passages that he
isolates. By instead isolating moments of high narrativity, I mean to highlight
the tenuous relationship between performance and description and the ways that
the two are not easily separated. As Foley has observed, this difficulty gains
special purchase within this play: "In the _Bacchae_ myth and action, odes and
iambic scenes, are intertwined from the start, as is appropriate in a plot that
represents a penetration of the secular world by a divine power" (@Foley1985
207–208). The play's moments of narrative action ensure that, like Pentheus at
918–922, the audience experiences double vision by seeing the narrative reports
onstage and by witnessing the action in their minds.^[See @Zeitlin1994 [152]
for how, as she puts it, Euripides "[expands] the traditional oracular quality
of graphic signs into theatrical experimentation with the special powers of the
visual imagination in the world of the stage."]

"The action of the play gradually becomes, until the final scenes, fully
ritualized and mythical, and Dionysiac poetry transforms reality as the chorus
becomes one with and even predicts the action" (@Foley1985 208).

"Whatever we conclude occurs in the palace scene [576–603], whether nothing
at all or a major or minor change in the stage building, no stage business at
this point could adequately imitate the apocalyptic destruction of the palace,
including lightning and earthquake, which the chorus sees while Pentheus does
not. The miracle must become for the audience more symbolic and prophetic than
realistic. The audience sees not a miracle, but a chorus enacting the experience
of a miracle, or presenting a theatrical illusion." (@Foley1985 221)


"It is not hard to see that any system of classification that could comprehend
all of these figures in a single category would tell us little. But the
challenge need not be that of deciding who is and who is not a messenger.
Such a project, I submit ,would amount to a matter of definition. A more
productive approach, I suggest, is to recognize that all tragic messengers, near
messengers, and pseudomessengers have been produced more or less in accordance
with, or at variance with, a conventional, 'ideal' type. As such, these 'real'
tragic figures represent the texts' ongoing engagement with an idealized
conventional form." (@Barrett2002 97)

- Right, so this gets close but doesn't seem to be quite right. It's rather that
all speakers in tragedy --- including choruses! --- are more or less narrative.
(But is "narrative" opposed to "dramatic" here?)

"I will argue that Buxton's formulation, although laudable for the attention
it pays to rhetorical variation in the narratives it studies, simplifies the
status of the messengers in Euripides' play [_Bacchae_]. Rather than being
'firmly _within_ the drama,' the messengers occupy a place on the stage
very different from that of the other _dramatis personae_. A reading founded
on metatheatrical studies of the play will show that an important part of
_Bacchae_'s self-conscious interest is directed at the status of the messengers,
particularly with respect to how they define and are defined by Pentheus."
(@Barrett2002 103)


### Aeschylus' _Agamemnon_

Having looked at the play with the highest difference in narrativity between traditionally labeled messengers and other speakers, I turn now to the play where that difference is smallest: Aeschylus' _Agamemnon_.

![Chart showing per-sentence and rolling 7-sentence average narrativity for
Aeschylus' _Agamemnon_](../figures/ALL_DRAMA_line_position_tlg0085.tlg005.png)

While the speech Herald who announces the Achaeans' return from Troy (503–582,
620–680) shows several peaks on the narrativity scale, these peaks pale in
comparison to those of the parodos (40–257) and Clytaemestra's "Beacon Speech"
(281–316). This array of narrative speeches and speakers casts doubt on the
traditional labeling of messenger versus not—or even on messenger function
versus not—suggesting that we should think in terms of narrative moments rather
than narrative characters.

The parodos opens with a long narrative sentence (40–54):

| δέκατον μὲν ἔτος τόδʼ ἐπεὶ Πριάμου
| μέγας ἀντίδικος
| Μενέλαος ἄναξ ἠδʼ Ἀγαμέμνων,
| διθρόνου Διόθεν καὶ δισκήπτρου
| τιμῆς ὀχυρὸν ζεῦγος Ἀτρειδᾶν,
| στόλον Ἀργείων χιλιοναύτην
| τῆσδʼ ἀπὸ χώρας
| ἦραν, στρατιῶτιν ἀρωγήν,
| μεγάλ' ἐκ θυμοῦ κλάζοντες Ἄρη,
| τρόπον αἰγυπιῶν οἵτʼ ἐκπατίοις
| ἄλγεσι παίδων ὕπατοι λεχέων
| στροφοδινοῦνται
| πτερύγων ἐρετμοῖσιν ἐρεσσόμενοι,
| δεμνιοτήρη
| πόνον ὀρταλίχων ὀλέσαντες·

| This is the tenth year since
| the great plaintiff against Priam,
| Lord Menelaus–and with him Agamemnon–
| the strong yoked Atreids, consisting of
| two-throned and two-sceptred honor from Zeus,
| set sail from this land the thousand-shipped
| fleet of Argives, their martial aid,
| crying out "Ares" loudly from their spirit,
| like vultures who, in lonely pains
| for their children,
| eddy around high above their nest,
| rowing with the oars of their wings,
| because they lost the bed-watching
| toil of their chicks.

The chorus quickly assumes a narrative stance in these opening anapests,
including a near-simile about the vultures and their nest. As with the parodos
in _Bacchae_, part of the narrative tilt here comes from extensive use of a
third-person perspective, but third-person verbs do not suffice on their own for
the model to classify a sentence as narrative. The following sentence, "Some god
on high—Apollo or Pan or Zeus—hears the shrill-screaming bird-cried lament of
these metics and sends a late-avenging [_hysteropoinos_] Erinys to their side"
(55–60), also operates only in the third-person, but the model does not classify
it as narrative. Although the model weights are a black box and difficult to
interpret, the model's discrimination between these two sentences suggests that
it picks up on genuine similarities and dissimilarities among passages that it
classifies as narrative and not. In this case, perhaps the vocabulary has some
influence: as @Griffin1986 [38 and passim] and others have noted, the Homeric
narrative tends to avoid subjects like vengeance (_poinē_).

"It is essential to observe that the reference to a 'sending' by Zeus does not
come until it has been firmly established (40ff.) that Agamemnon and Menelaus
are purusing a human quarrel (recalled at 62); and that it is led up to by the
simile of the vultures. And how are the vultures avenged? By a special divine
interposition? Did Aeschylus or his audience believe this? The vultures go
after the robbers and avenge themselves, wit hthe backing of divine powers."
(@Winnington-Ingram1983 86)

Narrativity falls over the course of the _Agamemnon_, such that these early
narrative moments establish the background before the play devolves into
murder and Cassandra's chaotic speech. This change over time—one of two
statistically significant decreases in narrativity, the other being Euripides'
_Electra_—points to a change from story to spectacle. Like the beacon-flame
itself, which Clytaemestra brings to rest upon the palace, the drama brings
the action from distant Troy to the theater in front of the audience's eyes.
As Winnington-Ingram observes, "To show [the events of _Agamemnon_] as the
inevitable product of past events, to accumulate foreboding, right up to
the striking blow, was well suited to the lyric mode which is employed"
(@Winnington-Ingram1983 73–74).

This tension builds not only thanks to the "lyric mode" of the drama but
also due to the audience's growing realization that they must reconcile their
internally visualized notion of narrated events with the action that gradually
unfolds on- and just-off-stage. As Rancière writes, "Literature teaches us
to choose between two interpretations: not two interpretations of the speech
or actions of others, but two interpretations of our own perceptions and the
feelings of affection that accompany them" (@Ranciere2005 101). _Agamemnon_,
by gradually realizing our own misunderstandings in the spectacle that plays
out before our eyes, builds tension by playing on the possibility that we have
misunderstood.

Narrative moments create this space for misunderstanding by refusing to
reconcile our interpretations with the other audience members' and with what
we see. Narrative thus creates the possibility for ruptured experience, for
the pleasure and pain that accompany the sudden realization that we have not
understood what we have heard and seen. Narrative moments thus also point to a
leap of faith, a suspension of disbelief and trust in the reporter that might
give way to betrayal. This trust and the possibilities that it raises stand at
the heart not only of the theatrical endeavor but of every day communication: at
a certain point, we must simply take our reporters at their word.

Although the narrativity trendlines for _Choephoroi_ and _Eumenides_ lack
statistical significance (their p-values are much greater than 0.05), taken with
_Agamemnon_ we can see a clear trend away from narrative and towards speech.

![Chart showing per-sentence and rolling 7-sentence average narrativity for
Aeschylus' _Choephoroi_](../figures/ALL_DRAMA_line_position_tlg0085.tlg006.png)

![Chart showing per-sentence and rolling 7-sentence average narrativity for
Aeschylus' _Eumenides_](../figures/ALL_DRAMA_line_position_tlg0085.tlg007.png)

Such a trend speaks to the trilogy's drive towards action that affects the
audience: as an etiological tale, the _Oresteia_'s move from narrative to
speech and action reinforces its explanation for the court on the Areopagus.
But this movement is not one of diegesis to mimesis; it is rather a movement
from narrative mimesis to dramatic mimesis, from representation by telling to
representation by doing. CITATION AND SUPPORTING EVIDENCE


### Euripides' _Suppliants_

_Suppliants_ does not have the same narrative echoes, but it's interesting
that only one speech registers as narrative, when it has both a herald and
a messenger.

### Sophocles' _Electra_

Unsurprisingly, Sophocles' _Electra_ ranks among the most striking examples
of the messenger function's high narravity, with the Tutor's speech sustaining
scores above 0.5 for nearly its entire duration.

![Scatter plot with a trendline showing seven-sentence average narrativity for
Sophocles' _Electra_](../figures/ALL_DRAMA_line_position_tlg0011.tlg005.png)

The rest of the play, by contrast, barely breaks above a narrativity score
of 0.2.

## Argument/interpretation

We have to stop thinking about "narrative characters" and start thinking about
"narrative moments."

"[A discipline] is a way of defining an idea of the thinkable, an idea of what
the objects of knowledge themselves can think and know" (@Ranciere2009 17).

- This applies to the interpretive functions on the stage as well

## Conclusion
