# Speechiness versus narrativity

What can a model trained on direct speech in the Homeric epics
tell us about narrativity?

## Measuring narrativity in tragedy using a model trained on epic

```txt
Messenger-speech sentences: 473; other tragedy sentences: 6181
P(narrative) mean: messenger=0.2213, other=0.0521
Narrative classification rate: messenger=0.2326, other=0.0514

Mann-Whitney U (H1: messenger P(narrative) > other): U=1815603.5, p=7.723e-19
Fisher's exact (H1: messenger narrative rate > other): odds_ratio=5.5870, p=4.782e-36

Per-play P(narrative) mean, messenger vs. other:
is_messenger_speech     other  messenger
play
tlg0006.tlg013       0.080036   0.145780
tlg0011.tlg001       0.052703   0.190959
tlg0011.tlg002       0.051836   0.181455
tlg0011.tlg003       0.039887   0.217637
tlg0011.tlg004       0.035832   0.238802
tlg0011.tlg005       0.024987   0.630883
tlg0011.tlg006       0.021881   0.264801
tlg0085.tlg005       0.120057   0.143941

Per-play (messenger - other) P(narrative) difference:
play
tlg0006.tlg013    0.065745
tlg0011.tlg001    0.138256
tlg0011.tlg002    0.129619
tlg0011.tlg003    0.177751
tlg0011.tlg004    0.202971
tlg0011.tlg005    0.605896
tlg0011.tlg006    0.242920
tlg0085.tlg005    0.023883
dtype: float64

Wilcoxon signed-rank across 8 plays (H1: messenger - other > 0): W=36.0, p=0.003906
```

## Narrativity of epic speakers

```txt
direct speech sentences: 7464; other sentences: 7686
P(narrative) mean: direct=0.0632, other=0.8993

Narrative classification rate: direct=0.0620, other=0.9076

Mann-Whitney U (H1: direct P(narrative) > other): U=2291776.0, p=1
Fisher's exact (H1: direct narrative rate > other): odds_ratio=0.0067, p=1
Per-work P(narrative) mean, direct vs. other:
                  is_direct_speech     other
                  P(narrative)
work
iliad             0.968293      0.065659
odyssey           0.792924      0.060630

Wilcoxon signed-rank across 2 plays (H1: direct - other > 0): W=0.0, p=1
```

See [./csv/epic_speakers_narrativity.csv](./csv/epic_speakers_narrativity.csv),
[./csv/epic_gender_narrativity.csv](./csv/epic_gender_narrativity.csv),
and [./csv/epic_speech_narrativity.csv](./csv/epic_speech_narrativity.csv)
for more detailed information.

## License

MIT License

Copyright © 2026 Charles Pletcher

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
