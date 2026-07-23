# Speechiness versus narrativity

What can a model trained on direct speech in the Homeric epics
tell us about narrativity?

## Measuring narrativity in tragedy using a model trained on epic

```txt
Messenger-speech sentences: 1547; other tragedy sentences: 24831
P(narrative) mean: messenger=0.3531, other=0.0619
Narrative classification rate: messenger=0.3581, other=0.0594

Mann-Whitney U (H1: messenger P(narrative) > other): U=26462518.5, p=6.642e-138
Fisher's exact (H1: messenger narrative rate > other): odds_ratio=8.8406, p=6.215e-241

Per-play P(narrative) mean, messenger vs. other:
is_messenger_speech     other  messenger
play
tlg0006.tlg001       0.056942   0.294052
tlg0006.tlg002       0.049092   0.298461
tlg0006.tlg003       0.027640   0.467079
tlg0006.tlg004       0.038139   0.405830
tlg0006.tlg005       0.049562   0.448286
tlg0006.tlg006       0.051691   0.492079
tlg0006.tlg007       0.054743   0.353074
tlg0006.tlg008       0.058613   0.552202
tlg0006.tlg009       0.064851   0.446833
tlg0006.tlg010       0.070514   0.513995
tlg0006.tlg011       0.098551        NaN
tlg0006.tlg012       0.051583   0.319694
tlg0006.tlg013       0.072374   0.267925
tlg0006.tlg014       0.064814   0.341563
tlg0006.tlg015       0.060795   0.457438
tlg0006.tlg016       0.051893   0.320008
tlg0006.tlg017       0.064536   0.566021
tlg0006.tlg018       0.059375   0.348191
tlg0006.tlg019       0.088118   0.267412
tlg0011.tlg001       0.061018   0.188827
tlg0011.tlg002       0.058135   0.197570
tlg0011.tlg003       0.060923   0.205493
tlg0011.tlg004       0.047265   0.236581
tlg0011.tlg005       0.031261   0.515711
tlg0011.tlg006       0.037798   0.215308
tlg0011.tlg007       0.039447   0.341272
tlg0085.tlg001       0.089090        NaN
tlg0085.tlg002       0.136093   0.439217
tlg0085.tlg003       0.065829        NaN
tlg0085.tlg004       0.120163   0.277719
tlg0085.tlg005       0.117430   0.157134
tlg0085.tlg006       0.083231        NaN
tlg0085.tlg007       0.058252        NaN

Per-play (messenger - other) P(narrative) difference:
play
tlg0006.tlg001    0.237110
tlg0006.tlg002    0.249369
tlg0006.tlg003    0.439439
tlg0006.tlg004    0.367692
tlg0006.tlg005    0.398725
tlg0006.tlg006    0.440388
tlg0006.tlg007    0.298331
tlg0006.tlg008    0.493589
tlg0006.tlg009    0.381982
tlg0006.tlg010    0.443481
tlg0006.tlg011   -0.098551
tlg0006.tlg012    0.268111
tlg0006.tlg013    0.195552
tlg0006.tlg014    0.276749
tlg0006.tlg015    0.396643
tlg0006.tlg016    0.268115
tlg0006.tlg017    0.501485
tlg0006.tlg018    0.288816
tlg0006.tlg019    0.179294
tlg0011.tlg001    0.127809
tlg0011.tlg002    0.139435
tlg0011.tlg003    0.144570
tlg0011.tlg004    0.189317
tlg0011.tlg005    0.484450
tlg0011.tlg006    0.177510
tlg0011.tlg007    0.301825
tlg0085.tlg001   -0.089090
tlg0085.tlg002    0.303124
tlg0085.tlg003   -0.065829
tlg0085.tlg004    0.157556
tlg0085.tlg005    0.039704
tlg0085.tlg006   -0.083231
tlg0085.tlg007   -0.058252
dtype: float64

Wilcoxon signed-rank across 33 plays (H1: messenger - other > 0): W=541.0, p=4.319e-08
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
