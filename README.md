# Measuring narrativity in tragedy using a model trained on epic

```
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
