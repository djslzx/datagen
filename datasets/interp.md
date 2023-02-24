local
```
v18: datasets/random_100k.txt
v19: datasets/random_100k_simpl.txt
v20: datasets/evo-bigpop/bigpop-0-gen=0-99.txt
v21: datasets/evo-bigpop/bigpop-0-gen=0-99-simpl.txt
```

remote
```
===
short run
===
289433 bigpop
289434 bigpop-simpl
289447 bigpop-filtered-simpl  # filter by baseline novelty (> 0.001)
289437 random
289438 random-simpl

===
longer run
===
291509 rand             # lr = 1e-3
291510 rand simpl       # lr = 1e-8
291508 bpop filt simpl  # lr = 2e-3
291507 bpop simpl       # lr = 2e-7
291506 bpop             # lr = 0.5 => very high error

===
lower learning rate on bpop
===
291573 bpop             # lr = 2e-7


===
set lr = 1e-7 for rands, filt
===
# lr = 1e-7
294290 rand simpl
294289 rand
294291 bpop filt simpl
```
