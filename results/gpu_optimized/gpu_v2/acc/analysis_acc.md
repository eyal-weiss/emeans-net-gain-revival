# EXPLORATORY: no freeze vs freeze on the PR head driver (79440bc), optimized GPU code

34 cells (as parts A/B), early stop on; values: geometric mean ratio over cells per seed, then mean over seeds with a 95% t interval; wins = seeds with a lower mean log ratio. Not pre-registered.

## T = 2 (34 cells, seeds 30-34)

- ng vs pr, test: -1.6% [-7.0%, +4.1%], lower on 3/5 seeds
- ng vs pr, train: -12.2% [-17.3%, -6.7%], lower on 5/5 seeds
- ng vs pr, time (scored runs, shared GPU): 1.45

- ngn vs pr, test: -3.2% [-8.7%, +2.7%], lower on 3/5 seeds
- ngn vs pr, train: -15.8% [-21.2%, -10.1%], lower on 5/5 seeds
- ngn vs pr, time (scored runs, shared GPU): 1.68

- ngn vs ng, test: -1.7% [-3.5%, +0.3%], lower on 5/5 seeds
- ngn vs ng, train: -4.2% [-5.5%, -2.9%], lower on 5/5 seeds
- ngn vs ng, time (scored runs, shared GPU): 1.16

Per cell, ngn vs ng (geometric mean over seeds):

| cell | test | train | moves ng / ngn |
|---|---|---|---|
| census_z 256 | +0.0% | +0.0% | 61 / 61 |
| codrna_z 256 | +0.0% | +0.0% | 52 / 52 |
| corel_z 256 | -0.0% | +0.0% | 124 / 124 |
| fashion_255 256 | -0.0% | +0.0% | 70 / 70 |
| geonames 256 | +0.0% | +0.0% | 98 / 98 |
| higgs_z 256 | +0.0% | +0.0% | 44 / 44 |
| ijcnn1_z 256 | +0.0% | +0.0% | 74 / 74 |
| kdd_z 256 | -0.1% | -11.0% | 157 / 175 |
| mnist_255 256 | +0.0% | +0.0% | 46 / 46 |
| msd_z 256 | -0.8% | -0.4% | 145 / 193 |
| power_z 256 | +0.0% | +0.0% | 130 / 130 |
| road3d_z 256 | +0.0% | +0.0% | 59 / 59 |
| sensit_z 256 | -0.4% | -0.2% | 166 / 199 |
| shuttle_z 256 | +0.3% | -0.4% | 382 / 472 |
| census_z 1024 | -0.1% | -0.0% | 261 / 264 |
| codrna_z 1024 | +0.0% | +0.0% | 366 / 366 |
| corel_z 1024 | -0.2% | -0.3% | 888 / 950 |
| fashion_255 1024 | -0.2% | -0.2% | 618 / 696 |
| geonames 1024 | +0.0% | +0.0% | 486 / 486 |
| higgs_z 1024 | -0.1% | -0.1% | 304 / 338 |
| ijcnn1_z 1024 | -0.2% | -0.2% | 489 / 525 |
| kdd_z 1024 | -1.7% | -50.1% | 790 / 1040 |
| mnist_255 1024 | -0.2% | -0.1% | 486 / 531 |
| msd_z 1024 | -0.4% | -0.1% | 617 / 849 |
| power_z 1024 | +0.0% | +0.0% | 624 / 624 |
| road3d_z 1024 | +0.0% | +0.0% | 311 / 311 |
| sensit_z 1024 | -0.6% | -0.1% | 674 / 854 |
| shuttle_z 1024 | -29.3% | -2.4% | 1028 / 1323 |
| census_z 4096 | -0.1% | -0.1% | 1164 / 1195 |
| codrna_z 4096 | -0.1% | -0.1% | 2106 / 2176 |
| geonames 4096 | +0.2% | -0.0% | 2607 / 2688 |
| kdd_z 4096 | -15.6% | -45.0% | 2894 / 3519 |
| msd_z 4096 | -0.3% | +0.2% | 3775 / 4698 |
| road3d_z 4096 | -0.1% | -0.1% | 2053 / 2087 |

## T = 5 (34 cells, seeds 30-32)

- ng vs pr, test: +1.4% [-2.5%, +5.4%], lower on 1/3 seeds
- ng vs pr, train: -8.4% [-19.4%, +4.1%], lower on 3/3 seeds
- ng vs pr, time (scored runs, shared GPU): 1.60

- ngn vs pr, test: +1.5% [-2.7%, +5.9%], lower on 1/3 seeds
- ngn vs pr, train: -8.6% [-19.8%, +4.1%], lower on 3/3 seeds
- ngn vs pr, time (scored runs, shared GPU): 1.69

- ngn vs ng, test: +0.1% [-0.2%, +0.4%], lower on 1/3 seeds
- ngn vs ng, train: -0.2% [-0.8%, +0.4%], lower on 2/3 seeds
- ngn vs ng, time (scored runs, shared GPU): 1.06

Per cell, ngn vs ng (geometric mean over seeds):

| cell | test | train | moves ng / ngn |
|---|---|---|---|
| census_z 256 | +0.0% | +0.0% | 59 / 59 |
| codrna_z 256 | +0.0% | +0.0% | 50 / 50 |
| corel_z 256 | +0.0% | +0.0% | 86 / 86 |
| fashion_255 256 | +0.0% | +0.0% | 67 / 67 |
| geonames 256 | +0.0% | +0.0% | 92 / 92 |
| higgs_z 256 | +0.0% | +0.0% | 19 / 19 |
| ijcnn1_z 256 | +0.0% | +0.0% | 72 / 72 |
| kdd_z 256 | -1.4% | -0.6% | 173 / 180 |
| mnist_255 256 | +0.0% | +0.0% | 33 / 33 |
| msd_z 256 | -0.1% | -0.0% | 149 / 159 |
| power_z 256 | +0.0% | +0.0% | 113 / 113 |
| road3d_z 256 | +0.0% | +0.0% | 54 / 54 |
| sensit_z 256 | -0.1% | -0.1% | 222 / 226 |
| shuttle_z 256 | +0.5% | -1.7% | 589 / 620 |
| census_z 1024 | +0.0% | +0.0% | 267 / 267 |
| codrna_z 1024 | +0.0% | +0.0% | 276 / 276 |
| corel_z 1024 | +0.1% | -0.0% | 767 / 777 |
| fashion_255 1024 | -0.0% | -0.0% | 636 / 644 |
| geonames 1024 | +0.0% | +0.0% | 451 / 451 |
| higgs_z 1024 | -0.0% | -0.0% | 171 / 175 |
| ijcnn1_z 1024 | +0.1% | -0.0% | 450 / 452 |
| kdd_z 1024 | +4.5% | +3.0% | 1098 / 1198 |
| mnist_255 1024 | -0.0% | -0.0% | 396 / 400 |
| msd_z 1024 | -0.2% | +0.0% | 1239 / 1341 |
| power_z 1024 | +0.0% | +0.0% | 556 / 556 |
| road3d_z 1024 | +0.0% | +0.0% | 309 / 309 |
| sensit_z 1024 | -0.1% | -0.2% | 941 / 987 |
| shuttle_z 1024 | +0.1% | -0.7% | 1314 / 1336 |
| census_z 4096 | -0.0% | -0.0% | 1096 / 1098 |
| codrna_z 4096 | +0.0% | -0.0% | 1564 / 1564 |
| geonames 4096 | +0.2% | +0.1% | 2517 / 2550 |
| kdd_z 4096 | +0.8% | -7.0% | 3268 / 3511 |
| msd_z 4096 | -0.0% | -0.0% | 5836 / 6165 |
| road3d_z 4096 | +0.0% | -0.0% | 1744 / 1746 |

