# Day 13 Retrieval Evaluation V2

- Dataset fingerprint: `422df141c935d46bfd14302abec50f32380e6e4c012159f8ad0ae5560c8a446a`
- Questions: 70
- Diagnostic cutoff: 100
- Recall@10: 0.880952
- F2@R: 0.491346

## By intent

| Group | P@10 | R@3 | R@5 | R@10 | F2@10 | MRR | P@R | F2@R | TP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| compare | 0.165217 | 0.485507 | 0.673913 | 0.876812 | 0.453797 | 0.593168 | 0.376812 | 0.474308 | 38 |
| growth | 0.160870 | 0.623188 | 0.684783 | 0.902174 | 0.450142 | 0.638820 | 0.405797 | 0.480331 | 37 |
| lookup | 0.125000 | 0.638889 | 0.812500 | 0.864583 | 0.365884 | 0.633333 | 0.482639 | 0.518229 | 30 |

## By gold cardinality

| Group | P@10 | R@3 | R@5 | R@10 | F2@10 | MRR | P@R | F2@R | TP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| one_table | 0.086111 | 0.638889 | 0.805556 | 0.861111 | 0.307540 | 0.550926 | 0.388889 | 0.388889 | 31 |
| three_or_more | 0.258333 | 0.319444 | 0.520833 | 0.763889 | 0.546329 | 0.595833 | 0.340278 | 0.491184 | 31 |
| two_tables | 0.195455 | 0.636364 | 0.704545 | 0.977273 | 0.542929 | 0.752381 | 0.522727 | 0.659091 | 43 |

## By period cardinality

| Group | P@10 | R@3 | R@5 | R@10 | F2@10 | MRR | P@R | F2@R | TP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| multiple_periods | 0.145455 | 0.636364 | 0.757576 | 0.924242 | 0.435305 | 0.672294 | 0.454545 | 0.535354 | 48 |
| one_period | 0.154054 | 0.536036 | 0.695946 | 0.842342 | 0.410993 | 0.577027 | 0.394144 | 0.452096 | 57 |

## By statement filter

| Group | P@10 | R@3 | R@5 | R@10 | F2@10 | MRR | P@R | F2@R | TP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| filtered | 0.146667 | 0.574074 | 0.772222 | 0.892593 | 0.409356 | 0.588519 | 0.390741 | 0.438390 | 66 |
| unfiltered | 0.156000 | 0.600000 | 0.640000 | 0.860000 | 0.446032 | 0.682095 | 0.480000 | 0.586667 | 39 |

## By report era

| Group | P@10 | R@3 | R@5 | R@10 | F2@10 | MRR | P@R | F2@R | TP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2015_2019 | 0.204762 | 0.420635 | 0.654762 | 0.865079 | 0.493594 | 0.530952 | 0.265873 | 0.367978 | 43 |
| 2020_2023 | 0.134146 | 0.634146 | 0.756098 | 0.890244 | 0.407472 | 0.653310 | 0.463415 | 0.528455 | 55 |
| 2024_2025 | 0.087500 | 0.750000 | 0.750000 | 0.875000 | 0.312500 | 0.700000 | 0.625000 | 0.625000 | 7 |

## Per-question evidence

- `retq_00888e79366b91100dacb03137b33c72c620c121dbf3e5fab1db36a23b41733e`: failure=none; R@10=1.000000; diagnostic_gold_ranks=`{}`
- `retq_027ca04462e4cc19229848df810cb6c6aa404ddd4b19659fee6cabd954fbcfd2`: failure=none; R@10=1.000000; diagnostic_gold_ranks=`{}`
- `retq_035ab7b50aa16f2da0835e329105ef45254fff823aab0375464fe935992d7301`: failure=none; R@10=1.000000; diagnostic_gold_ranks=`{}`
- `retq_07495f696860ec468fa5a71f67fa84b86bcfdbcd1b55d1deaf3fbbbebf45a52b`: failure=none; R@10=1.000000; diagnostic_gold_ranks=`{}`
- `retq_0a32a6d94a6e7bad8479d11ebbc10495710bc76f86ee2b0bde7d77462fa29d99`: failure=none; R@10=1.000000; diagnostic_gold_ranks=`{}`
- `retq_0eaaf1a9804038a40987869bdcc2da226dbe3aba08d02943873b029d4f172848`: failure=zero_gold_hits; R@10=0.000000; diagnostic_gold_ranks=`{"tbl_43553d14f989fbed07d18b01f093f4c6f25d89e30917cde81ae3217bb747b42e": 20}`
- `retq_113ffd796a5be812ba4e774e0f114c81be109064417ec267cdeca9cc693885a0`: failure=none; R@10=1.000000; diagnostic_gold_ranks=`{}`
- `retq_140b9a346d497feed47dd86d6eb1a2bb0fa9c2c4ef873306cad2b631a8e347ce`: failure=none; R@10=1.000000; diagnostic_gold_ranks=`{}`
- `retq_15f357970d80fc9aa487f0da8d252b505d1217d23c365019000c60a366d1f620`: failure=none; R@10=1.000000; diagnostic_gold_ranks=`{}`
- `retq_184e75fa031778165edf54a7777b54c7cf874c7b1df60857be094b591c2564f1`: failure=none; R@10=1.000000; diagnostic_gold_ranks=`{}`
- `retq_18d416f9aaed3300ef83d93fd5743112c68fc2f6a11d518cc4f029be24e3766a`: failure=none; R@10=1.000000; diagnostic_gold_ranks=`{}`
- `retq_1b05912ef66e0d457aaa4f6f1f6e9750bf1d63ca0680278ef53ccf5714858c40`: failure=none; R@10=1.000000; diagnostic_gold_ranks=`{}`
- `retq_21d0fcc00dee1bd24d014fda92df6768482761e46ff4d1b809e4a8a61a8f2e58`: failure=none; R@10=1.000000; diagnostic_gold_ranks=`{}`
- `retq_276accff7b518a3d1b034d720a140950c4be2c4703534b5f7e130f3e3e2d29ab`: failure=none; R@10=1.000000; diagnostic_gold_ranks=`{}`
- `retq_2bf1a7ff817063f8a7583a5809b021257faa2bf25252135d085d38ffeb21a84f`: failure=none; R@10=1.000000; diagnostic_gold_ranks=`{}`
- `retq_2c6e03ad7f1a391d3673e2d8148a083ee759a1c1149c6f379241323f8b20221d`: failure=none; R@10=1.000000; diagnostic_gold_ranks=`{}`
- `retq_3311745c98ba5efc0f6b7e78f74fbb69d061ec16f6f266fa0d82f83b1564dee0`: failure=none; R@10=1.000000; diagnostic_gold_ranks=`{}`
- `retq_3e6c4ae3ff4d30628320182dce41772a231e92f8cb6d740a476bccdee64f5b06`: failure=none; R@10=1.000000; diagnostic_gold_ranks=`{}`
- `retq_46612a9c276fa94b4b682731803874a302616a3d4aaea063b36609e4603b7196`: failure=zero_gold_hits; R@10=0.000000; diagnostic_gold_ranks=`{"tbl_c1bf93f626705cff88c7f64eb14839c9d3993ce49cf334053eb59f9c5a8de9a0": 42}`
- `retq_4ae188f21b74a920b9293a7898f06bc80be5d7c48c87e2b2cc6731194768ac42`: failure=none; R@10=1.000000; diagnostic_gold_ranks=`{}`
- `retq_4af1d194037feec883a2ad351fad2d857c5a714df82de48c323b5b03da976fd4`: failure=partial_gold_hits; R@10=0.666667; diagnostic_gold_ranks=`{"tbl_5f59924a66120962e7825d29f63bfd67f40f97052199e33d8a0cf9284a116caf": 12}`
- `retq_4ed36641e810a4c72f8383ff9869cfa6cbccff95bef524f08fce369a24525cf4`: failure=none; R@10=1.000000; diagnostic_gold_ranks=`{}`
- `retq_518fddfe44aa2ba59c07c69fba10ccc85837f034cd9bd0f31802eefd97757c7c`: failure=none; R@10=1.000000; diagnostic_gold_ranks=`{}`
- `retq_5a293140dab6370835bb93b17bb0503467626e4386d5e5ca5264afd3d2cff41b`: failure=none; R@10=1.000000; diagnostic_gold_ranks=`{}`
- `retq_5adfcd4becee37f88a87d56f5a73cf8bd06c7a1ffea48f9f951c9997bdacafed`: failure=none; R@10=1.000000; diagnostic_gold_ranks=`{}`
- `retq_5bc82c57026f9a9c465c6de9a5423aa4dcc47c3bcb909980e2a8b34312c723e5`: failure=partial_gold_hits; R@10=0.750000; diagnostic_gold_ranks=`{"tbl_d1a533562e37f19983730375b041745f25652491d0544a9e5bc8fe6acf8085c1": 16}`
- `retq_5e5a7482abba820a4e8819833d85d90cbb45fcd33bd23e5214a29f9f4e861f5d`: failure=none; R@10=1.000000; diagnostic_gold_ranks=`{}`
- `retq_5e780dd26bf4c16168dc8f823b62918e50fe52d1acd0e673f1ac1a8bfc390dbd`: failure=none; R@10=1.000000; diagnostic_gold_ranks=`{}`
- `retq_6a6a6024dd19f425b10ad6fb6e58f1bf8842e1886363b812d379f54a35e44655`: failure=none; R@10=1.000000; diagnostic_gold_ranks=`{}`
- `retq_70fa3ba240c02f67e25d7f74451a303b71bdab96a0a660807d3f1d48621e2318`: failure=none; R@10=1.000000; diagnostic_gold_ranks=`{}`
- `retq_76c79261bdcb719e588c362ee794146de50e423a79e93c14949057684cc02dcf`: failure=zero_gold_hits; R@10=0.000000; diagnostic_gold_ranks=`{"tbl_c1bf93f626705cff88c7f64eb14839c9d3993ce49cf334053eb59f9c5a8de9a0": 44}`
- `retq_799bb5213c176968fa13d78639d80e3e479c3ce0e6d1531912c7fc9dd0ffdf84`: failure=none; R@10=1.000000; diagnostic_gold_ranks=`{}`
- `retq_7a23e39cc00a015db310ae9865241133d6c2bebebb760ab4a9b0341c96de6628`: failure=none; R@10=1.000000; diagnostic_gold_ranks=`{}`
- `retq_7afd1c8e800a18c317c7f2b540f77433d318a5b3aa2c483fc333294615416da2`: failure=none; R@10=1.000000; diagnostic_gold_ranks=`{}`
- `retq_7d8b6d321910394b5f0eaeeb22a747545e1d297ae6f08f8a3f9243c11c557125`: failure=none; R@10=1.000000; diagnostic_gold_ranks=`{}`
- `retq_843c4f4d5dbc6284787d601a04e41c8bd5e98f2374d96a5432d69ad85daedbe8`: failure=none; R@10=1.000000; diagnostic_gold_ranks=`{}`
- `retq_859f3db5be0b80c43b04417d2369df915845fd9cfec70503dbe753f23ea43fe6`: failure=none; R@10=1.000000; diagnostic_gold_ranks=`{}`
- `retq_865b76cc9a17d3afd764551b50a9de5b68384b62c2366a2273804e80d1d999a1`: failure=none; R@10=1.000000; diagnostic_gold_ranks=`{}`
- `retq_8e43d1b9fd61ad81fb038d853c71c1c4b582ab7c502fae41ed33edf22737bc50`: failure=none; R@10=1.000000; diagnostic_gold_ranks=`{}`
- `retq_9024e98663eb4d6d935ed2bd5c014bdeff9ba0d94a7e62038833deb6ff1b464a`: failure=none; R@10=1.000000; diagnostic_gold_ranks=`{}`
- `retq_905e8568becf90520e588c3292c269b0a7030ba5ec4eff4f4cbcf8d21a0fba23`: failure=zero_gold_hits; R@10=0.000000; diagnostic_gold_ranks=`{"tbl_c1bf93f626705cff88c7f64eb14839c9d3993ce49cf334053eb59f9c5a8de9a0": 21}`
- `retq_940e4706d5eea84a3bc3a90d109ddc7298d497fb90cbf4899fb043edb2268064`: failure=none; R@10=1.000000; diagnostic_gold_ranks=`{}`
- `retq_9490ea8fbfcc0834be9ca1d779411b8cf60ceeaf4816cbd05abc9c4ceeb48118`: failure=none; R@10=1.000000; diagnostic_gold_ranks=`{}`
- `retq_9a05b36a4b132789fe7ddd37f32f3ea8f6d14b2f51bef2ca73991c166adfa1d5`: failure=zero_gold_hits; R@10=0.000000; diagnostic_gold_ranks=`{"tbl_14356d221adccea2fca9acbeee9d317b6f47f366293dac5df7af6e949304a3a7": 19, "tbl_7178f397f374720112e3934ca2ab11c3be497c59c51d32701d91706486988430": 12, "tbl_b2157506dc11a7cd6a1f3a6e30090d1f696a96da3c3325119b4489a2adb7f0d1": 21}`
- `retq_9abb0ade76092d5443f249b43fb913340f16cfc5318c8631869c939e065534bf`: failure=none; R@10=1.000000; diagnostic_gold_ranks=`{}`
- `retq_9cf253d87eebc6cff165ab32f1d759d9779d11bd67331bb56806fcf6af78a8c4`: failure=none; R@10=1.000000; diagnostic_gold_ranks=`{}`
- `retq_9ecbb4727d458b5de69ed6ff3355d237b4482e86765f3b9d339c37f86c66c3d7`: failure=none; R@10=1.000000; diagnostic_gold_ranks=`{}`
- `retq_a2d5888138ba3e4af86938cf0854c85da519cf1d4c63f8e610fc722853f816a2`: failure=none; R@10=1.000000; diagnostic_gold_ranks=`{}`
- `retq_a51c4303ed6e770889e289f07ef1704ee7ccd896f35768a69a203118062b9e16`: failure=none; R@10=1.000000; diagnostic_gold_ranks=`{}`
- `retq_a5af50731f92719510085938eae0f29f7cb39b3b0457e15e99748f5f61a22a86`: failure=none; R@10=1.000000; diagnostic_gold_ranks=`{}`
- `retq_a92e064c27aea88ce6b2d4d81bae32e94fc23ac351036e8fd88058ed484d3d62`: failure=zero_gold_hits; R@10=0.000000; diagnostic_gold_ranks=`{"tbl_4a47c080f3d554db3a4571de05947364e88cc56c9d3c4072f8ec2ba0360ac805": 12}`
- `retq_af86db7ee5e169a65e2211bd3a20af63c4ecd906bf97d4572b15d0b6ea01859b`: failure=zero_gold_hits; R@10=0.000000; diagnostic_gold_ranks=`{"tbl_48dc6bdecfc04c5fece016f1a2e710e71b7576fb8ed92e482ec71ce61a6345fb": 17, "tbl_9eb0512b316a80754491a8cfb2acac52b79c8cd1c89e7c366f272a27f846a470": 18, "tbl_ff659cea1878c04aa499e1360803b41e95dcf12aea9bee5859dbbb483a115bbf": 19}`
- `retq_b241522bcdab166c95e5e24ad9fb63a9eede7baf7c8432392db31274c063c389`: failure=partial_gold_hits; R@10=0.500000; diagnostic_gold_ranks=`{"tbl_047406408377a99a416e8363bae707fc49347d465958e3f8f9556ba82075547b": 11}`
- `retq_b48a824c13365c7b2537ece813ecc3f1e940d2765444b9b920a75418eb0e4a61`: failure=none; R@10=1.000000; diagnostic_gold_ranks=`{}`
- `retq_b74d8b5b9e878a98bf9431932d39659660d3ba35571478baa6e678ce932c4a45`: failure=none; R@10=1.000000; diagnostic_gold_ranks=`{}`
- `retq_b8f5f17c7fa19ee92df6e919eb20b531855fc9c9fc46294da23b13d9025070de`: failure=partial_gold_hits; R@10=0.750000; diagnostic_gold_ranks=`{"tbl_add6276f804803e43654a26340b64c08c31cb2a83b931af1819d8d94d41aebf1": 11}`
- `retq_bf660ff667f8d577c20962ceb66c154468642b12c827e7a52024b3fa2a5277bf`: failure=none; R@10=1.000000; diagnostic_gold_ranks=`{}`
- `retq_c04ab744e55a62eb20943d9d43f55c5fd18f88ecbd530a61cc4a0c818af4ff17`: failure=none; R@10=1.000000; diagnostic_gold_ranks=`{}`
- `retq_c069554491e65164bb302c37f0d9c83d0283546dbc61e26448e2cc5512cb06e2`: failure=none; R@10=1.000000; diagnostic_gold_ranks=`{}`
- `retq_c3b64537544ba45a111dd7a0aced43190d1073ed76f5927986e770e74c0c08b5`: failure=none; R@10=1.000000; diagnostic_gold_ranks=`{}`
- `retq_d01701ab4c5e7d93a4af6f43f55ef17be41d4e2297131a4a388636edfdc8e8f2`: failure=none; R@10=1.000000; diagnostic_gold_ranks=`{}`
- `retq_d18b935bc13fd5a2ddaae8ac584243d78c0d55b66468f26d18e8d92bd5af1dcd`: failure=none; R@10=1.000000; diagnostic_gold_ranks=`{}`
- `retq_d232835d75041d3a6a2f1b8c143f5cb34e83df0ce9dbfd82e70510b073045214`: failure=none; R@10=1.000000; diagnostic_gold_ranks=`{}`
- `retq_daf3295706f9a99546fdaafdda237fc03c541ae74522a82b46bd39ed6d863bb4`: failure=none; R@10=1.000000; diagnostic_gold_ranks=`{}`
- `retq_de3c339b336e378295c8a5298d140ca41aa623690c6aed3616d4b9abc3dbe919`: failure=none; R@10=1.000000; diagnostic_gold_ranks=`{}`
- `retq_df6f68634d600a5378892f46c619a80464948abd11da8a233fe225e749590992`: failure=none; R@10=1.000000; diagnostic_gold_ranks=`{}`
- `retq_e2059ca4277998aa17e8e41c3881e93960b982416b06a73b520628e942e263d5`: failure=none; R@10=1.000000; diagnostic_gold_ranks=`{}`
- `retq_ea3b207f9bcd90985977155a65aa00de9d9127eaa488d93394daa8e2c569ab71`: failure=none; R@10=1.000000; diagnostic_gold_ranks=`{}`
- `retq_ea502802099c055da44f3cf5e1352444a2df08eed91049e1a8e463f3277ba886`: failure=none; R@10=1.000000; diagnostic_gold_ranks=`{}`
- `retq_f4d7995ffd78410b783739fab33df211f00ce268cea4a5403c2b33e0ccb7832f`: failure=none; R@10=1.000000; diagnostic_gold_ranks=`{}`
