# Provenance: questions-gold-v1.jsonl

Derived mechanically from `data/qa/answer-gold-v1.jsonl` (Task 11, spec
2026-08-24-masked-pal-answering Step 2/3b adaptation).

`submission export` / `submission row-batches` require
`{"id": int>0, "question": str}` (`RawQuestion`), while `answer-gold-v1.jsonl`
keys its records by the string `question_id` (`retq_<64 hex>`). This file
assigns stable integer ids so the pipeline can run over exactly the gold
question set:

- Records are ordered by ascending `answer-gold-v1.jsonl` `question_id`
  (lexicographic on the retq hex string) and numbered `id = 1..58`.
- The mapping is fixed at creation time and must not be regenerated with a
  different rule while any decision file or evaluation built on it exists.
- Question text is copied byte-for-byte from `answer-gold-v1.jsonl`.

Mapping (id -> question_id):

- `1` -> `retq_00888e79366b91100dacb03137b33c72c620c121dbf3e5fab1db36a23b41733e`
- `2` -> `retq_0133c310e43952efcd5c9967409658db1fa361f4d47c2cbad0b8b2c5ec9040d9`
- `3` -> `retq_027ca04462e4cc19229848df810cb6c6aa404ddd4b19659fee6cabd954fbcfd2`
- `4` -> `retq_0a32a6d94a6e7bad8479d11ebbc10495710bc76f86ee2b0bde7d77462fa29d99`
- `5` -> `retq_113ffd796a5be812ba4e774e0f114c81be109064417ec267cdeca9cc693885a0`
- `6` -> `retq_143e63ad07075edafd9eb17370aa5ec20e9d558350c7c5bacc9feb287a1b5fa7`
- `7` -> `retq_1addba7ab56db7b390d9fb81cbd9f393d0fa17084e11b777839775271940a4cc`
- `8` -> `retq_1b05912ef66e0d457aaa4f6f1f6e9750bf1d63ca0680278ef53ccf5714858c40`
- `9` -> `retq_276accff7b518a3d1b034d720a140950c4be2c4703534b5f7e130f3e3e2d29ab`
- `10` -> `retq_2e2e9058f5006fa744aa84b2ab7381e765b503928badd6b6e7f8cec8c404898a`
- `11` -> `retq_3522c3a494d4824184a1e6c9670566b796a78b337c300e5b535e5473f5735ebb`
- `12` -> `retq_3873007dbd7f74d2badfa13c1e2069b57a7b5da460c69051818b069d6643f7c1`
- `13` -> `retq_3c0f00e3e21b90db5501ef16f298b0a84b6fb324e8e974519c6345ca239f02dc`
- `14` -> `retq_45efa3fa414bc23d0be7c3b47c465b9de7ebdcd66ad013ab3846b2b25a0a69af`
- `15` -> `retq_4ed36641e810a4c72f8383ff9869cfa6cbccff95bef524f08fce369a24525cf4`
- `16` -> `retq_518f173352f81ed630a71c6f8ad8166f1b38ce2c12d988c473d0b3f57c328041`
- `17` -> `retq_5a293140dab6370835bb93b17bb0503467626e4386d5e5ca5264afd3d2cff41b`
- `18` -> `retq_5adfcd4becee37f88a87d56f5a73cf8bd06c7a1ffea48f9f951c9997bdacafed`
- `19` -> `retq_5b1c24dfb50df43ff78d3c977e55b18c4e355cfe846f76f3bbaa42bb31144a9b`
- `20` -> `retq_5e5a7482abba820a4e8819833d85d90cbb45fcd33bd23e5214a29f9f4e861f5d`
- `21` -> `retq_5f2b6c1e1a04deed1c95f3d2e84a07c16b27c5cc28b33b03c8dfcba4ec056f74`
- `22` -> `retq_6a6a6024dd19f425b10ad6fb6e58f1bf8842e1886363b812d379f54a35e44655`
- `23` -> `retq_71b7db3ef64655693c358693bc68aadd106fd02f1b17c731c5de601b5cd9e42c`
- `24` -> `retq_75ebb973470157e3bdbe22b8d4453518d050eea7c56b43262c8e9fe4a2320652`
- `25` -> `retq_799bb5213c176968fa13d78639d80e3e479c3ce0e6d1531912c7fc9dd0ffdf84`
- `26` -> `retq_7a34830e0d5f92bb02fd76acc22290c32919cb6ea350bddd4952be9f448a6344`
- `27` -> `retq_7afd1c8e800a18c317c7f2b540f77433d318a5b3aa2c483fc333294615416da2`
- `28` -> `retq_7d8b6d321910394b5f0eaeeb22a747545e1d297ae6f08f8a3f9243c11c557125`
- `29` -> `retq_890e2025f771970a55cbd61a2aed04c292b80f83dfa8e23fa5761703e3a9c8ea`
- `30` -> `retq_8b06072cadb1887ff29e1bcc93dc8267d0d8e5c1a352a6f4c6a09bb4715dbd51`
- `31` -> `retq_8e43d1b9fd61ad81fb038d853c71c1c4b582ab7c502fae41ed33edf22737bc50`
- `32` -> `retq_93711e845932ba1c19b8ae5c9463022300555c0fcf07d488f8fc36c6b21066a3`
- `33` -> `retq_9490ea8fbfcc0834be9ca1d779411b8cf60ceeaf4816cbd05abc9c4ceeb48118`
- `34` -> `retq_9abb0ade76092d5443f249b43fb913340f16cfc5318c8631869c939e065534bf`
- `35` -> `retq_a2d5888138ba3e4af86938cf0854c85da519cf1d4c63f8e610fc722853f816a2`
- `36` -> `retq_ab353aadeb58f9ee5dee1609744a94b15ffc1c1c1297769d18b7a65f1b6b8e27`
- `37` -> `retq_b241522bcdab166c95e5e24ad9fb63a9eede7baf7c8432392db31274c063c389`
- `38` -> `retq_b48a824c13365c7b2537ece813ecc3f1e940d2765444b9b920a75418eb0e4a61`
- `39` -> `retq_b74d8b5b9e878a98bf9431932d39659660d3ba35571478baa6e678ce932c4a45`
- `40` -> `retq_bd2672094ccec8f44f032edf6dfee322681f67f22fd82169ce43ea89f199f505`
- `41` -> `retq_be9685d6a61366f955048464a7aae08d9229c346afe41f9dea1d29be8e7b0f1b`
- `42` -> `retq_bf660ff667f8d577c20962ceb66c154468642b12c827e7a52024b3fa2a5277bf`
- `43` -> `retq_c04ab744e55a62eb20943d9d43f55c5fd18f88ecbd530a61cc4a0c818af4ff17`
- `44` -> `retq_c259207b8436b0c78e72ad1c2445c9fb07dfedc6d952abb1504f77d34278da4c`
- `45` -> `retq_c287b7445583b29e918314c4eb672a5dcfb0f63ff9f5dc75e556527e587f7c8e`
- `46` -> `retq_cd301dce1be5405cc2554b6b92190b50a32e528652ef01cf5b0563d301feb401`
- `47` -> `retq_cd5ae1735b57c020abced2d8482d91ba1126a8a173570fbd059f8a0004f3116d`
- `48` -> `retq_cebc73ab05448dc576c2b50bc5dc1981d70d55ea892838ddf8313ae7ae041041`
- `49` -> `retq_d01701ab4c5e7d93a4af6f43f55ef17be41d4e2297131a4a388636edfdc8e8f2`
- `50` -> `retq_daf3295706f9a99546fdaafdda237fc03c541ae74522a82b46bd39ed6d863bb4`
- `51` -> `retq_de3c339b336e378295c8a5298d140ca41aa623690c6aed3616d4b9abc3dbe919`
- `52` -> `retq_e2059ca4277998aa17e8e41c3881e93960b982416b06a73b520628e942e263d5`
- `53` -> `retq_ea3b207f9bcd90985977155a65aa00de9d9127eaa488d93394daa8e2c569ab71`
- `54` -> `retq_ea502802099c055da44f3cf5e1352444a2df08eed91049e1a8e463f3277ba886`
- `55` -> `retq_f4d7995ffd78410b783739fab33df211f00ce268cea4a5403c2b33e0ccb7832f`
- `56` -> `retq_fca1bb6ec48ae0cac44cc391b5a073d7b0362f4b6927c319e09c637c252a451b`
- `57` -> `retq_fe7dd24ddc18840c779fdc814edc2c88945e449ba5130540703407f857ffa243`
- `58` -> `retq_febd815baa6cb0ab9489402e31e53356bbea6da84c14531558ef00ca9a06eb55`
