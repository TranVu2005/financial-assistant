# Day 8 BM25 Retrieval Evaluation

- Dataset fingerprint: `422df141c935d46bfd14302abec50f32380e6e4c012159f8ad0ae5560c8a446a`
- Questions: 30
- Precision@10: 0.146667
- Recall@10: 0.883333
- F2@10: 0.431217

## Metrics by intent

| Intent | Precision@10 | Recall@10 | F2@10 | True positives |
| --- | ---: | ---: | ---: | ---: |
| compare | 0.170000 | 0.850000 | 0.472222 | 17 |
| growth | 0.180000 | 0.900000 | 0.500000 | 18 |
| lookup | 0.090000 | 0.900000 | 0.321429 | 9 |

## Query evidence

### retq_00888e79366b91100dacb03137b33c72c620c121dbf3e5fab1db36a23b41733e

- Question: So sánh lưu chuyển tiền thuần từ hoạt động kinh doanh của CTG giữa năm 2022 và năm 2023.
- Intent: compare
- Failure: none
- Predicted table IDs: tbl_5a6128f7f8df589d7546dd7e7364079ec4758a5747fe349ec88f9aa70020ebfd, tbl_e195adb9dfd5e9c080b12cd84e11aeb0526d9150681a22d2d119e55ebd8fc766, tbl_eb52eb65997633f7856a3a6a5477a8e2258b6d311c6fad0a2b32aef7318c82fc, tbl_4bd8bdfdba928ef4a1171833de1f7945887d13472b5dea375220b74462297d26, tbl_a43bcc4e35e1f88425f4f6a930a25a3970fa17f09065d67ed5da52c5aa7faa16, tbl_22c0746dc41f87e2f3e9dcdb3de72433951f418366a767bb335ff37c84ff98c1, tbl_72528b9e67999b76aff958479dd3f93230f505eefb530f70be0cec919b33fe6d, tbl_58a8f71151b6eef68c4e4b0f32549ed3fd91fe2fcbe753135fe1a71a62d7e343, tbl_db9daba556b7f680ec8142f7b3f03abc44ed98cb30bda0f91d5c9eea62490853, tbl_959a9c2be217581b913b656c5e41590d93b83f006a2b5fcc21c66cde7392ba03
- Gold table IDs: tbl_22c0746dc41f87e2f3e9dcdb3de72433951f418366a767bb335ff37c84ff98c1, tbl_5a6128f7f8df589d7546dd7e7364079ec4758a5747fe349ec88f9aa70020ebfd
- Missing gold table IDs: (none)
- Eligible documents: 409
- Empty reason: (none)
- Filter counts: company_codes=1964/1964; periods=33721/409
- Scores and matched tokens: tbl_5a6128f7f8df589d7546dd7e7364079ec4758a5747fe349ec88f9aa70020ebfd=16.966583 [2022, 2023, cash, chuyển, ctg, doanh, flow, hoạt, kinh, lưu, operating, thuần, tiền, từ, động]; tbl_e195adb9dfd5e9c080b12cd84e11aeb0526d9150681a22d2d119e55ebd8fc766=16.966583 [2022, 2023, cash, chuyển, ctg, doanh, flow, hoạt, kinh, lưu, operating, thuần, tiền, từ, động]; tbl_eb52eb65997633f7856a3a6a5477a8e2258b6d311c6fad0a2b32aef7318c82fc=16.900412 [2022, 2023, cash, chuyển, ctg, flow, hoạt, lưu, thuần, tiền, từ, động]; tbl_4bd8bdfdba928ef4a1171833de1f7945887d13472b5dea375220b74462297d26=16.696131 [2023, cash, chuyển, ctg, doanh, flow, hoạt, kinh, lưu, operating, thuần, tiền, từ, động]; tbl_a43bcc4e35e1f88425f4f6a930a25a3970fa17f09065d67ed5da52c5aa7faa16=16.696131 [2023, cash, chuyển, ctg, doanh, flow, hoạt, kinh, lưu, operating, thuần, tiền, từ, động]; tbl_22c0746dc41f87e2f3e9dcdb3de72433951f418366a767bb335ff37c84ff98c1=16.689587 [2022, cash, chuyển, ctg, doanh, flow, hoạt, kinh, lưu, operating, thuần, tiền, từ, động]; tbl_72528b9e67999b76aff958479dd3f93230f505eefb530f70be0cec919b33fe6d=16.689587 [2022, cash, chuyển, ctg, doanh, flow, hoạt, kinh, lưu, operating, thuần, tiền, từ, động]; tbl_58a8f71151b6eef68c4e4b0f32549ed3fd91fe2fcbe753135fe1a71a62d7e343=16.191204 [2023, cash, chuyển, ctg, flow, hoạt, lưu, thuần, tiền, từ, động]; tbl_db9daba556b7f680ec8142f7b3f03abc44ed98cb30bda0f91d5c9eea62490853=10.189269 [2022, 2023, ctg, doanh, hoạt, kinh, thuần, từ, động]; tbl_959a9c2be217581b913b656c5e41590d93b83f006a2b5fcc21c66cde7392ba03=9.365417 [2022, ctg, của, năm, so, sánh]

### retq_027ca04462e4cc19229848df810cb6c6aa404ddd4b19659fee6cabd954fbcfd2

- Question: Tính tốc độ tăng trưởng tổng tài sản riêng của GEG từ năm 2022 đến năm 2023.
- Intent: growth
- Failure: none
- Predicted table IDs: tbl_e22902ca93e0e314bb09ec37a330ada39678b4f578c553c1ac27e5ea379610ca, tbl_0dfb3a70b900e257c2630c3ba6f24a15c8bca002dc705a1ebe08359ef5f9e0df, tbl_e5c0a901bacbe5cfab2d10ae398a388af73e490b61721b1bc62bdf474d0d0525, tbl_ffea10f77cf9a157cfa54b1f7b286857197b6b4e2f656e87d8dfe4f8414b933d, tbl_93cbdbbc1e0cea8d74a783a4be18d7511befae1617f35d384a67465c781815aa, tbl_6cdda3d6c4b9af0db9fb168f7e0b3f2e8d328aadd0f3a72b5059b6ff3143f54e, tbl_fe92fb564decf93dc0c7191c7669249b74222875724539a5a008eedfb76f1dfc, tbl_0fc30404e8267176273eefab4865f4bc28d99205b95cb7b39c842e550dafb7b4, tbl_97c21b1016a674d36cabdf0ec75d22957b148500473cbc9b5e2130ae8300965b, tbl_a59d06c7e08d7a56a779b96823409dae8fd70b7a8c48308748eb0643f752c985
- Gold table IDs: tbl_e22902ca93e0e314bb09ec37a330ada39678b4f578c553c1ac27e5ea379610ca, tbl_e5c0a901bacbe5cfab2d10ae398a388af73e490b61721b1bc62bdf474d0d0525
- Missing gold table IDs: (none)
- Eligible documents: 355
- Empty reason: (none)
- Filter counts: company_codes=1744/1744; periods=33721/355
- Scores and matched tokens: tbl_e22902ca93e0e314bb09ec37a330ada39678b4f578c553c1ac27e5ea379610ca=9.742980 [2023, assets, geg, năm, riêng, sản, total, tài, tổng]; tbl_0dfb3a70b900e257c2630c3ba6f24a15c8bca002dc705a1ebe08359ef5f9e0df=9.174307 [2022, assets, geg, năm, sản, total, tài, tổng]; tbl_e5c0a901bacbe5cfab2d10ae398a388af73e490b61721b1bc62bdf474d0d0525=8.875970 [2022, assets, geg, năm, sản, total, tài, tổng]; tbl_ffea10f77cf9a157cfa54b1f7b286857197b6b4e2f656e87d8dfe4f8414b933d=8.535094 [2023, assets, geg, sản, total, tài, tổng]; tbl_93cbdbbc1e0cea8d74a783a4be18d7511befae1617f35d384a67465c781815aa=7.744334 [2023, assets, geg, sản, total, tài, tổng]; tbl_6cdda3d6c4b9af0db9fb168f7e0b3f2e8d328aadd0f3a72b5059b6ff3143f54e=7.737502 [2022, assets, geg, sản, total, tài, tổng]; tbl_fe92fb564decf93dc0c7191c7669249b74222875724539a5a008eedfb76f1dfc=7.737502 [2022, assets, geg, sản, total, tài, tổng]; tbl_0fc30404e8267176273eefab4865f4bc28d99205b95cb7b39c842e550dafb7b4=6.320115 [2023, của, geg, năm, riêng, total, tổng]; tbl_97c21b1016a674d36cabdf0ec75d22957b148500473cbc9b5e2130ae8300965b=5.866467 [2022, 2023, geg, năm, tài, đến]; tbl_a59d06c7e08d7a56a779b96823409dae8fd70b7a8c48308748eb0643f752c985=5.714056 [2023, của, geg, tổng, từ]

### retq_0a32a6d94a6e7bad8479d11ebbc10495710bc76f86ee2b0bde7d77462fa29d99

- Question: Tra cứu doanh thu thuần của NVL năm 2023.
- Intent: lookup
- Failure: none
- Predicted table IDs: tbl_02a3922bff2f3f78effd03ad727db83917f648ae3406b5efb08080d7a841515e, tbl_aafdc9bb01ff9acdd84c41ac075791c6a87201f1af479f41ddc8c1d4814a87d7, tbl_2162e6ddc897b5fe4ee3ad648cba4b05166749d5608c84419e28a804d7d48bff, tbl_6e9abf4aa6e5d9ed997cdeaccb996e31a6fef14603f1482d94689fcacfe6a3e4, tbl_f15acb078ad12ecdde9aa221b9d91c355a34a427e8db0bd5eae1cf790a597062, tbl_6a907b391e622401e760ee7fdd86fab2abf87a0b0dc0449be8cf0ec1677aff99, tbl_5445636bbfc1b33c52e9995d35df4d8746f556ac7aea235c3b31e1cb274e0740, tbl_4894cdf7668d3fd5586a82139776cc500ebd5fc3ef353f8b9eec7bfd7bba3b24, tbl_231a953b00742b7249d0076916eabb4bf7b8c3ebd68387da50282e6ba737a781, tbl_0bd42722eed465fc9ac833b876dd99361753cc5f7370815c81c74104b46c7486
- Gold table IDs: tbl_6e9abf4aa6e5d9ed997cdeaccb996e31a6fef14603f1482d94689fcacfe6a3e4
- Missing gold table IDs: (none)
- Eligible documents: 17
- Empty reason: (none)
- Filter counts: company_codes=1860/1860; periods=18228/233; statement_types=7862/17
- Scores and matched tokens: tbl_02a3922bff2f3f78effd03ad727db83917f648ae3406b5efb08080d7a841515e=9.856146 [2023, doanh, net, nvl, revenue, thu, thuần]; tbl_aafdc9bb01ff9acdd84c41ac075791c6a87201f1af479f41ddc8c1d4814a87d7=9.167949 [2023, của, doanh, net, nvl, revenue, thu, thuần]; tbl_2162e6ddc897b5fe4ee3ad648cba4b05166749d5608c84419e28a804d7d48bff=7.817772 [2023, doanh, net, nvl, revenue, thu, thuần]; tbl_6e9abf4aa6e5d9ed997cdeaccb996e31a6fef14603f1482d94689fcacfe6a3e4=7.023121 [2023, doanh, net, nvl, revenue, thu, thuần]; tbl_f15acb078ad12ecdde9aa221b9d91c355a34a427e8db0bd5eae1cf790a597062=4.212120 [2023, doanh, nvl, thu]; tbl_6a907b391e622401e760ee7fdd86fab2abf87a0b0dc0449be8cf0ec1677aff99=4.158857 [2023, doanh, nvl, thu]; tbl_5445636bbfc1b33c52e9995d35df4d8746f556ac7aea235c3b31e1cb274e0740=4.044482 [2023, của, nvl, năm]; tbl_4894cdf7668d3fd5586a82139776cc500ebd5fc3ef353f8b9eec7bfd7bba3b24=4.006441 [2023, của, nvl, năm]; tbl_231a953b00742b7249d0076916eabb4bf7b8c3ebd68387da50282e6ba737a781=4.005098 [2023, doanh, nvl]; tbl_0bd42722eed465fc9ac833b876dd99361753cc5f7370815c81c74104b46c7486=3.964604 [2023, doanh, nvl]

### retq_113ffd796a5be812ba4e774e0f114c81be109064417ec267cdeca9cc693885a0

- Question: So sánh doanh thu thuần về bán hàng và cung cấp dịch vụ của KHG giữa năm 2022 và năm 2023.
- Intent: compare
- Failure: none
- Predicted table IDs: tbl_c13ae0c50c647da5a1b0bc166d46660b8f713fcb202e344ec3fd40e70c3adc2d, tbl_ee81cc1b015207ce62bde5c5ec6aa42f82a2b639b7a703f5f00d72626c123a97, tbl_011c3b8764a6d4b0e025bd696501233c05eb7b49defe46e1cc5007b5794c51b4, tbl_b795c1b17487a1a67e4d1e3dd26e88e0b5f025f5964628ae1b224727e30e5959, tbl_fed883a2950883c1fe641123e9eea746557b8c20e6ba04de09a86eeaab4bb4f6, tbl_0ca6d656f0d9aadb4d80eeceea331b643e04f79a15d44b53eed75d9e6b74f951, tbl_7804c2a4adb572de0fb9b0d2964785387e7f37b3f0f390e8b984d4f1af16ca9c, tbl_d2591ed9f2e0b3395dd830373b80c69c6c0fbe02060a4c041d1527c94c48aef4, tbl_2c4478e2258897f79eaccd4dd1a86959423ecfcbf6fcfa70f2746999b21b10cd, tbl_a6afbf2a4e14b1077b554bc01d43553bfa60573be7d42393b01ccdd1943e487f
- Gold table IDs: tbl_c13ae0c50c647da5a1b0bc166d46660b8f713fcb202e344ec3fd40e70c3adc2d, tbl_ee81cc1b015207ce62bde5c5ec6aa42f82a2b639b7a703f5f00d72626c123a97
- Missing gold table IDs: (none)
- Eligible documents: 253
- Empty reason: (none)
- Filter counts: company_codes=806/806; periods=33721/253
- Scores and matched tokens: tbl_c13ae0c50c647da5a1b0bc166d46660b8f713fcb202e344ec3fd40e70c3adc2d=18.441416 [2022, 2023, bán, cung, cấp, doanh, dịch, hàng, khg, net, revenue, thu, thuần, và, về, vụ]; tbl_ee81cc1b015207ce62bde5c5ec6aa42f82a2b639b7a703f5f00d72626c123a97=17.547295 [2022, bán, cung, cấp, doanh, dịch, hàng, khg, net, revenue, thu, thuần, và, về, vụ]; tbl_011c3b8764a6d4b0e025bd696501233c05eb7b49defe46e1cc5007b5794c51b4=17.027874 [2022, bán, cung, cấp, doanh, dịch, hàng, khg, net, revenue, thu, thuần, và, vụ]; tbl_b795c1b17487a1a67e4d1e3dd26e88e0b5f025f5964628ae1b224727e30e5959=17.026678 [2023, bán, cung, doanh, dịch, hàng, khg, net, revenue, thu, thuần, và, về, vụ]; tbl_fed883a2950883c1fe641123e9eea746557b8c20e6ba04de09a86eeaab4bb4f6=16.224360 [2023, bán, cung, cấp, doanh, hàng, khg, net, revenue, thu, thuần, và, vụ]; tbl_0ca6d656f0d9aadb4d80eeceea331b643e04f79a15d44b53eed75d9e6b74f951=15.849414 [2022, 2023, bán, cung, doanh, dịch, hàng, khg, net, revenue, thu, thuần, và, vụ]; tbl_7804c2a4adb572de0fb9b0d2964785387e7f37b3f0f390e8b984d4f1af16ca9c=6.716981 [2022, 2023, của, dịch, khg, về]; tbl_d2591ed9f2e0b3395dd830373b80c69c6c0fbe02060a4c041d1527c94c48aef4=6.716981 [2022, 2023, của, dịch, khg, về]; tbl_2c4478e2258897f79eaccd4dd1a86959423ecfcbf6fcfa70f2746999b21b10cd=6.191847 [2022, 2023, của, khg, thu, và]; tbl_a6afbf2a4e14b1077b554bc01d43553bfa60573be7d42393b01ccdd1943e487f=5.986239 [2022, 2023, của, khg, thu, và]

### retq_1b05912ef66e0d457aaa4f6f1f6e9750bf1d63ca0680278ef53ccf5714858c40

- Question: So sánh doanh thu thuần của VGT giữa năm 2022 và năm 2023.
- Intent: compare
- Failure: none
- Predicted table IDs: tbl_973227f5e2dd1abdc06bfad4b15f0c4875c9deb51df912479158a3181a166a96, tbl_51da1cbb672a452be154cec9a681bc01efe01061ba8d049d299945a4d62f1b3b, tbl_700916b83a9c279e7d7aa92356ef6451c3788f9868142161ff9cfac1ac8b0ca7, tbl_d6f4a8333105e0a773679a13c1d4c69d57622e605e7005a692abbbdea2c9eaea, tbl_8d21a88d300d8b0951642ea6401c6909d8f1648a677c586b29bc9eb6e18bcf0c, tbl_1725ebbd226753a0b65386525ae024d153081927463e67eb6d2fb5da9f966ff8, tbl_2193d2ed58feff1b819652ad19599d479b2d90a7f2118cdb8cbdd653aabf1ee4, tbl_008c7d9ff7935743bb765bc8489bc1fc25b4610a235b8b5637d0ba133b713896, tbl_b35bdc026fae20558af3dc57027667b106f254b5a268c0a4185c305e8b10dea6, tbl_58f46ee9d5a58fb4b78cd097a7dfcc234a235a24a9893ee4a5c56836dc9bed64
- Gold table IDs: tbl_700916b83a9c279e7d7aa92356ef6451c3788f9868142161ff9cfac1ac8b0ca7, tbl_973227f5e2dd1abdc06bfad4b15f0c4875c9deb51df912479158a3181a166a96
- Missing gold table IDs: (none)
- Eligible documents: 342
- Empty reason: (none)
- Filter counts: company_codes=1700/1700; periods=33721/342
- Scores and matched tokens: tbl_973227f5e2dd1abdc06bfad4b15f0c4875c9deb51df912479158a3181a166a96=10.345021 [2023, doanh, net, revenue, thu, thuần, vgt]; tbl_51da1cbb672a452be154cec9a681bc01efe01061ba8d049d299945a4d62f1b3b=10.287153 [2023, doanh, net, revenue, thu, thuần, vgt]; tbl_700916b83a9c279e7d7aa92356ef6451c3788f9868142161ff9cfac1ac8b0ca7=10.280278 [2022, doanh, net, revenue, thu, thuần, vgt]; tbl_d6f4a8333105e0a773679a13c1d4c69d57622e605e7005a692abbbdea2c9eaea=6.996376 [2023, của, doanh, revenue, thu, vgt, và]; tbl_8d21a88d300d8b0951642ea6401c6909d8f1648a677c586b29bc9eb6e18bcf0c=6.763334 [2022, 2023, doanh, thu, thuần, vgt]; tbl_1725ebbd226753a0b65386525ae024d153081927463e67eb6d2fb5da9f966ff8=6.248589 [2023, của, doanh, revenue, thu, vgt, và]; tbl_2193d2ed58feff1b819652ad19599d479b2d90a7f2118cdb8cbdd653aabf1ee4=6.224142 [2023, của, doanh, revenue, thu, vgt, và]; tbl_008c7d9ff7935743bb765bc8489bc1fc25b4610a235b8b5637d0ba133b713896=6.219269 [2022, của, doanh, revenue, thu, vgt, và]; tbl_b35bdc026fae20558af3dc57027667b106f254b5a268c0a4185c305e8b10dea6=6.123594 [2023, của, doanh, revenue, thu, vgt, và]; tbl_58f46ee9d5a58fb4b78cd097a7dfcc234a235a24a9893ee4a5c56836dc9bed64=6.119066 [2022, của, doanh, revenue, thu, vgt, và]

### retq_276accff7b518a3d1b034d720a140950c4be2c4703534b5f7e130f3e3e2d29ab

- Question: So sánh lưu chuyển tiền thuần từ hoạt động kinh doanh của CEO giữa năm 2022 và năm 2023.
- Intent: compare
- Failure: none
- Predicted table IDs: tbl_3739f942a9c1843ffe9cd7a4d9edc68218501714148f91c5c89672f0e0e86189, tbl_6e4b99fbc765e5d4cfd6939f8b6344d1a95ddc2f33f07381dec1d179298a4331, tbl_563660a79f23c998cae04f5ba32553bae87422e6022f5773e07e8f5c145644a0, tbl_3bc0d37f4b340921d379dab7d121cebb8475001228ec98d942c418d288f53b04, tbl_04b890c73a5fd34f02ee1f3ac9d956763bcd851bbefe551cdf5830f37abb083a, tbl_0cc001ed54985f5912f4f6adf42ef0e69ed968cec28526b6f1556d35aece2320, tbl_e0bcfca93e2626e55291565db4b41e5329eacbfb3d1b6cfefb47ed83483a9633, tbl_3deb0dd025bded446da14d5898776ba94a63d53b8a40cc08275a520ea5598744, tbl_38ef7213e6380791e719d2738eb9584442c1dc5c1c6a5eeb659f433b71b07c7a, tbl_da6a4d3859a5904adfeacb8d714d04686a51f4cf98da4915aa6948d095508b88
- Gold table IDs: tbl_3739f942a9c1843ffe9cd7a4d9edc68218501714148f91c5c89672f0e0e86189, tbl_6e4b99fbc765e5d4cfd6939f8b6344d1a95ddc2f33f07381dec1d179298a4331
- Missing gold table IDs: (none)
- Eligible documents: 278
- Empty reason: (none)
- Filter counts: company_codes=1332/1332; periods=33721/278
- Scores and matched tokens: tbl_3739f942a9c1843ffe9cd7a4d9edc68218501714148f91c5c89672f0e0e86189=22.054537 [2023, cash, ceo, chuyển, doanh, flow, hoạt, kinh, lưu, năm, operating, thuần, tiền, từ, và, động]; tbl_6e4b99fbc765e5d4cfd6939f8b6344d1a95ddc2f33f07381dec1d179298a4331=21.536625 [2022, cash, ceo, chuyển, doanh, flow, hoạt, kinh, lưu, operating, thuần, tiền, từ, và, động]; tbl_563660a79f23c998cae04f5ba32553bae87422e6022f5773e07e8f5c145644a0=20.389980 [2023, cash, ceo, chuyển, doanh, flow, hoạt, kinh, lưu, năm, operating, thuần, tiền, từ, động]; tbl_3bc0d37f4b340921d379dab7d121cebb8475001228ec98d942c418d288f53b04=20.381113 [2022, cash, ceo, chuyển, doanh, flow, hoạt, kinh, lưu, năm, operating, thuần, tiền, từ, động]; tbl_04b890c73a5fd34f02ee1f3ac9d956763bcd851bbefe551cdf5830f37abb083a=14.264473 [2023, cash, ceo, chuyển, flow, hoạt, lưu, thuần, tiền, từ, và, động]; tbl_0cc001ed54985f5912f4f6adf42ef0e69ed968cec28526b6f1556d35aece2320=14.258120 [2022, cash, ceo, chuyển, flow, hoạt, lưu, thuần, tiền, từ, và, động]; tbl_e0bcfca93e2626e55291565db4b41e5329eacbfb3d1b6cfefb47ed83483a9633=9.027768 [2022, cash, ceo, chuyển, flow, lưu, năm, tiền]; tbl_3deb0dd025bded446da14d5898776ba94a63d53b8a40cc08275a520ea5598744=8.923436 [2022, 2023, cash, ceo, chuyển, flow, lưu, tiền]; tbl_38ef7213e6380791e719d2738eb9584442c1dc5c1c6a5eeb659f433b71b07c7a=8.869734 [2022, 2023, cash, ceo, chuyển, flow, lưu, tiền]; tbl_da6a4d3859a5904adfeacb8d714d04686a51f4cf98da4915aa6948d095508b88=8.785340 [2023, cash, ceo, chuyển, flow, lưu, năm, tiền]

### retq_46612a9c276fa94b4b682731803874a302616a3d4aaea063b36609e4603b7196

- Question: So sánh tổng tài sản hợp nhất của HDB giữa năm 2022 và năm 2023.
- Intent: compare
- Failure: zero_gold_hits
- Predicted table IDs: tbl_372401f3c194d40741dc2672604546ff4bb354f00dc3c0b9b0b8a2e697ed2510, tbl_1023e8dce509a03c9b05d6827a8c2178a530c665efaf58947721e0f8e7695ef6, tbl_c4f2b5873b56347e476b7e7cdac4202ef415d872371bb054101976f080c1ecb3, tbl_44ea0dfdb1ae6fb87b5feea9f3d0459610faa035bb346c65bd40c2866642bd66, tbl_70eff70e8607a22d5aefa3a3362de6aebc934b526d8096ea45ecdad5ab2d3e91, tbl_929b58991fe4338c757276556a424ddb430c2d44aca0703c3f4acca21f2fb2d6, tbl_ad7f3da068de1feba8aa6aeaa19e1ecec7bbf9c163a6ab47bb5b6aab9367530e, tbl_5dd01704e92cdbd559d6723cbce0c7a42177f939d8cd5b1f72fa414345980269, tbl_0b806ba5d2a01cecd33be5927a877bab3711ee1c2111d6796a4a7ab430338533, tbl_2f32b7520214715111a367b968521ceccfa86e45b0f20eb0aea152afdb795e6c
- Gold table IDs: tbl_c1bf93f626705cff88c7f64eb14839c9d3993ce49cf334053eb59f9c5a8de9a0
- Missing gold table IDs: tbl_c1bf93f626705cff88c7f64eb14839c9d3993ce49cf334053eb59f9c5a8de9a0
- Eligible documents: 672
- Empty reason: (none)
- Filter counts: company_codes=1835/1835; periods=33721/672
- Scores and matched tokens: tbl_372401f3c194d40741dc2672604546ff4bb354f00dc3c0b9b0b8a2e697ed2510=10.889311 [2023, assets, hdb, hợp, nhất, sản, total, tài, tổng]; tbl_1023e8dce509a03c9b05d6827a8c2178a530c665efaf58947721e0f8e7695ef6=10.883622 [2022, assets, hdb, hợp, nhất, sản, total, tài, tổng]; tbl_c4f2b5873b56347e476b7e7cdac4202ef415d872371bb054101976f080c1ecb3=10.414494 [2023, assets, của, hdb, năm, sản, total, tài, tổng, và]; tbl_44ea0dfdb1ae6fb87b5feea9f3d0459610faa035bb346c65bd40c2866642bd66=9.811822 [2023, assets, của, hdb, sản, total, tài, tổng, và]; tbl_70eff70e8607a22d5aefa3a3362de6aebc934b526d8096ea45ecdad5ab2d3e91=9.319831 [2023, assets, hdb, sản, total, tài, tổng]; tbl_929b58991fe4338c757276556a424ddb430c2d44aca0703c3f4acca21f2fb2d6=9.319831 [2023, assets, hdb, sản, total, tài, tổng]; tbl_ad7f3da068de1feba8aa6aeaa19e1ecec7bbf9c163a6ab47bb5b6aab9367530e=9.319831 [2023, assets, hdb, sản, total, tài, tổng]; tbl_5dd01704e92cdbd559d6723cbce0c7a42177f939d8cd5b1f72fa414345980269=9.313551 [2022, assets, hdb, sản, total, tài, tổng]; tbl_0b806ba5d2a01cecd33be5927a877bab3711ee1c2111d6796a4a7ab430338533=9.234558 [2023, assets, hdb, sản, total, tài, tổng]; tbl_2f32b7520214715111a367b968521ceccfa86e45b0f20eb0aea152afdb795e6c=9.234558 [2023, assets, hdb, sản, total, tài, tổng]

### retq_4ae188f21b74a920b9293a7898f06bc80be5d7c48c87e2b2cc6731194768ac42

- Question: Tra cứu tổng tài sản hợp nhất của ACB năm 2023.
- Intent: lookup
- Failure: none
- Predicted table IDs: tbl_1fea09c00553183173733966a79d1ad2795094a415e277a0034fc4dd8aa6b616, tbl_c4d774fe3b6b41739275aee795d588229ebb9e3c42c3b8d8603e2298d39d0acb, tbl_857e287a58e6dccf9d4978c2f0e458db1a4cdbe6ed08657cccd241bf104ffb78, tbl_0fbe34be5f3b1a78b7892752173842d14709c9f542060e336dce302b455a3423, tbl_246ba304052b51bc43f27ce66aa31241839dec967a273fa2ccbfaae9c8a58566, tbl_cd7b7022672e950a5aa9d9bc15ca146dac9b61e0fe9f03c4d26396ba4278cf24, tbl_17eb7606a77f488613cc04da83445dab3dc01082a04ae9c781e6a06ba156deda, tbl_25624fc3bfc359417a29274d84a28c07c70934c716f8992dd8ae33cbd5a421d8, tbl_367e91052d8de68625864f35c9cb8f4ee9bd9133d6ecc600fbcb5e53ac8605bd, tbl_a59e16dcd77cb38309eb5c53932deb11a7a6ee6abfb34645f74f521787a336a8
- Gold table IDs: tbl_1fea09c00553183173733966a79d1ad2795094a415e277a0034fc4dd8aa6b616
- Missing gold table IDs: (none)
- Eligible documents: 21
- Empty reason: (none)
- Filter counts: company_codes=2507/2507; periods=18228/270; statement_types=7745/21
- Scores and matched tokens: tbl_1fea09c00553183173733966a79d1ad2795094a415e277a0034fc4dd8aa6b616=10.832029 [2023, acb, assets, hợp, nhất, sản, total, tài, tổng]; tbl_c4d774fe3b6b41739275aee795d588229ebb9e3c42c3b8d8603e2298d39d0acb=9.976265 [2023, acb, assets, năm, sản, total, tài, tổng]; tbl_857e287a58e6dccf9d4978c2f0e458db1a4cdbe6ed08657cccd241bf104ffb78=9.555859 [2023, acb, assets, sản, total, tài, tổng]; tbl_0fbe34be5f3b1a78b7892752173842d14709c9f542060e336dce302b455a3423=9.248669 [2023, acb, assets, sản, total, tài, tổng]; tbl_246ba304052b51bc43f27ce66aa31241839dec967a273fa2ccbfaae9c8a58566=9.193978 [2023, acb, assets, sản, total, tài, tổng]; tbl_cd7b7022672e950a5aa9d9bc15ca146dac9b61e0fe9f03c4d26396ba4278cf24=9.122112 [2023, acb, assets, sản, total, tài, tổng]; tbl_17eb7606a77f488613cc04da83445dab3dc01082a04ae9c781e6a06ba156deda=8.544292 [2023, acb, assets, sản, total, tài, tổng]; tbl_25624fc3bfc359417a29274d84a28c07c70934c716f8992dd8ae33cbd5a421d8=8.333928 [2023, acb, assets, sản, total, tài, tổng]; tbl_367e91052d8de68625864f35c9cb8f4ee9bd9133d6ecc600fbcb5e53ac8605bd=8.333928 [2023, acb, assets, sản, total, tài, tổng]; tbl_a59e16dcd77cb38309eb5c53932deb11a7a6ee6abfb34645f74f521787a336a8=8.333928 [2023, acb, assets, sản, total, tài, tổng]

### retq_4ed36641e810a4c72f8383ff9869cfa6cbccff95bef524f08fce369a24525cf4

- Question: So sánh doanh thu thuần của NVL giữa năm 2022 và năm 2023.
- Intent: compare
- Failure: none
- Predicted table IDs: tbl_02a3922bff2f3f78effd03ad727db83917f648ae3406b5efb08080d7a841515e, tbl_1170276160b57a7fa7121eb0f833e88d3d90c0ad4d565851114ca39168cc7f08, tbl_aafdc9bb01ff9acdd84c41ac075791c6a87201f1af479f41ddc8c1d4814a87d7, tbl_2162e6ddc897b5fe4ee3ad648cba4b05166749d5608c84419e28a804d7d48bff, tbl_7986c240e412231ee3754abc28e958e2586d97c686cbd8f2c9c36df5c55ffa9b, tbl_873a7448428a338c209db22bf8c225cc271ba5d64340a700cec7266716471506, tbl_b8f32ef036f35f8f5a8bf3d7083da1ec559c9a6d5d017e99f3b7ad3fa45c8a42, tbl_6e9abf4aa6e5d9ed997cdeaccb996e31a6fef14603f1482d94689fcacfe6a3e4, tbl_77ba9cc02985be74b367bf1b18048b72eca5c0183ac473fafccc1be7f22a66a4, tbl_5445636bbfc1b33c52e9995d35df4d8746f556ac7aea235c3b31e1cb274e0740
- Gold table IDs: tbl_6e9abf4aa6e5d9ed997cdeaccb996e31a6fef14603f1482d94689fcacfe6a3e4, tbl_b8f32ef036f35f8f5a8bf3d7083da1ec559c9a6d5d017e99f3b7ad3fa45c8a42
- Missing gold table IDs: (none)
- Eligible documents: 421
- Empty reason: (none)
- Filter counts: company_codes=1860/1860; periods=33721/421
- Scores and matched tokens: tbl_02a3922bff2f3f78effd03ad727db83917f648ae3406b5efb08080d7a841515e=11.524703 [2022, 2023, doanh, net, nvl, revenue, thu, thuần, và]; tbl_1170276160b57a7fa7121eb0f833e88d3d90c0ad4d565851114ca39168cc7f08=9.247945 [2022, doanh, net, nvl, năm, revenue, thu, thuần]; tbl_aafdc9bb01ff9acdd84c41ac075791c6a87201f1af479f41ddc8c1d4814a87d7=9.167949 [2023, của, doanh, net, nvl, revenue, thu, thuần]; tbl_2162e6ddc897b5fe4ee3ad648cba4b05166749d5608c84419e28a804d7d48bff=9.140422 [2022, 2023, doanh, net, nvl, revenue, thu, thuần, và]; tbl_7986c240e412231ee3754abc28e958e2586d97c686cbd8f2c9c36df5c55ffa9b=9.061725 [2022, của, doanh, net, nvl, revenue, thu, thuần]; tbl_873a7448428a338c209db22bf8c225cc271ba5d64340a700cec7266716471506=8.907876 [2022, doanh, net, nvl, năm, revenue, thu, thuần]; tbl_b8f32ef036f35f8f5a8bf3d7083da1ec559c9a6d5d017e99f3b7ad3fa45c8a42=8.418895 [2022, doanh, net, nvl, revenue, thu, thuần, và]; tbl_6e9abf4aa6e5d9ed997cdeaccb996e31a6fef14603f1482d94689fcacfe6a3e4=8.125764 [2022, 2023, doanh, net, nvl, revenue, thu, thuần, và]; tbl_77ba9cc02985be74b367bf1b18048b72eca5c0183ac473fafccc1be7f22a66a4=7.984527 [2022, doanh, nvl, revenue, thu, thuần, và]; tbl_5445636bbfc1b33c52e9995d35df4d8746f556ac7aea235c3b31e1cb274e0740=6.894130 [2022, 2023, của, nvl, năm, so]

### retq_5a293140dab6370835bb93b17bb0503467626e4386d5e5ca5264afd3d2cff41b

- Question: Tính tốc độ tăng trưởng doanh thu thuần của NVL từ năm 2022 đến năm 2023.
- Intent: growth
- Failure: none
- Predicted table IDs: tbl_02a3922bff2f3f78effd03ad727db83917f648ae3406b5efb08080d7a841515e, tbl_1170276160b57a7fa7121eb0f833e88d3d90c0ad4d565851114ca39168cc7f08, tbl_aafdc9bb01ff9acdd84c41ac075791c6a87201f1af479f41ddc8c1d4814a87d7, tbl_7986c240e412231ee3754abc28e958e2586d97c686cbd8f2c9c36df5c55ffa9b, tbl_873a7448428a338c209db22bf8c225cc271ba5d64340a700cec7266716471506, tbl_b8f32ef036f35f8f5a8bf3d7083da1ec559c9a6d5d017e99f3b7ad3fa45c8a42, tbl_2162e6ddc897b5fe4ee3ad648cba4b05166749d5608c84419e28a804d7d48bff, tbl_6e9abf4aa6e5d9ed997cdeaccb996e31a6fef14603f1482d94689fcacfe6a3e4, tbl_5a1009d656542e10e2125ba9d36251df3191cd19944a806da71132e5c3c378de, tbl_f7a76e75e51daf18ed13d40076305b8850dea63988778e50ac0e6de2e62824f5
- Gold table IDs: tbl_6e9abf4aa6e5d9ed997cdeaccb996e31a6fef14603f1482d94689fcacfe6a3e4, tbl_b8f32ef036f35f8f5a8bf3d7083da1ec559c9a6d5d017e99f3b7ad3fa45c8a42
- Missing gold table IDs: (none)
- Eligible documents: 421
- Empty reason: (none)
- Filter counts: company_codes=1860/1860; periods=33721/421
- Scores and matched tokens: tbl_02a3922bff2f3f78effd03ad727db83917f648ae3406b5efb08080d7a841515e=10.524113 [2022, 2023, doanh, net, nvl, revenue, thu, thuần]; tbl_1170276160b57a7fa7121eb0f833e88d3d90c0ad4d565851114ca39168cc7f08=9.247945 [2022, doanh, net, nvl, năm, revenue, thu, thuần]; tbl_aafdc9bb01ff9acdd84c41ac075791c6a87201f1af479f41ddc8c1d4814a87d7=9.167949 [2023, của, doanh, net, nvl, revenue, thu, thuần]; tbl_7986c240e412231ee3754abc28e958e2586d97c686cbd8f2c9c36df5c55ffa9b=9.061725 [2022, của, doanh, net, nvl, revenue, thu, thuần]; tbl_873a7448428a338c209db22bf8c225cc271ba5d64340a700cec7266716471506=8.907876 [2022, doanh, net, nvl, năm, revenue, thu, thuần]; tbl_b8f32ef036f35f8f5a8bf3d7083da1ec559c9a6d5d017e99f3b7ad3fa45c8a42=8.322694 [2022, doanh, net, nvl, revenue, thu, thuần, từ]; tbl_2162e6ddc897b5fe4ee3ad648cba4b05166749d5608c84419e28a804d7d48bff=8.312964 [2022, 2023, doanh, net, nvl, revenue, thu, thuần]; tbl_6e9abf4aa6e5d9ed997cdeaccb996e31a6fef14603f1482d94689fcacfe6a3e4=8.204388 [2022, 2023, doanh, net, nvl, revenue, thu, thuần, từ]; tbl_5a1009d656542e10e2125ba9d36251df3191cd19944a806da71132e5c3c378de=7.349585 [2023, doanh, nvl, thuần, tính, từ]; tbl_f7a76e75e51daf18ed13d40076305b8850dea63988778e50ac0e6de2e62824f5=7.349585 [2023, doanh, nvl, thuần, tính, từ]

### retq_5adfcd4becee37f88a87d56f5a73cf8bd06c7a1ffea48f9f951c9997bdacafed

- Question: Tính tốc độ tăng trưởng doanh thu thuần của VGT từ năm 2022 đến năm 2023.
- Intent: growth
- Failure: none
- Predicted table IDs: tbl_973227f5e2dd1abdc06bfad4b15f0c4875c9deb51df912479158a3181a166a96, tbl_51da1cbb672a452be154cec9a681bc01efe01061ba8d049d299945a4d62f1b3b, tbl_700916b83a9c279e7d7aa92356ef6451c3788f9868142161ff9cfac1ac8b0ca7, tbl_8d21a88d300d8b0951642ea6401c6909d8f1648a677c586b29bc9eb6e18bcf0c, tbl_764442325b509bebabf791dd86e4611653c8747dd2a0b91eb118693d13cc5610, tbl_2ad0a529c8f8b1ffb8b31d6a4066f9a73185e51c13aeebb9d6b85adbe117bed3, tbl_725902b206361faa3a131f440f7c4a37ec9694aca5929aee97577bf0c5adf60d, tbl_d6ff72345cb655f3c60e7c98bfe3684eb951f9d558415d6d2a366a658e1f8728, tbl_66c292ea71aa5f1bdecf02352702f8ce0ce3c78ed48c245e240538584a5c9383, tbl_3875d54d5ece1c78681aabd274c24c7d7180a57f425a97b83bc0b70322da92e2
- Gold table IDs: tbl_700916b83a9c279e7d7aa92356ef6451c3788f9868142161ff9cfac1ac8b0ca7, tbl_973227f5e2dd1abdc06bfad4b15f0c4875c9deb51df912479158a3181a166a96
- Missing gold table IDs: (none)
- Eligible documents: 342
- Empty reason: (none)
- Filter counts: company_codes=1700/1700; periods=33721/342
- Scores and matched tokens: tbl_973227f5e2dd1abdc06bfad4b15f0c4875c9deb51df912479158a3181a166a96=10.345021 [2023, doanh, net, revenue, thu, thuần, vgt]; tbl_51da1cbb672a452be154cec9a681bc01efe01061ba8d049d299945a4d62f1b3b=10.287153 [2023, doanh, net, revenue, thu, thuần, vgt]; tbl_700916b83a9c279e7d7aa92356ef6451c3788f9868142161ff9cfac1ac8b0ca7=10.280278 [2022, doanh, net, revenue, thu, thuần, vgt]; tbl_8d21a88d300d8b0951642ea6401c6909d8f1648a677c586b29bc9eb6e18bcf0c=6.763334 [2022, 2023, doanh, thu, thuần, vgt]; tbl_764442325b509bebabf791dd86e4611653c8747dd2a0b91eb118693d13cc5610=6.687061 [2022, của, doanh, thu, thuần, từ, vgt]; tbl_2ad0a529c8f8b1ffb8b31d6a4066f9a73185e51c13aeebb9d6b85adbe117bed3=6.668076 [2023, của, thu, từ, vgt, đến]; tbl_725902b206361faa3a131f440f7c4a37ec9694aca5929aee97577bf0c5adf60d=6.524985 [2023, của, doanh, thu, thuần, từ, vgt]; tbl_d6ff72345cb655f3c60e7c98bfe3684eb951f9d558415d6d2a366a658e1f8728=6.524985 [2023, của, doanh, thu, thuần, từ, vgt]; tbl_66c292ea71aa5f1bdecf02352702f8ce0ce3c78ed48c245e240538584a5c9383=6.519297 [2022, của, doanh, thu, thuần, từ, vgt]; tbl_3875d54d5ece1c78681aabd274c24c7d7180a57f425a97b83bc0b70322da92e2=6.492417 [2023, của, doanh, thu, thuần, từ, vgt]

### retq_5e780dd26bf4c16168dc8f823b62918e50fe52d1acd0e673f1ac1a8bfc390dbd

- Question: Tính tốc độ tăng trưởng tổng tài sản hợp nhất của ACB từ năm 2022 đến năm 2023.
- Intent: growth
- Failure: none
- Predicted table IDs: tbl_1fea09c00553183173733966a79d1ad2795094a415e277a0034fc4dd8aa6b616, tbl_a6c2f4b94692ec0008a9f9b550f9d3c27eceea641af58b740c93ac04817e7e44, tbl_c4d774fe3b6b41739275aee795d588229ebb9e3c42c3b8d8603e2298d39d0acb, tbl_857e287a58e6dccf9d4978c2f0e458db1a4cdbe6ed08657cccd241bf104ffb78, tbl_d5d5b86f6ef69e35a78bb9db82b84dd28fa373134fda1aa81138aefa966fc3ea, tbl_0fbe34be5f3b1a78b7892752173842d14709c9f542060e336dce302b455a3423, tbl_002b5e0f4b961fa5dec48576b2ffd1c352cdc75f5f7255fc1c936fb424b736c5, tbl_0729ea24875c1d63f2087d26eda0c9d746caf1e821de548cfbd6da62a7e2c51a, tbl_64d7e7f0f1ac3d3e1adcf9d4f6092ea79edc56bb69eb0aaab03a0fc254196384, tbl_e60c30ceb769bf2592b82255b159dc1fffa5cfba24504a93be725eb0ac529c6f
- Gold table IDs: tbl_1fea09c00553183173733966a79d1ad2795094a415e277a0034fc4dd8aa6b616, tbl_a6c2f4b94692ec0008a9f9b550f9d3c27eceea641af58b740c93ac04817e7e44
- Missing gold table IDs: (none)
- Eligible documents: 500
- Empty reason: (none)
- Filter counts: company_codes=2507/2507; periods=33721/500
- Scores and matched tokens: tbl_1fea09c00553183173733966a79d1ad2795094a415e277a0034fc4dd8aa6b616=10.832029 [2023, acb, assets, hợp, nhất, sản, total, tài, tổng]; tbl_a6c2f4b94692ec0008a9f9b550f9d3c27eceea641af58b740c93ac04817e7e44=10.826311 [2022, acb, assets, hợp, nhất, sản, total, tài, tổng]; tbl_c4d774fe3b6b41739275aee795d588229ebb9e3c42c3b8d8603e2298d39d0acb=9.976265 [2023, acb, assets, năm, sản, total, tài, tổng]; tbl_857e287a58e6dccf9d4978c2f0e458db1a4cdbe6ed08657cccd241bf104ffb78=9.555859 [2023, acb, assets, sản, total, tài, tổng]; tbl_d5d5b86f6ef69e35a78bb9db82b84dd28fa373134fda1aa81138aefa966fc3ea=9.305341 [2022, acb, assets, sản, total, tài, tổng]; tbl_0fbe34be5f3b1a78b7892752173842d14709c9f542060e336dce302b455a3423=9.248669 [2023, acb, assets, sản, total, tài, tổng]; tbl_002b5e0f4b961fa5dec48576b2ffd1c352cdc75f5f7255fc1c936fb424b736c5=9.242352 [2022, acb, assets, sản, total, tài, tổng]; tbl_0729ea24875c1d63f2087d26eda0c9d746caf1e821de548cfbd6da62a7e2c51a=9.242352 [2022, acb, assets, sản, total, tài, tổng]; tbl_64d7e7f0f1ac3d3e1adcf9d4f6092ea79edc56bb69eb0aaab03a0fc254196384=9.242352 [2022, acb, assets, sản, total, tài, tổng]; tbl_e60c30ceb769bf2592b82255b159dc1fffa5cfba24504a93be725eb0ac529c6f=9.242352 [2022, acb, assets, sản, total, tài, tổng]

### retq_6a6a6024dd19f425b10ad6fb6e58f1bf8842e1886363b812d379f54a35e44655

- Question: Tính tốc độ tăng trưởng tổng tài sản của DBC từ năm 2022 đến năm 2023.
- Intent: growth
- Failure: none
- Predicted table IDs: tbl_749c1cdabaaa5187ade0e89453c5fa7cd1d1613b718d69e815542be629584e7c, tbl_74e16f98694ee28e3ff766295a7e9e5abfd634330b4beae00f227f0adaf71471, tbl_b5d669baf182515beda7a898924015f9e9c1cca964f9fad732605626b7582a53, tbl_c7a3dc290e8710501c256e8b445c2c75812eaf920533f990063a97d5eecbca2a, tbl_200253b5c1efa6da56e621fba128d836e0013e650999d16be5a70fea681908fc, tbl_1bfc5d4ab2d27d38450371e1315b8ff833ae79ce4b8e4ee69bc1ecc10942ef36, tbl_88e63205de2b51d7e5588d6cebea6c40310588683ade51a8cd368a74e6022331, tbl_047406408377a99a416e8363bae707fc49347d465958e3f8f9556ba82075547b, tbl_4c23cce7abca32ac741dc70045e3d19fe5f499d9f03c0c01cce36b90d61ebced, tbl_735f2d4c7268b319866584df812b54a727d23ee544120801e81b8fe0a4fb5203
- Gold table IDs: tbl_047406408377a99a416e8363bae707fc49347d465958e3f8f9556ba82075547b, tbl_88e63205de2b51d7e5588d6cebea6c40310588683ade51a8cd368a74e6022331
- Missing gold table IDs: (none)
- Eligible documents: 268
- Empty reason: (none)
- Filter counts: company_codes=1524/1524; periods=33721/268
- Scores and matched tokens: tbl_749c1cdabaaa5187ade0e89453c5fa7cd1d1613b718d69e815542be629584e7c=9.744782 [2022, assets, dbc, năm, sản, total, tài, tổng]; tbl_74e16f98694ee28e3ff766295a7e9e5abfd634330b4beae00f227f0adaf71471=9.606115 [2022, 2023, assets, dbc, năm, sản, total, tài, tổng]; tbl_b5d669baf182515beda7a898924015f9e9c1cca964f9fad732605626b7582a53=9.249885 [2022, assets, dbc, năm, sản, total, tài, tổng]; tbl_c7a3dc290e8710501c256e8b445c2c75812eaf920533f990063a97d5eecbca2a=9.175082 [2023, assets, dbc, năm, sản, total, tài, tổng]; tbl_200253b5c1efa6da56e621fba128d836e0013e650999d16be5a70fea681908fc=9.172060 [2022, 2023, assets, dbc, năm, sản, total, tài, tổng]; tbl_1bfc5d4ab2d27d38450371e1315b8ff833ae79ce4b8e4ee69bc1ecc10942ef36=9.045750 [2023, assets, dbc, sản, total, tài, tính, tổng]; tbl_88e63205de2b51d7e5588d6cebea6c40310588683ade51a8cd368a74e6022331=9.045750 [2023, assets, dbc, sản, total, tài, tính, tổng]; tbl_047406408377a99a416e8363bae707fc49347d465958e3f8f9556ba82075547b=9.038874 [2022, assets, dbc, sản, total, tài, tính, tổng]; tbl_4c23cce7abca32ac741dc70045e3d19fe5f499d9f03c0c01cce36b90d61ebced=9.038874 [2022, assets, dbc, sản, total, tài, tính, tổng]; tbl_735f2d4c7268b319866584df812b54a727d23ee544120801e81b8fe0a4fb5203=8.855544 [2023, assets, dbc, năm, sản, total, tài, tổng]

### retq_76c79261bdcb719e588c362ee794146de50e423a79e93c14949057684cc02dcf

- Question: Tính tốc độ tăng trưởng tổng tài sản hợp nhất của HDB từ năm 2022 đến năm 2023.
- Intent: growth
- Failure: zero_gold_hits
- Predicted table IDs: tbl_372401f3c194d40741dc2672604546ff4bb354f00dc3c0b9b0b8a2e697ed2510, tbl_1023e8dce509a03c9b05d6827a8c2178a530c665efaf58947721e0f8e7695ef6, tbl_904ff7161b3b85eaf7986d779bb6b3e93e7012a2a1c1bbe5b804478943585f2d, tbl_c4f2b5873b56347e476b7e7cdac4202ef415d872371bb054101976f080c1ecb3, tbl_70eff70e8607a22d5aefa3a3362de6aebc934b526d8096ea45ecdad5ab2d3e91, tbl_929b58991fe4338c757276556a424ddb430c2d44aca0703c3f4acca21f2fb2d6, tbl_ad7f3da068de1feba8aa6aeaa19e1ecec7bbf9c163a6ab47bb5b6aab9367530e, tbl_5dd01704e92cdbd559d6723cbce0c7a42177f939d8cd5b1f72fa414345980269, tbl_0b806ba5d2a01cecd33be5927a877bab3711ee1c2111d6796a4a7ab430338533, tbl_2f32b7520214715111a367b968521ceccfa86e45b0f20eb0aea152afdb795e6c
- Gold table IDs: tbl_c1bf93f626705cff88c7f64eb14839c9d3993ce49cf334053eb59f9c5a8de9a0
- Missing gold table IDs: tbl_c1bf93f626705cff88c7f64eb14839c9d3993ce49cf334053eb59f9c5a8de9a0
- Eligible documents: 672
- Empty reason: (none)
- Filter counts: company_codes=1835/1835; periods=33721/672
- Scores and matched tokens: tbl_372401f3c194d40741dc2672604546ff4bb354f00dc3c0b9b0b8a2e697ed2510=10.889311 [2023, assets, hdb, hợp, nhất, sản, total, tài, tổng]; tbl_1023e8dce509a03c9b05d6827a8c2178a530c665efaf58947721e0f8e7695ef6=10.883622 [2022, assets, hdb, hợp, nhất, sản, total, tài, tổng]; tbl_904ff7161b3b85eaf7986d779bb6b3e93e7012a2a1c1bbe5b804478943585f2d=10.726690 [2023, assets, hdb, sản, total, tài, tổng, từ, đến]; tbl_c4f2b5873b56347e476b7e7cdac4202ef415d872371bb054101976f080c1ecb3=9.612303 [2023, assets, của, hdb, năm, sản, total, tài, tổng]; tbl_70eff70e8607a22d5aefa3a3362de6aebc934b526d8096ea45ecdad5ab2d3e91=9.319831 [2023, assets, hdb, sản, total, tài, tổng]; tbl_929b58991fe4338c757276556a424ddb430c2d44aca0703c3f4acca21f2fb2d6=9.319831 [2023, assets, hdb, sản, total, tài, tổng]; tbl_ad7f3da068de1feba8aa6aeaa19e1ecec7bbf9c163a6ab47bb5b6aab9367530e=9.319831 [2023, assets, hdb, sản, total, tài, tổng]; tbl_5dd01704e92cdbd559d6723cbce0c7a42177f939d8cd5b1f72fa414345980269=9.313551 [2022, assets, hdb, sản, total, tài, tổng]; tbl_0b806ba5d2a01cecd33be5927a877bab3711ee1c2111d6796a4a7ab430338533=9.234558 [2023, assets, hdb, sản, total, tài, tổng]; tbl_2f32b7520214715111a367b968521ceccfa86e45b0f20eb0aea152afdb795e6c=9.234558 [2023, assets, hdb, sản, total, tài, tổng]

### retq_799bb5213c176968fa13d78639d80e3e479c3ce0e6d1531912c7fc9dd0ffdf84

- Question: Tính tốc độ tăng trưởng lưu chuyển tiền thuần từ hoạt động kinh doanh của CTG từ năm 2022 đến năm 2023.
- Intent: growth
- Failure: none
- Predicted table IDs: tbl_5a6128f7f8df589d7546dd7e7364079ec4758a5747fe349ec88f9aa70020ebfd, tbl_e195adb9dfd5e9c080b12cd84e11aeb0526d9150681a22d2d119e55ebd8fc766, tbl_eb52eb65997633f7856a3a6a5477a8e2258b6d311c6fad0a2b32aef7318c82fc, tbl_4bd8bdfdba928ef4a1171833de1f7945887d13472b5dea375220b74462297d26, tbl_a43bcc4e35e1f88425f4f6a930a25a3970fa17f09065d67ed5da52c5aa7faa16, tbl_22c0746dc41f87e2f3e9dcdb3de72433951f418366a767bb335ff37c84ff98c1, tbl_72528b9e67999b76aff958479dd3f93230f505eefb530f70be0cec919b33fe6d, tbl_58a8f71151b6eef68c4e4b0f32549ed3fd91fe2fcbe753135fe1a71a62d7e343, tbl_db9daba556b7f680ec8142f7b3f03abc44ed98cb30bda0f91d5c9eea62490853, tbl_0b5dd2aabc25acf5f8820e071304554862c448762bd2e80c0d4efa8a0258efa0
- Gold table IDs: tbl_22c0746dc41f87e2f3e9dcdb3de72433951f418366a767bb335ff37c84ff98c1, tbl_5a6128f7f8df589d7546dd7e7364079ec4758a5747fe349ec88f9aa70020ebfd
- Missing gold table IDs: (none)
- Eligible documents: 409
- Empty reason: (none)
- Filter counts: company_codes=1964/1964; periods=33721/409
- Scores and matched tokens: tbl_5a6128f7f8df589d7546dd7e7364079ec4758a5747fe349ec88f9aa70020ebfd=16.966583 [2022, 2023, cash, chuyển, ctg, doanh, flow, hoạt, kinh, lưu, operating, thuần, tiền, từ, động]; tbl_e195adb9dfd5e9c080b12cd84e11aeb0526d9150681a22d2d119e55ebd8fc766=16.966583 [2022, 2023, cash, chuyển, ctg, doanh, flow, hoạt, kinh, lưu, operating, thuần, tiền, từ, động]; tbl_eb52eb65997633f7856a3a6a5477a8e2258b6d311c6fad0a2b32aef7318c82fc=16.900412 [2022, 2023, cash, chuyển, ctg, flow, hoạt, lưu, thuần, tiền, từ, động]; tbl_4bd8bdfdba928ef4a1171833de1f7945887d13472b5dea375220b74462297d26=16.696131 [2023, cash, chuyển, ctg, doanh, flow, hoạt, kinh, lưu, operating, thuần, tiền, từ, động]; tbl_a43bcc4e35e1f88425f4f6a930a25a3970fa17f09065d67ed5da52c5aa7faa16=16.696131 [2023, cash, chuyển, ctg, doanh, flow, hoạt, kinh, lưu, operating, thuần, tiền, từ, động]; tbl_22c0746dc41f87e2f3e9dcdb3de72433951f418366a767bb335ff37c84ff98c1=16.689587 [2022, cash, chuyển, ctg, doanh, flow, hoạt, kinh, lưu, operating, thuần, tiền, từ, động]; tbl_72528b9e67999b76aff958479dd3f93230f505eefb530f70be0cec919b33fe6d=16.689587 [2022, cash, chuyển, ctg, doanh, flow, hoạt, kinh, lưu, operating, thuần, tiền, từ, động]; tbl_58a8f71151b6eef68c4e4b0f32549ed3fd91fe2fcbe753135fe1a71a62d7e343=16.191204 [2023, cash, chuyển, ctg, flow, hoạt, lưu, thuần, tiền, từ, động]; tbl_db9daba556b7f680ec8142f7b3f03abc44ed98cb30bda0f91d5c9eea62490853=10.189269 [2022, 2023, ctg, doanh, hoạt, kinh, thuần, từ, động]; tbl_0b5dd2aabc25acf5f8820e071304554862c448762bd2e80c0d4efa8a0258efa0=9.364302 [2023, ctg, doanh, hoạt, kinh, thuần, từ, động]

### retq_7afd1c8e800a18c317c7f2b540f77433d318a5b3aa2c483fc333294615416da2

- Question: Tính tốc độ tăng trưởng lưu chuyển tiền thuần từ hoạt động kinh doanh của GEE từ năm 2022 đến năm 2023.
- Intent: growth
- Failure: none
- Predicted table IDs: tbl_3daf2d5a6f938548c4bfd09d449ec92fa96da2f3a61b61a049c906f5006afe97, tbl_de031e48c006ab9ab8a2ef63631c8431a15c0e4f911bcfc50b70eae9d7bed69e, tbl_2c5defc9f235a2cb5ca9c1ac242ad2d9494fda30a38428c64c86be259d58586e, tbl_4502ee46a12ed5d542ba6c763a886eb813e033b81bc4f4776c127dbeed17be35, tbl_7a45bad8e1f8cf633ddb7ad222f3406f2aa9814b3058f3740136c521a6fd0c31, tbl_401087a11ca7f72fa9b808e9ce7a94933db3891933875fc1932611eb7938d9a9, tbl_da1d47a8a53bfd926aff4a1e6b269387bb3032d369efd8ea27fb5a46eed1214f, tbl_8e14f884401a116c3fdb6f7089e40ea7c9dc909e7b4a99b33dd8b2c380fe282a, tbl_40959cf176ed6f953824f9fb0ae6bb8f52fdffb9576332506d325471e80f2968, tbl_70e9fae7dd5bab38f9c6af9926df6aea102db7162ae8b097dcc9ae1727f35f47
- Gold table IDs: tbl_3daf2d5a6f938548c4bfd09d449ec92fa96da2f3a61b61a049c906f5006afe97, tbl_4502ee46a12ed5d542ba6c763a886eb813e033b81bc4f4776c127dbeed17be35
- Missing gold table IDs: (none)
- Eligible documents: 319
- Empty reason: (none)
- Filter counts: company_codes=936/936; periods=33721/319
- Scores and matched tokens: tbl_3daf2d5a6f938548c4bfd09d449ec92fa96da2f3a61b61a049c906f5006afe97=19.372528 [2022, 2023, cash, chuyển, doanh, flow, gee, hoạt, kinh, lưu, operating, thuần, tiền, từ, động]; tbl_de031e48c006ab9ab8a2ef63631c8431a15c0e4f911bcfc50b70eae9d7bed69e=19.248081 [2022, 2023, cash, chuyển, doanh, flow, gee, hoạt, kinh, lưu, operating, thuần, tiền, từ, động]; tbl_2c5defc9f235a2cb5ca9c1ac242ad2d9494fda30a38428c64c86be259d58586e=18.741270 [2023, cash, chuyển, doanh, flow, gee, hoạt, kinh, lưu, operating, thuần, tiền, từ, động]; tbl_4502ee46a12ed5d542ba6c763a886eb813e033b81bc4f4776c127dbeed17be35=18.735836 [2022, cash, chuyển, doanh, flow, gee, hoạt, kinh, lưu, operating, thuần, tiền, từ, động]; tbl_7a45bad8e1f8cf633ddb7ad222f3406f2aa9814b3058f3740136c521a6fd0c31=18.601702 [2022, cash, chuyển, doanh, flow, gee, hoạt, kinh, lưu, operating, thuần, tiền, từ, động]; tbl_401087a11ca7f72fa9b808e9ce7a94933db3891933875fc1932611eb7938d9a9=14.192246 [2022, 2023, cash, chuyển, flow, gee, hoạt, lưu, thuần, tiền, từ, động]; tbl_da1d47a8a53bfd926aff4a1e6b269387bb3032d369efd8ea27fb5a46eed1214f=14.192246 [2022, 2023, cash, chuyển, flow, gee, hoạt, lưu, thuần, tiền, từ, động]; tbl_8e14f884401a116c3fdb6f7089e40ea7c9dc909e7b4a99b33dd8b2c380fe282a=13.422557 [2023, cash, chuyển, flow, gee, hoạt, lưu, thuần, tiền, từ, động]; tbl_40959cf176ed6f953824f9fb0ae6bb8f52fdffb9576332506d325471e80f2968=13.415933 [2022, cash, chuyển, flow, gee, hoạt, lưu, thuần, tiền, từ, động]; tbl_70e9fae7dd5bab38f9c6af9926df6aea102db7162ae8b097dcc9ae1727f35f47=13.415933 [2022, cash, chuyển, flow, gee, hoạt, lưu, thuần, tiền, từ, động]

### retq_8e43d1b9fd61ad81fb038d853c71c1c4b582ab7c502fae41ed33edf22737bc50

- Question: Tra cứu tổng tài sản của DBC năm 2023.
- Intent: lookup
- Failure: none
- Predicted table IDs: tbl_c7a3dc290e8710501c256e8b445c2c75812eaf920533f990063a97d5eecbca2a, tbl_74e16f98694ee28e3ff766295a7e9e5abfd634330b4beae00f227f0adaf71471, tbl_735f2d4c7268b319866584df812b54a727d23ee544120801e81b8fe0a4fb5203, tbl_200253b5c1efa6da56e621fba128d836e0013e650999d16be5a70fea681908fc, tbl_1bfc5d4ab2d27d38450371e1315b8ff833ae79ce4b8e4ee69bc1ecc10942ef36, tbl_88e63205de2b51d7e5588d6cebea6c40310588683ade51a8cd368a74e6022331, tbl_96e3cc366bed469062138c19446a2f82a2b47a3e1723ec9a9671c9dc4e62dc08, tbl_e23ca68cb3ce1b4f3555569206cbd979148b09947a8f26838b67fd49268e81f9, tbl_e3ee982ba272d5d18bc7ee46e296a83c93a67868a40538dfa0ec52578a76877a, tbl_fa5b884b74a3084008578985108ac4fff216795c9ac07b4de17e970d82b5e3a3
- Gold table IDs: tbl_88e63205de2b51d7e5588d6cebea6c40310588683ade51a8cd368a74e6022331
- Missing gold table IDs: (none)
- Eligible documents: 137
- Empty reason: (none)
- Filter counts: company_codes=1524/1524; periods=18228/137
- Scores and matched tokens: tbl_c7a3dc290e8710501c256e8b445c2c75812eaf920533f990063a97d5eecbca2a=9.175082 [2023, assets, dbc, năm, sản, total, tài, tổng]; tbl_74e16f98694ee28e3ff766295a7e9e5abfd634330b4beae00f227f0adaf71471=8.920013 [2023, assets, dbc, năm, sản, total, tài, tổng]; tbl_735f2d4c7268b319866584df812b54a727d23ee544120801e81b8fe0a4fb5203=8.855544 [2023, assets, dbc, năm, sản, total, tài, tổng]; tbl_200253b5c1efa6da56e621fba128d836e0013e650999d16be5a70fea681908fc=8.521295 [2023, assets, dbc, năm, sản, total, tài, tổng]; tbl_1bfc5d4ab2d27d38450371e1315b8ff833ae79ce4b8e4ee69bc1ecc10942ef36=7.847108 [2023, assets, dbc, sản, total, tài, tổng]; tbl_88e63205de2b51d7e5588d6cebea6c40310588683ade51a8cd368a74e6022331=7.847108 [2023, assets, dbc, sản, total, tài, tổng]; tbl_96e3cc366bed469062138c19446a2f82a2b47a3e1723ec9a9671c9dc4e62dc08=5.115978 [2023, của, dbc, sản, tài]; tbl_e23ca68cb3ce1b4f3555569206cbd979148b09947a8f26838b67fd49268e81f9=5.115978 [2023, của, dbc, sản, tài]; tbl_e3ee982ba272d5d18bc7ee46e296a83c93a67868a40538dfa0ec52578a76877a=5.058585 [2023, dbc, total, tổng]; tbl_fa5b884b74a3084008578985108ac4fff216795c9ac07b4de17e970d82b5e3a3=5.058585 [2023, dbc, total, tổng]

### retq_905e8568becf90520e588c3292c269b0a7030ba5ec4eff4f4cbcf8d21a0fba23

- Question: Tra cứu tổng tài sản hợp nhất của HDB năm 2023.
- Intent: lookup
- Failure: zero_gold_hits
- Predicted table IDs: tbl_372401f3c194d40741dc2672604546ff4bb354f00dc3c0b9b0b8a2e697ed2510, tbl_c4f2b5873b56347e476b7e7cdac4202ef415d872371bb054101976f080c1ecb3, tbl_70eff70e8607a22d5aefa3a3362de6aebc934b526d8096ea45ecdad5ab2d3e91, tbl_929b58991fe4338c757276556a424ddb430c2d44aca0703c3f4acca21f2fb2d6, tbl_ad7f3da068de1feba8aa6aeaa19e1ecec7bbf9c163a6ab47bb5b6aab9367530e, tbl_0b806ba5d2a01cecd33be5927a877bab3711ee1c2111d6796a4a7ab430338533, tbl_2f32b7520214715111a367b968521ceccfa86e45b0f20eb0aea152afdb795e6c, tbl_99cab0babba599e3c5b47709778ef4e01f13de8e0dce95717c2181b84984f430, tbl_d272df1c5a9744f702469a8b26c15ccefee1300877b438fe8e454bdb0b0b706f, tbl_0dd98cc4d3f022ccce3c302b0324c1b60e54404c21d0d8c98794b30356d70f0e
- Gold table IDs: tbl_c1bf93f626705cff88c7f64eb14839c9d3993ce49cf334053eb59f9c5a8de9a0
- Missing gold table IDs: tbl_c1bf93f626705cff88c7f64eb14839c9d3993ce49cf334053eb59f9c5a8de9a0
- Eligible documents: 346
- Empty reason: (none)
- Filter counts: company_codes=1835/1835; periods=18228/346
- Scores and matched tokens: tbl_372401f3c194d40741dc2672604546ff4bb354f00dc3c0b9b0b8a2e697ed2510=10.889311 [2023, assets, hdb, hợp, nhất, sản, total, tài, tổng]; tbl_c4f2b5873b56347e476b7e7cdac4202ef415d872371bb054101976f080c1ecb3=9.612303 [2023, assets, của, hdb, năm, sản, total, tài, tổng]; tbl_70eff70e8607a22d5aefa3a3362de6aebc934b526d8096ea45ecdad5ab2d3e91=9.319831 [2023, assets, hdb, sản, total, tài, tổng]; tbl_929b58991fe4338c757276556a424ddb430c2d44aca0703c3f4acca21f2fb2d6=9.319831 [2023, assets, hdb, sản, total, tài, tổng]; tbl_ad7f3da068de1feba8aa6aeaa19e1ecec7bbf9c163a6ab47bb5b6aab9367530e=9.319831 [2023, assets, hdb, sản, total, tài, tổng]; tbl_0b806ba5d2a01cecd33be5927a877bab3711ee1c2111d6796a4a7ab430338533=9.234558 [2023, assets, hdb, sản, total, tài, tổng]; tbl_2f32b7520214715111a367b968521ceccfa86e45b0f20eb0aea152afdb795e6c=9.234558 [2023, assets, hdb, sản, total, tài, tổng]; tbl_99cab0babba599e3c5b47709778ef4e01f13de8e0dce95717c2181b84984f430=9.106573 [2023, assets, hdb, sản, total, tài, tổng]; tbl_d272df1c5a9744f702469a8b26c15ccefee1300877b438fe8e454bdb0b0b706f=9.106573 [2023, assets, hdb, sản, total, tài, tổng]; tbl_0dd98cc4d3f022ccce3c302b0324c1b60e54404c21d0d8c98794b30356d70f0e=9.069023 [2023, assets, hdb, sản, total, tài, tổng]

### retq_9490ea8fbfcc0834be9ca1d779411b8cf60ceeaf4816cbd05abc9c4ceeb48118

- Question: Tính tốc độ tăng trưởng lưu chuyển tiền thuần từ hoạt động kinh doanh của CEO từ năm 2022 đến năm 2023.
- Intent: growth
- Failure: none
- Predicted table IDs: tbl_3739f942a9c1843ffe9cd7a4d9edc68218501714148f91c5c89672f0e0e86189, tbl_6e4b99fbc765e5d4cfd6939f8b6344d1a95ddc2f33f07381dec1d179298a4331, tbl_563660a79f23c998cae04f5ba32553bae87422e6022f5773e07e8f5c145644a0, tbl_3bc0d37f4b340921d379dab7d121cebb8475001228ec98d942c418d288f53b04, tbl_04b890c73a5fd34f02ee1f3ac9d956763bcd851bbefe551cdf5830f37abb083a, tbl_0cc001ed54985f5912f4f6adf42ef0e69ed968cec28526b6f1556d35aece2320, tbl_e0bcfca93e2626e55291565db4b41e5329eacbfb3d1b6cfefb47ed83483a9633, tbl_3deb0dd025bded446da14d5898776ba94a63d53b8a40cc08275a520ea5598744, tbl_38ef7213e6380791e719d2738eb9584442c1dc5c1c6a5eeb659f433b71b07c7a, tbl_da6a4d3859a5904adfeacb8d714d04686a51f4cf98da4915aa6948d095508b88
- Gold table IDs: tbl_3739f942a9c1843ffe9cd7a4d9edc68218501714148f91c5c89672f0e0e86189, tbl_6e4b99fbc765e5d4cfd6939f8b6344d1a95ddc2f33f07381dec1d179298a4331
- Missing gold table IDs: (none)
- Eligible documents: 278
- Empty reason: (none)
- Filter counts: company_codes=1332/1332; periods=33721/278
- Scores and matched tokens: tbl_3739f942a9c1843ffe9cd7a4d9edc68218501714148f91c5c89672f0e0e86189=21.536692 [2023, cash, ceo, chuyển, doanh, flow, hoạt, kinh, lưu, năm, operating, thuần, tiền, từ, động]; tbl_6e4b99fbc765e5d4cfd6939f8b6344d1a95ddc2f33f07381dec1d179298a4331=21.005714 [2022, cash, ceo, chuyển, doanh, flow, hoạt, kinh, lưu, operating, thuần, tiền, từ, động]; tbl_563660a79f23c998cae04f5ba32553bae87422e6022f5773e07e8f5c145644a0=20.389980 [2023, cash, ceo, chuyển, doanh, flow, hoạt, kinh, lưu, năm, operating, thuần, tiền, từ, động]; tbl_3bc0d37f4b340921d379dab7d121cebb8475001228ec98d942c418d288f53b04=20.381113 [2022, cash, ceo, chuyển, doanh, flow, hoạt, kinh, lưu, năm, operating, thuần, tiền, từ, động]; tbl_04b890c73a5fd34f02ee1f3ac9d956763bcd851bbefe551cdf5830f37abb083a=13.649815 [2023, cash, ceo, chuyển, flow, hoạt, lưu, thuần, tiền, từ, động]; tbl_0cc001ed54985f5912f4f6adf42ef0e69ed968cec28526b6f1556d35aece2320=13.643461 [2022, cash, ceo, chuyển, flow, hoạt, lưu, thuần, tiền, từ, động]; tbl_e0bcfca93e2626e55291565db4b41e5329eacbfb3d1b6cfefb47ed83483a9633=10.412238 [2022, cash, ceo, chuyển, flow, lưu, năm, tiền, đến]; tbl_3deb0dd025bded446da14d5898776ba94a63d53b8a40cc08275a520ea5598744=8.923436 [2022, 2023, cash, ceo, chuyển, flow, lưu, tiền]; tbl_38ef7213e6380791e719d2738eb9584442c1dc5c1c6a5eeb659f433b71b07c7a=8.869734 [2022, 2023, cash, ceo, chuyển, flow, lưu, tiền]; tbl_da6a4d3859a5904adfeacb8d714d04686a51f4cf98da4915aa6948d095508b88=8.785340 [2023, cash, ceo, chuyển, flow, lưu, năm, tiền]

### retq_9abb0ade76092d5443f249b43fb913340f16cfc5318c8631869c939e065534bf

- Question: Tra cứu doanh thu thuần của VGT năm 2023.
- Intent: lookup
- Failure: none
- Predicted table IDs: tbl_973227f5e2dd1abdc06bfad4b15f0c4875c9deb51df912479158a3181a166a96, tbl_51da1cbb672a452be154cec9a681bc01efe01061ba8d049d299945a4d62f1b3b, tbl_d6f4a8333105e0a773679a13c1d4c69d57622e605e7005a692abbbdea2c9eaea, tbl_8d21a88d300d8b0951642ea6401c6909d8f1648a677c586b29bc9eb6e18bcf0c, tbl_a977f686cbadec43c2953082844b2fedc0ba3096192ff723072ce71759443fea, tbl_1725ebbd226753a0b65386525ae024d153081927463e67eb6d2fb5da9f966ff8, tbl_2193d2ed58feff1b819652ad19599d479b2d90a7f2118cdb8cbdd653aabf1ee4, tbl_b35bdc026fae20558af3dc57027667b106f254b5a268c0a4185c305e8b10dea6, tbl_1081ce5deab08a1793c1708679f529894a1f72acf3ea33c2e778ea6663b27c84, tbl_725902b206361faa3a131f440f7c4a37ec9694aca5929aee97577bf0c5adf60d
- Gold table IDs: tbl_973227f5e2dd1abdc06bfad4b15f0c4875c9deb51df912479158a3181a166a96
- Missing gold table IDs: (none)
- Eligible documents: 195
- Empty reason: (none)
- Filter counts: company_codes=1700/1700; periods=18228/195
- Scores and matched tokens: tbl_973227f5e2dd1abdc06bfad4b15f0c4875c9deb51df912479158a3181a166a96=10.345021 [2023, doanh, net, revenue, thu, thuần, vgt]; tbl_51da1cbb672a452be154cec9a681bc01efe01061ba8d049d299945a4d62f1b3b=10.287153 [2023, doanh, net, revenue, thu, thuần, vgt]; tbl_d6f4a8333105e0a773679a13c1d4c69d57622e605e7005a692abbbdea2c9eaea=6.359388 [2023, của, doanh, revenue, thu, vgt]; tbl_8d21a88d300d8b0951642ea6401c6909d8f1648a677c586b29bc9eb6e18bcf0c=5.904732 [2023, doanh, thu, thuần, vgt]; tbl_a977f686cbadec43c2953082844b2fedc0ba3096192ff723072ce71759443fea=5.904732 [2023, doanh, thu, thuần, vgt]; tbl_1725ebbd226753a0b65386525ae024d153081927463e67eb6d2fb5da9f966ff8=5.775032 [2023, của, doanh, revenue, thu, vgt]; tbl_2193d2ed58feff1b819652ad19599d479b2d90a7f2118cdb8cbdd653aabf1ee4=5.752707 [2023, của, doanh, revenue, thu, vgt]; tbl_b35bdc026fae20558af3dc57027667b106f254b5a268c0a4185c305e8b10dea6=5.685530 [2023, của, doanh, revenue, thu, vgt]; tbl_1081ce5deab08a1793c1708679f529894a1f72acf3ea33c2e778ea6663b27c84=5.666284 [2023, của, doanh, revenue, thu, vgt]; tbl_725902b206361faa3a131f440f7c4a37ec9694aca5929aee97577bf0c5adf60d=5.522708 [2023, của, doanh, thu, thuần, vgt]

### retq_a2d5888138ba3e4af86938cf0854c85da519cf1d4c63f8e610fc722853f816a2

- Question: Tra cứu lưu chuyển tiền thuần từ hoạt động kinh doanh của CEO năm 2023.
- Intent: lookup
- Failure: none
- Predicted table IDs: tbl_3739f942a9c1843ffe9cd7a4d9edc68218501714148f91c5c89672f0e0e86189, tbl_563660a79f23c998cae04f5ba32553bae87422e6022f5773e07e8f5c145644a0, tbl_04b890c73a5fd34f02ee1f3ac9d956763bcd851bbefe551cdf5830f37abb083a, tbl_da6a4d3859a5904adfeacb8d714d04686a51f4cf98da4915aa6948d095508b88, tbl_3deb0dd025bded446da14d5898776ba94a63d53b8a40cc08275a520ea5598744, tbl_38ef7213e6380791e719d2738eb9584442c1dc5c1c6a5eeb659f433b71b07c7a, tbl_686f341f70b8a52ee0e5e1d0d2aea95b26d711e83c6811124d836340969d4e4a, tbl_d79402f5f6a3fad4c816b65d41d0920021768169b5e062ab39f5472468a846e8, tbl_230152820c3f3d9720e42f619236e87094800fab026987609544761746d31e74, tbl_0d6c74997de626b69e647ac369c48af0e4f5ea7394df186c0902181d49cb0fe2
- Gold table IDs: tbl_3739f942a9c1843ffe9cd7a4d9edc68218501714148f91c5c89672f0e0e86189
- Missing gold table IDs: (none)
- Eligible documents: 154
- Empty reason: (none)
- Filter counts: company_codes=1332/1332; periods=18228/154
- Scores and matched tokens: tbl_3739f942a9c1843ffe9cd7a4d9edc68218501714148f91c5c89672f0e0e86189=21.536692 [2023, cash, ceo, chuyển, doanh, flow, hoạt, kinh, lưu, năm, operating, thuần, tiền, từ, động]; tbl_563660a79f23c998cae04f5ba32553bae87422e6022f5773e07e8f5c145644a0=20.389980 [2023, cash, ceo, chuyển, doanh, flow, hoạt, kinh, lưu, năm, operating, thuần, tiền, từ, động]; tbl_04b890c73a5fd34f02ee1f3ac9d956763bcd851bbefe551cdf5830f37abb083a=13.649815 [2023, cash, ceo, chuyển, flow, hoạt, lưu, thuần, tiền, từ, động]; tbl_da6a4d3859a5904adfeacb8d714d04686a51f4cf98da4915aa6948d095508b88=8.785340 [2023, cash, ceo, chuyển, flow, lưu, năm, tiền]; tbl_3deb0dd025bded446da14d5898776ba94a63d53b8a40cc08275a520ea5598744=8.158407 [2023, cash, ceo, chuyển, flow, lưu, tiền]; tbl_38ef7213e6380791e719d2738eb9584442c1dc5c1c6a5eeb659f433b71b07c7a=8.109308 [2023, cash, ceo, chuyển, flow, lưu, tiền]; tbl_686f341f70b8a52ee0e5e1d0d2aea95b26d711e83c6811124d836340969d4e4a=8.109308 [2023, cash, ceo, chuyển, flow, lưu, tiền]; tbl_d79402f5f6a3fad4c816b65d41d0920021768169b5e062ab39f5472468a846e8=7.545342 [2023, ceo, doanh, hoạt, kinh, động]; tbl_230152820c3f3d9720e42f619236e87094800fab026987609544761746d31e74=6.886677 [2023, ceo, doanh, hoạt, kinh, động]; tbl_0d6c74997de626b69e647ac369c48af0e4f5ea7394df186c0902181d49cb0fe2=6.066193 [2023, ceo, doanh, hoạt, kinh, động]

### retq_b241522bcdab166c95e5e24ad9fb63a9eede7baf7c8432392db31274c063c389

- Question: So sánh tổng tài sản của DBC giữa năm 2022 và năm 2023.
- Intent: compare
- Failure: partial_gold_hits
- Predicted table IDs: tbl_749c1cdabaaa5187ade0e89453c5fa7cd1d1613b718d69e815542be629584e7c, tbl_74e16f98694ee28e3ff766295a7e9e5abfd634330b4beae00f227f0adaf71471, tbl_b5d669baf182515beda7a898924015f9e9c1cca964f9fad732605626b7582a53, tbl_c7a3dc290e8710501c256e8b445c2c75812eaf920533f990063a97d5eecbca2a, tbl_200253b5c1efa6da56e621fba128d836e0013e650999d16be5a70fea681908fc, tbl_735f2d4c7268b319866584df812b54a727d23ee544120801e81b8fe0a4fb5203, tbl_3049ae87567cdc4dbbefa05fb2283ca1d08b8d0a220f9f34ed186b0ad43b19ba, tbl_08423242d8a97936d16b0f934b1be775f14956b4bd19a1f42c89a2386127ca8b, tbl_1bfc5d4ab2d27d38450371e1315b8ff833ae79ce4b8e4ee69bc1ecc10942ef36, tbl_88e63205de2b51d7e5588d6cebea6c40310588683ade51a8cd368a74e6022331
- Gold table IDs: tbl_047406408377a99a416e8363bae707fc49347d465958e3f8f9556ba82075547b, tbl_88e63205de2b51d7e5588d6cebea6c40310588683ade51a8cd368a74e6022331
- Missing gold table IDs: tbl_047406408377a99a416e8363bae707fc49347d465958e3f8f9556ba82075547b
- Eligible documents: 268
- Empty reason: (none)
- Filter counts: company_codes=1524/1524; periods=33721/268
- Scores and matched tokens: tbl_749c1cdabaaa5187ade0e89453c5fa7cd1d1613b718d69e815542be629584e7c=10.295134 [2022, assets, dbc, năm, sản, total, tài, tổng, và]; tbl_74e16f98694ee28e3ff766295a7e9e5abfd634330b4beae00f227f0adaf71471=10.177385 [2022, 2023, assets, dbc, năm, sản, total, tài, tổng, và]; tbl_b5d669baf182515beda7a898924015f9e9c1cca964f9fad732605626b7582a53=9.821155 [2022, assets, dbc, năm, sản, total, tài, tổng, và]; tbl_c7a3dc290e8710501c256e8b445c2c75812eaf920533f990063a97d5eecbca2a=9.740214 [2023, assets, dbc, năm, sản, total, tài, tổng, và]; tbl_200253b5c1efa6da56e621fba128d836e0013e650999d16be5a70fea681908fc=9.713907 [2022, 2023, assets, dbc, năm, sản, total, tài, tổng, và]; tbl_735f2d4c7268b319866584df812b54a727d23ee544120801e81b8fe0a4fb5203=9.397391 [2023, assets, dbc, năm, sản, total, tài, tổng, và]; tbl_3049ae87567cdc4dbbefa05fb2283ca1d08b8d0a220f9f34ed186b0ad43b19ba=9.396541 [2022, assets, dbc, năm, sản, total, tài, tổng, và]; tbl_08423242d8a97936d16b0f934b1be775f14956b4bd19a1f42c89a2386127ca8b=9.057542 [2022, assets, dbc, năm, sản, total, tài, tổng, và]; tbl_1bfc5d4ab2d27d38450371e1315b8ff833ae79ce4b8e4ee69bc1ecc10942ef36=7.847108 [2023, assets, dbc, sản, total, tài, tổng]; tbl_88e63205de2b51d7e5588d6cebea6c40310588683ade51a8cd368a74e6022331=7.847108 [2023, assets, dbc, sản, total, tài, tổng]

### retq_b48a824c13365c7b2537ece813ecc3f1e940d2765444b9b920a75418eb0e4a61

- Question: Tra cứu doanh thu thuần về bán hàng và cung cấp dịch vụ của KHG năm 2023.
- Intent: lookup
- Failure: none
- Predicted table IDs: tbl_c13ae0c50c647da5a1b0bc166d46660b8f713fcb202e344ec3fd40e70c3adc2d, tbl_b795c1b17487a1a67e4d1e3dd26e88e0b5f025f5964628ae1b224727e30e5959, tbl_fed883a2950883c1fe641123e9eea746557b8c20e6ba04de09a86eeaab4bb4f6, tbl_0ca6d656f0d9aadb4d80eeceea331b643e04f79a15d44b53eed75d9e6b74f951, tbl_0fd3888092d9f1fcadef0dfa06dafa32cac3c3f61a6ea43f09492711784d27ec, tbl_462f7faa37fdd1416fdba611296de7a9b727af3d9c1952db41657ce9c41ce47a, tbl_4b2277a7e4fac95efaa2942fe786a994a2e6b5c8a7863da980e6800fbdebd24f, tbl_8f541080801042dbe34759258863ad6f295001a0fcb28d2f39d30bd0589d5d15, tbl_f507857adfd4701cf8a2540444690aeedce585941a851ccf96107c64eaa23ec9
- Gold table IDs: tbl_c13ae0c50c647da5a1b0bc166d46660b8f713fcb202e344ec3fd40e70c3adc2d
- Missing gold table IDs: (none)
- Eligible documents: 9
- Empty reason: (none)
- Filter counts: company_codes=806/806; periods=18228/147; statement_types=7862/9
- Scores and matched tokens: tbl_c13ae0c50c647da5a1b0bc166d46660b8f713fcb202e344ec3fd40e70c3adc2d=17.776962 [2023, bán, cung, cấp, doanh, dịch, hàng, khg, net, revenue, thu, thuần, và, về, vụ]; tbl_b795c1b17487a1a67e4d1e3dd26e88e0b5f025f5964628ae1b224727e30e5959=17.026678 [2023, bán, cung, doanh, dịch, hàng, khg, net, revenue, thu, thuần, và, về, vụ]; tbl_fed883a2950883c1fe641123e9eea746557b8c20e6ba04de09a86eeaab4bb4f6=16.224360 [2023, bán, cung, cấp, doanh, hàng, khg, net, revenue, thu, thuần, và, vụ]; tbl_0ca6d656f0d9aadb4d80eeceea331b643e04f79a15d44b53eed75d9e6b74f951=15.159566 [2023, bán, cung, doanh, dịch, hàng, khg, net, revenue, thu, thuần, và, vụ]; tbl_0fd3888092d9f1fcadef0dfa06dafa32cac3c3f61a6ea43f09492711784d27ec=3.299160 [2023, khg]; tbl_462f7faa37fdd1416fdba611296de7a9b727af3d9c1952db41657ce9c41ce47a=3.299160 [2023, khg]; tbl_4b2277a7e4fac95efaa2942fe786a994a2e6b5c8a7863da980e6800fbdebd24f=3.054828 [2023, khg]; tbl_8f541080801042dbe34759258863ad6f295001a0fcb28d2f39d30bd0589d5d15=3.054828 [2023, khg]; tbl_f507857adfd4701cf8a2540444690aeedce585941a851ccf96107c64eaa23ec9=3.054828 [2023, khg]

### retq_b74d8b5b9e878a98bf9431932d39659660d3ba35571478baa6e678ce932c4a45

- Question: Tra cứu tổng tài sản riêng của GEG năm 2023.
- Intent: lookup
- Failure: none
- Predicted table IDs: tbl_e22902ca93e0e314bb09ec37a330ada39678b4f578c553c1ac27e5ea379610ca, tbl_93cbdbbc1e0cea8d74a783a4be18d7511befae1617f35d384a67465c781815aa, tbl_d799350bf2be9ec17ad16f8e5f2c4ec533835504fd990e856c9b42885b0ad1fe
- Gold table IDs: tbl_e22902ca93e0e314bb09ec37a330ada39678b4f578c553c1ac27e5ea379610ca
- Missing gold table IDs: (none)
- Eligible documents: 3
- Empty reason: (none)
- Filter counts: company_codes=1744/1744; periods=18228/197; statement_types=7745/3
- Scores and matched tokens: tbl_e22902ca93e0e314bb09ec37a330ada39678b4f578c553c1ac27e5ea379610ca=9.742980 [2023, assets, geg, năm, riêng, sản, total, tài, tổng]; tbl_93cbdbbc1e0cea8d74a783a4be18d7511befae1617f35d384a67465c781815aa=7.744334 [2023, assets, geg, sản, total, tài, tổng]; tbl_d799350bf2be9ec17ad16f8e5f2c4ec533835504fd990e856c9b42885b0ad1fe=5.544333 [2023, assets, geg, năm, sản, tài]

### retq_bf660ff667f8d577c20962ceb66c154468642b12c827e7a52024b3fa2a5277bf

- Question: Tra cứu lưu chuyển tiền thuần từ hoạt động kinh doanh của GEE năm 2023.
- Intent: lookup
- Failure: none
- Predicted table IDs: tbl_2c5defc9f235a2cb5ca9c1ac242ad2d9494fda30a38428c64c86be259d58586e, tbl_3daf2d5a6f938548c4bfd09d449ec92fa96da2f3a61b61a049c906f5006afe97, tbl_de031e48c006ab9ab8a2ef63631c8431a15c0e4f911bcfc50b70eae9d7bed69e, tbl_401087a11ca7f72fa9b808e9ce7a94933db3891933875fc1932611eb7938d9a9, tbl_8e14f884401a116c3fdb6f7089e40ea7c9dc909e7b4a99b33dd8b2c380fe282a, tbl_da1d47a8a53bfd926aff4a1e6b269387bb3032d369efd8ea27fb5a46eed1214f, tbl_2a003ebc1e92ca3525d9cd3a820a842657530369a974a89a6f0708b5e0134f97, tbl_6b8c59c61f99af59271d8d593a38f40743511b71df93895c7551badc0cba3512, tbl_c667644ee890d79e63a10faa320c71574767bc6f32b78339bd96904bdfff555b, tbl_cee818d7d4f3bf23a8c41efe1a72d8113b9c696a1643d1a011c673d3b55dbac5
- Gold table IDs: tbl_3daf2d5a6f938548c4bfd09d449ec92fa96da2f3a61b61a049c906f5006afe97
- Missing gold table IDs: (none)
- Eligible documents: 21
- Empty reason: (none)
- Filter counts: company_codes=936/936; periods=18228/176; statement_types=13712/21
- Scores and matched tokens: tbl_2c5defc9f235a2cb5ca9c1ac242ad2d9494fda30a38428c64c86be259d58586e=18.741270 [2023, cash, chuyển, doanh, flow, gee, hoạt, kinh, lưu, operating, thuần, tiền, từ, động]; tbl_3daf2d5a6f938548c4bfd09d449ec92fa96da2f3a61b61a049c906f5006afe97=18.741270 [2023, cash, chuyển, doanh, flow, gee, hoạt, kinh, lưu, operating, thuần, tiền, từ, động]; tbl_de031e48c006ab9ab8a2ef63631c8431a15c0e4f911bcfc50b70eae9d7bed69e=18.607218 [2023, cash, chuyển, doanh, flow, gee, hoạt, kinh, lưu, operating, thuần, tiền, từ, động]; tbl_401087a11ca7f72fa9b808e9ce7a94933db3891933875fc1932611eb7938d9a9=13.422557 [2023, cash, chuyển, flow, gee, hoạt, lưu, thuần, tiền, từ, động]; tbl_8e14f884401a116c3fdb6f7089e40ea7c9dc909e7b4a99b33dd8b2c380fe282a=13.422557 [2023, cash, chuyển, flow, gee, hoạt, lưu, thuần, tiền, từ, động]; tbl_da1d47a8a53bfd926aff4a1e6b269387bb3032d369efd8ea27fb5a46eed1214f=13.422557 [2023, cash, chuyển, flow, gee, hoạt, lưu, thuần, tiền, từ, động]; tbl_2a003ebc1e92ca3525d9cd3a820a842657530369a974a89a6f0708b5e0134f97=6.834436 [2023, cash, doanh, flow, gee, kinh]; tbl_6b8c59c61f99af59271d8d593a38f40743511b71df93895c7551badc0cba3512=6.834436 [2023, cash, doanh, flow, gee, kinh]; tbl_c667644ee890d79e63a10faa320c71574767bc6f32b78339bd96904bdfff555b=6.834436 [2023, cash, doanh, flow, gee, kinh]; tbl_cee818d7d4f3bf23a8c41efe1a72d8113b9c696a1643d1a011c673d3b55dbac5=6.834436 [2023, cash, doanh, flow, gee, kinh]

### retq_c04ab744e55a62eb20943d9d43f55c5fd18f88ecbd530a61cc4a0c818af4ff17

- Question: So sánh tổng tài sản riêng của GEG giữa năm 2022 và năm 2023.
- Intent: compare
- Failure: none
- Predicted table IDs: tbl_e22902ca93e0e314bb09ec37a330ada39678b4f578c553c1ac27e5ea379610ca, tbl_0dfb3a70b900e257c2630c3ba6f24a15c8bca002dc705a1ebe08359ef5f9e0df, tbl_e5c0a901bacbe5cfab2d10ae398a388af73e490b61721b1bc62bdf474d0d0525, tbl_ffea10f77cf9a157cfa54b1f7b286857197b6b4e2f656e87d8dfe4f8414b933d, tbl_93cbdbbc1e0cea8d74a783a4be18d7511befae1617f35d384a67465c781815aa, tbl_6cdda3d6c4b9af0db9fb168f7e0b3f2e8d328aadd0f3a72b5059b6ff3143f54e, tbl_fe92fb564decf93dc0c7191c7669249b74222875724539a5a008eedfb76f1dfc, tbl_0fc30404e8267176273eefab4865f4bc28d99205b95cb7b39c842e550dafb7b4, tbl_d799350bf2be9ec17ad16f8e5f2c4ec533835504fd990e856c9b42885b0ad1fe, tbl_dd409faf210d56f2d267a5baf30358a40988c9ec7b2dc8b2b2c53ec384b9ad28
- Gold table IDs: tbl_e22902ca93e0e314bb09ec37a330ada39678b4f578c553c1ac27e5ea379610ca, tbl_e5c0a901bacbe5cfab2d10ae398a388af73e490b61721b1bc62bdf474d0d0525
- Missing gold table IDs: (none)
- Eligible documents: 355
- Empty reason: (none)
- Filter counts: company_codes=1744/1744; periods=33721/355
- Scores and matched tokens: tbl_e22902ca93e0e314bb09ec37a330ada39678b4f578c553c1ac27e5ea379610ca=10.173868 [2023, assets, geg, năm, riêng, sản, total, tài, tổng, và]; tbl_0dfb3a70b900e257c2630c3ba6f24a15c8bca002dc705a1ebe08359ef5f9e0df=9.174307 [2022, assets, geg, năm, sản, total, tài, tổng]; tbl_e5c0a901bacbe5cfab2d10ae398a388af73e490b61721b1bc62bdf474d0d0525=8.875970 [2022, assets, geg, năm, sản, total, tài, tổng]; tbl_ffea10f77cf9a157cfa54b1f7b286857197b6b4e2f656e87d8dfe4f8414b933d=8.535094 [2023, assets, geg, sản, total, tài, tổng]; tbl_93cbdbbc1e0cea8d74a783a4be18d7511befae1617f35d384a67465c781815aa=7.744334 [2023, assets, geg, sản, total, tài, tổng]; tbl_6cdda3d6c4b9af0db9fb168f7e0b3f2e8d328aadd0f3a72b5059b6ff3143f54e=7.737502 [2022, assets, geg, sản, total, tài, tổng]; tbl_fe92fb564decf93dc0c7191c7669249b74222875724539a5a008eedfb76f1dfc=7.737502 [2022, assets, geg, sản, total, tài, tổng]; tbl_0fc30404e8267176273eefab4865f4bc28d99205b95cb7b39c842e550dafb7b4=6.320115 [2023, của, geg, năm, riêng, total, tổng]; tbl_d799350bf2be9ec17ad16f8e5f2c4ec533835504fd990e856c9b42885b0ad1fe=6.091820 [2023, assets, geg, năm, sản, tài, và]; tbl_dd409faf210d56f2d267a5baf30358a40988c9ec7b2dc8b2b2c53ec384b9ad28=6.024984 [2022, assets, geg, năm, sản, tài, và]

### retq_c069554491e65164bb302c37f0d9c83d0283546dbc61e26448e2cc5512cb06e2

- Question: So sánh tổng tài sản hợp nhất của ACB giữa năm 2022 và năm 2023.
- Intent: compare
- Failure: none
- Predicted table IDs: tbl_1fea09c00553183173733966a79d1ad2795094a415e277a0034fc4dd8aa6b616, tbl_a6c2f4b94692ec0008a9f9b550f9d3c27eceea641af58b740c93ac04817e7e44, tbl_c4d774fe3b6b41739275aee795d588229ebb9e3c42c3b8d8603e2298d39d0acb, tbl_857e287a58e6dccf9d4978c2f0e458db1a4cdbe6ed08657cccd241bf104ffb78, tbl_d5d5b86f6ef69e35a78bb9db82b84dd28fa373134fda1aa81138aefa966fc3ea, tbl_0fbe34be5f3b1a78b7892752173842d14709c9f542060e336dce302b455a3423, tbl_002b5e0f4b961fa5dec48576b2ffd1c352cdc75f5f7255fc1c936fb424b736c5, tbl_0729ea24875c1d63f2087d26eda0c9d746caf1e821de548cfbd6da62a7e2c51a, tbl_64d7e7f0f1ac3d3e1adcf9d4f6092ea79edc56bb69eb0aaab03a0fc254196384, tbl_e60c30ceb769bf2592b82255b159dc1fffa5cfba24504a93be725eb0ac529c6f
- Gold table IDs: tbl_1fea09c00553183173733966a79d1ad2795094a415e277a0034fc4dd8aa6b616, tbl_a6c2f4b94692ec0008a9f9b550f9d3c27eceea641af58b740c93ac04817e7e44
- Missing gold table IDs: (none)
- Eligible documents: 500
- Empty reason: (none)
- Filter counts: company_codes=2507/2507; periods=33721/500
- Scores and matched tokens: tbl_1fea09c00553183173733966a79d1ad2795094a415e277a0034fc4dd8aa6b616=10.832029 [2023, acb, assets, hợp, nhất, sản, total, tài, tổng]; tbl_a6c2f4b94692ec0008a9f9b550f9d3c27eceea641af58b740c93ac04817e7e44=10.826311 [2022, acb, assets, hợp, nhất, sản, total, tài, tổng]; tbl_c4d774fe3b6b41739275aee795d588229ebb9e3c42c3b8d8603e2298d39d0acb=9.976265 [2023, acb, assets, năm, sản, total, tài, tổng]; tbl_857e287a58e6dccf9d4978c2f0e458db1a4cdbe6ed08657cccd241bf104ffb78=9.555859 [2023, acb, assets, sản, total, tài, tổng]; tbl_d5d5b86f6ef69e35a78bb9db82b84dd28fa373134fda1aa81138aefa966fc3ea=9.305341 [2022, acb, assets, sản, total, tài, tổng]; tbl_0fbe34be5f3b1a78b7892752173842d14709c9f542060e336dce302b455a3423=9.248669 [2023, acb, assets, sản, total, tài, tổng]; tbl_002b5e0f4b961fa5dec48576b2ffd1c352cdc75f5f7255fc1c936fb424b736c5=9.242352 [2022, acb, assets, sản, total, tài, tổng]; tbl_0729ea24875c1d63f2087d26eda0c9d746caf1e821de548cfbd6da62a7e2c51a=9.242352 [2022, acb, assets, sản, total, tài, tổng]; tbl_64d7e7f0f1ac3d3e1adcf9d4f6092ea79edc56bb69eb0aaab03a0fc254196384=9.242352 [2022, acb, assets, sản, total, tài, tổng]; tbl_e60c30ceb769bf2592b82255b159dc1fffa5cfba24504a93be725eb0ac529c6f=9.242352 [2022, acb, assets, sản, total, tài, tổng]

### retq_d01701ab4c5e7d93a4af6f43f55ef17be41d4e2297131a4a388636edfdc8e8f2

- Question: Tra cứu lưu chuyển tiền thuần từ hoạt động kinh doanh của CTG năm 2023.
- Intent: lookup
- Failure: none
- Predicted table IDs: tbl_4bd8bdfdba928ef4a1171833de1f7945887d13472b5dea375220b74462297d26, tbl_a43bcc4e35e1f88425f4f6a930a25a3970fa17f09065d67ed5da52c5aa7faa16, tbl_5a6128f7f8df589d7546dd7e7364079ec4758a5747fe349ec88f9aa70020ebfd, tbl_e195adb9dfd5e9c080b12cd84e11aeb0526d9150681a22d2d119e55ebd8fc766, tbl_58a8f71151b6eef68c4e4b0f32549ed3fd91fe2fcbe753135fe1a71a62d7e343, tbl_eb52eb65997633f7856a3a6a5477a8e2258b6d311c6fad0a2b32aef7318c82fc, tbl_0b5dd2aabc25acf5f8820e071304554862c448762bd2e80c0d4efa8a0258efa0, tbl_915af040ac226dc91634595c1a1273e95a8fb4aa49da2dd79c2879fe9614eb3d, tbl_db9daba556b7f680ec8142f7b3f03abc44ed98cb30bda0f91d5c9eea62490853, tbl_42536ec8e457b3296f0cbaa24d65f7cd4d99c622d97c4b052ef8ecb108d287a9
- Gold table IDs: tbl_5a6128f7f8df589d7546dd7e7364079ec4758a5747fe349ec88f9aa70020ebfd
- Missing gold table IDs: (none)
- Eligible documents: 224
- Empty reason: (none)
- Filter counts: company_codes=1964/1964; periods=18228/224
- Scores and matched tokens: tbl_4bd8bdfdba928ef4a1171833de1f7945887d13472b5dea375220b74462297d26=16.696131 [2023, cash, chuyển, ctg, doanh, flow, hoạt, kinh, lưu, operating, thuần, tiền, từ, động]; tbl_a43bcc4e35e1f88425f4f6a930a25a3970fa17f09065d67ed5da52c5aa7faa16=16.696131 [2023, cash, chuyển, ctg, doanh, flow, hoạt, kinh, lưu, operating, thuần, tiền, từ, động]; tbl_5a6128f7f8df589d7546dd7e7364079ec4758a5747fe349ec88f9aa70020ebfd=16.228371 [2023, cash, chuyển, ctg, doanh, flow, hoạt, kinh, lưu, operating, thuần, tiền, từ, động]; tbl_e195adb9dfd5e9c080b12cd84e11aeb0526d9150681a22d2d119e55ebd8fc766=16.228371 [2023, cash, chuyển, ctg, doanh, flow, hoạt, kinh, lưu, operating, thuần, tiền, từ, động]; tbl_58a8f71151b6eef68c4e4b0f32549ed3fd91fe2fcbe753135fe1a71a62d7e343=16.191204 [2023, cash, chuyển, ctg, flow, hoạt, lưu, thuần, tiền, từ, động]; tbl_eb52eb65997633f7856a3a6a5477a8e2258b6d311c6fad0a2b32aef7318c82fc=16.191204 [2023, cash, chuyển, ctg, flow, hoạt, lưu, thuần, tiền, từ, động]; tbl_0b5dd2aabc25acf5f8820e071304554862c448762bd2e80c0d4efa8a0258efa0=9.364302 [2023, ctg, doanh, hoạt, kinh, thuần, từ, động]; tbl_915af040ac226dc91634595c1a1273e95a8fb4aa49da2dd79c2879fe9614eb3d=9.364302 [2023, ctg, doanh, hoạt, kinh, thuần, từ, động]; tbl_db9daba556b7f680ec8142f7b3f03abc44ed98cb30bda0f91d5c9eea62490853=9.364302 [2023, ctg, doanh, hoạt, kinh, thuần, từ, động]; tbl_42536ec8e457b3296f0cbaa24d65f7cd4d99c622d97c4b052ef8ecb108d287a9=8.125906 [2023, cash, chuyển, ctg, flow, lưu, tiền]

### retq_daf3295706f9a99546fdaafdda237fc03c541ae74522a82b46bd39ed6d863bb4

- Question: Tính tốc độ tăng trưởng doanh thu thuần về bán hàng và cung cấp dịch vụ của KHG từ năm 2022 đến năm 2023.
- Intent: growth
- Failure: none
- Predicted table IDs: tbl_c13ae0c50c647da5a1b0bc166d46660b8f713fcb202e344ec3fd40e70c3adc2d, tbl_ee81cc1b015207ce62bde5c5ec6aa42f82a2b639b7a703f5f00d72626c123a97, tbl_011c3b8764a6d4b0e025bd696501233c05eb7b49defe46e1cc5007b5794c51b4, tbl_b795c1b17487a1a67e4d1e3dd26e88e0b5f025f5964628ae1b224727e30e5959, tbl_fed883a2950883c1fe641123e9eea746557b8c20e6ba04de09a86eeaab4bb4f6, tbl_0ca6d656f0d9aadb4d80eeceea331b643e04f79a15d44b53eed75d9e6b74f951, tbl_5127567d7aec7c06b86a160ad7aa45fd9596f6b3bed21cc8be0d1bbd81d1a8d6, tbl_862e0f3eeb84077a57670cfaf0082f2666820c64f04a39ac8563db44525b46bb, tbl_727677479659576c76057acc2ecb378337b616b56f01d0d66655c1d4ae5550a5, tbl_c340a781395a1af3bd67d218847f36893611903c9a92fe0436a9ba43678256ad
- Gold table IDs: tbl_c13ae0c50c647da5a1b0bc166d46660b8f713fcb202e344ec3fd40e70c3adc2d, tbl_ee81cc1b015207ce62bde5c5ec6aa42f82a2b639b7a703f5f00d72626c123a97
- Missing gold table IDs: (none)
- Eligible documents: 253
- Empty reason: (none)
- Filter counts: company_codes=806/806; periods=33721/253
- Scores and matched tokens: tbl_c13ae0c50c647da5a1b0bc166d46660b8f713fcb202e344ec3fd40e70c3adc2d=18.441416 [2022, 2023, bán, cung, cấp, doanh, dịch, hàng, khg, net, revenue, thu, thuần, và, về, vụ]; tbl_ee81cc1b015207ce62bde5c5ec6aa42f82a2b639b7a703f5f00d72626c123a97=17.547295 [2022, bán, cung, cấp, doanh, dịch, hàng, khg, net, revenue, thu, thuần, và, về, vụ]; tbl_011c3b8764a6d4b0e025bd696501233c05eb7b49defe46e1cc5007b5794c51b4=17.027874 [2022, bán, cung, cấp, doanh, dịch, hàng, khg, net, revenue, thu, thuần, và, vụ]; tbl_b795c1b17487a1a67e4d1e3dd26e88e0b5f025f5964628ae1b224727e30e5959=17.026678 [2023, bán, cung, doanh, dịch, hàng, khg, net, revenue, thu, thuần, và, về, vụ]; tbl_fed883a2950883c1fe641123e9eea746557b8c20e6ba04de09a86eeaab4bb4f6=16.224360 [2023, bán, cung, cấp, doanh, hàng, khg, net, revenue, thu, thuần, và, vụ]; tbl_0ca6d656f0d9aadb4d80eeceea331b643e04f79a15d44b53eed75d9e6b74f951=15.849414 [2022, 2023, bán, cung, doanh, dịch, hàng, khg, net, revenue, thu, thuần, và, vụ]; tbl_5127567d7aec7c06b86a160ad7aa45fd9596f6b3bed21cc8be0d1bbd81d1a8d6=8.580218 [2022, 2023, doanh, khg, thuần, tính, từ]; tbl_862e0f3eeb84077a57670cfaf0082f2666820c64f04a39ac8563db44525b46bb=8.291899 [2022, 2023, doanh, khg, thuần, tính, từ]; tbl_727677479659576c76057acc2ecb378337b616b56f01d0d66655c1d4ae5550a5=7.906503 [2022, doanh, khg, thuần, tính, từ]; tbl_c340a781395a1af3bd67d218847f36893611903c9a92fe0436a9ba43678256ad=7.696219 [2023, doanh, khg, thuần, tính, từ]

### retq_ea3b207f9bcd90985977155a65aa00de9d9127eaa488d93394daa8e2c569ab71

- Question: So sánh lưu chuyển tiền thuần từ hoạt động kinh doanh của GEE giữa năm 2022 và năm 2023.
- Intent: compare
- Failure: none
- Predicted table IDs: tbl_3daf2d5a6f938548c4bfd09d449ec92fa96da2f3a61b61a049c906f5006afe97, tbl_de031e48c006ab9ab8a2ef63631c8431a15c0e4f911bcfc50b70eae9d7bed69e, tbl_2c5defc9f235a2cb5ca9c1ac242ad2d9494fda30a38428c64c86be259d58586e, tbl_4502ee46a12ed5d542ba6c763a886eb813e033b81bc4f4776c127dbeed17be35, tbl_7a45bad8e1f8cf633ddb7ad222f3406f2aa9814b3058f3740136c521a6fd0c31, tbl_401087a11ca7f72fa9b808e9ce7a94933db3891933875fc1932611eb7938d9a9, tbl_da1d47a8a53bfd926aff4a1e6b269387bb3032d369efd8ea27fb5a46eed1214f, tbl_8e14f884401a116c3fdb6f7089e40ea7c9dc909e7b4a99b33dd8b2c380fe282a, tbl_40959cf176ed6f953824f9fb0ae6bb8f52fdffb9576332506d325471e80f2968, tbl_70e9fae7dd5bab38f9c6af9926df6aea102db7162ae8b097dcc9ae1727f35f47
- Gold table IDs: tbl_3daf2d5a6f938548c4bfd09d449ec92fa96da2f3a61b61a049c906f5006afe97, tbl_4502ee46a12ed5d542ba6c763a886eb813e033b81bc4f4776c127dbeed17be35
- Missing gold table IDs: (none)
- Eligible documents: 319
- Empty reason: (none)
- Filter counts: company_codes=936/936; periods=33721/319
- Scores and matched tokens: tbl_3daf2d5a6f938548c4bfd09d449ec92fa96da2f3a61b61a049c906f5006afe97=19.372528 [2022, 2023, cash, chuyển, doanh, flow, gee, hoạt, kinh, lưu, operating, thuần, tiền, từ, động]; tbl_de031e48c006ab9ab8a2ef63631c8431a15c0e4f911bcfc50b70eae9d7bed69e=19.248081 [2022, 2023, cash, chuyển, doanh, flow, gee, hoạt, kinh, lưu, operating, thuần, tiền, từ, động]; tbl_2c5defc9f235a2cb5ca9c1ac242ad2d9494fda30a38428c64c86be259d58586e=18.741270 [2023, cash, chuyển, doanh, flow, gee, hoạt, kinh, lưu, operating, thuần, tiền, từ, động]; tbl_4502ee46a12ed5d542ba6c763a886eb813e033b81bc4f4776c127dbeed17be35=18.735836 [2022, cash, chuyển, doanh, flow, gee, hoạt, kinh, lưu, operating, thuần, tiền, từ, động]; tbl_7a45bad8e1f8cf633ddb7ad222f3406f2aa9814b3058f3740136c521a6fd0c31=18.601702 [2022, cash, chuyển, doanh, flow, gee, hoạt, kinh, lưu, operating, thuần, tiền, từ, động]; tbl_401087a11ca7f72fa9b808e9ce7a94933db3891933875fc1932611eb7938d9a9=14.192246 [2022, 2023, cash, chuyển, flow, gee, hoạt, lưu, thuần, tiền, từ, động]; tbl_da1d47a8a53bfd926aff4a1e6b269387bb3032d369efd8ea27fb5a46eed1214f=14.192246 [2022, 2023, cash, chuyển, flow, gee, hoạt, lưu, thuần, tiền, từ, động]; tbl_8e14f884401a116c3fdb6f7089e40ea7c9dc909e7b4a99b33dd8b2c380fe282a=13.422557 [2023, cash, chuyển, flow, gee, hoạt, lưu, thuần, tiền, từ, động]; tbl_40959cf176ed6f953824f9fb0ae6bb8f52fdffb9576332506d325471e80f2968=13.415933 [2022, cash, chuyển, flow, gee, hoạt, lưu, thuần, tiền, từ, động]; tbl_70e9fae7dd5bab38f9c6af9926df6aea102db7162ae8b097dcc9ae1727f35f47=13.415933 [2022, cash, chuyển, flow, gee, hoạt, lưu, thuần, tiền, từ, động]

