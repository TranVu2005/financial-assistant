# Day 8 BM25 Retrieval Evaluation

- Dataset fingerprint: `422df141c935d46bfd14302abec50f32380e6e4c012159f8ad0ae5560c8a446a`
- Questions: 70
- Precision@10: 0.150000
- Recall@10: 0.880952
- F2@10: 0.422455

## Metrics by intent

| Intent | Precision@10 | Recall@10 | F2@10 | True positives |
| --- | ---: | ---: | ---: | ---: |
| compare | 0.165217 | 0.876812 | 0.453797 | 38 |
| growth | 0.160870 | 0.902174 | 0.450142 | 37 |
| lookup | 0.125000 | 0.864583 | 0.365884 | 30 |

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

### retq_035ab7b50aa16f2da0835e329105ef45254fff823aab0375464fe935992d7301

- Question: Tra cứu đồng thời bốn bảng thuyết minh về danh sách công ty con của HPG năm 2017.
- Intent: lookup
- Failure: none
- Predicted table IDs: tbl_0ef37af53d44c4b68d57f7bf2605767e25d20cab6f6448d7ef9e80784972e5ab, tbl_74222750730e1e2328a41f6ef1c3efd98c569243445c7f74cd7f187541483c07, tbl_8527949ec89e7412439126f1e06bbe4985ee0f12c304b0eaf7e3ff29e9698884, tbl_acd28a9e88fc9ff9395da95879b7ff77c6843ea9bbbc2e0a11e486c302a2389b, tbl_dcb21cd18e2ffedae94cf26e787516fb797c8ef716bbc95210e781e3b035e5b8
- Gold table IDs: tbl_0ef37af53d44c4b68d57f7bf2605767e25d20cab6f6448d7ef9e80784972e5ab, tbl_74222750730e1e2328a41f6ef1c3efd98c569243445c7f74cd7f187541483c07, tbl_acd28a9e88fc9ff9395da95879b7ff77c6843ea9bbbc2e0a11e486c302a2389b, tbl_dcb21cd18e2ffedae94cf26e787516fb797c8ef716bbc95210e781e3b035e5b8
- Missing gold table IDs: (none)
- Eligible documents: 5
- Empty reason: (none)
- Filter counts: company_codes=1299/1299; periods=14081/155; statement_types=1479/5
- Scores and matched tokens: tbl_0ef37af53d44c4b68d57f7bf2605767e25d20cab6f6448d7ef9e80784972e5ab=6.994010 [2017, hpg, minh, năm, thuyết]; tbl_74222750730e1e2328a41f6ef1c3efd98c569243445c7f74cd7f187541483c07=6.994010 [2017, hpg, minh, năm, thuyết]; tbl_8527949ec89e7412439126f1e06bbe4985ee0f12c304b0eaf7e3ff29e9698884=6.994010 [2017, hpg, minh, năm, thuyết]; tbl_acd28a9e88fc9ff9395da95879b7ff77c6843ea9bbbc2e0a11e486c302a2389b=6.994010 [2017, hpg, minh, năm, thuyết]; tbl_dcb21cd18e2ffedae94cf26e787516fb797c8ef716bbc95210e781e3b035e5b8=6.994010 [2017, hpg, minh, năm, thuyết]

### retq_07495f696860ec468fa5a71f67fa84b86bcfdbcd1b55d1deaf3fbbbebf45a52b

- Question: Đối chiếu thông tin bộ phận, nợ phải thu quá hạn, biến động TSCĐ và các khoản vay của MPC năm 2018.
- Intent: compare
- Failure: none
- Predicted table IDs: tbl_78a20886a2ca480358ea38bb0a46bbdfd854750f9d2c25ec0353d4c7d48fa7d0, tbl_47766b9215be28985b43310b8e9582d7d2a021eb30a2c5111d961fadfba3a601, tbl_b3cf5fb3467154be3a8e190746175350ff2753d9ef811ca3e8895c32d7cb3679, tbl_f22c8c9bb63e049b4ad24e0bb99698efb063cb163f4d4df898b660484cfc7f05, tbl_fd18f6f79d273f0d6b54e6b42d2b0dd05c154e8a3863ec920b5b8594273cf9c1, tbl_311914d8ba482d5dac25a2292d0ffed4ea3d89b3680344ec3d05c00d2e109a9f, tbl_abe7c042c924c885ecfa471d654d72a5daf63e9429ec64ed19ff290061fec508, tbl_d4f4dd8262f580dc88a7263f923da481c4075529d91cd17d735b536d861979ca, tbl_e25049dc10ac4b8d59a3d1bc3db8d0893e0382c01e361580e8405651ef572a51
- Gold table IDs: tbl_78a20886a2ca480358ea38bb0a46bbdfd854750f9d2c25ec0353d4c7d48fa7d0, tbl_abe7c042c924c885ecfa471d654d72a5daf63e9429ec64ed19ff290061fec508, tbl_b3cf5fb3467154be3a8e190746175350ff2753d9ef811ca3e8895c32d7cb3679, tbl_e25049dc10ac4b8d59a3d1bc3db8d0893e0382c01e361580e8405651ef572a51
- Missing gold table IDs: (none)
- Eligible documents: 9
- Empty reason: (none)
- Filter counts: company_codes=1493/1493; periods=14691/183; statement_types=1479/9
- Scores and matched tokens: tbl_78a20886a2ca480358ea38bb0a46bbdfd854750f9d2c25ec0353d4c7d48fa7d0=6.489413 [2018, assets, mpc, năm, nợ, phải]; tbl_47766b9215be28985b43310b8e9582d7d2a021eb30a2c5111d961fadfba3a601=4.259589 [2018, mpc, năm]; tbl_b3cf5fb3467154be3a8e190746175350ff2753d9ef811ca3e8895c32d7cb3679=4.222355 [2018, mpc, năm]; tbl_f22c8c9bb63e049b4ad24e0bb99698efb063cb163f4d4df898b660484cfc7f05=4.022792 [2018, mpc, năm]; tbl_fd18f6f79d273f0d6b54e6b42d2b0dd05c154e8a3863ec920b5b8594273cf9c1=4.022792 [2018, mpc, năm]; tbl_311914d8ba482d5dac25a2292d0ffed4ea3d89b3680344ec3d05c00d2e109a9f=4.001965 [2018, mpc, năm]; tbl_abe7c042c924c885ecfa471d654d72a5daf63e9429ec64ed19ff290061fec508=4.001965 [2018, mpc, năm]; tbl_d4f4dd8262f580dc88a7263f923da481c4075529d91cd17d735b536d861979ca=4.001965 [2018, mpc, năm]; tbl_e25049dc10ac4b8d59a3d1bc3db8d0893e0382c01e361580e8405651ef572a51=4.001965 [2018, mpc, năm]

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

### retq_0eaaf1a9804038a40987869bdcc2da226dbe3aba08d02943873b029d4f172848

- Question: Tra cứu cho vay khách hàng và chứng khoán đầu tư của STB tại cuối năm 2024.
- Intent: lookup
- Failure: zero_gold_hits
- Predicted table IDs: tbl_deecbbbc5fb25e0418697f5f2a924972d60e8c8f94ae272ac64a4f4d5e176f85, tbl_14a9f0bf0f63c3dd4a4dbbb9ebd8171a56a32dda99b91f63989ab05fda26e760, tbl_5f555d03dbc0fd5375c6f4834ad689fb057eb3db74d98930f9bb58c39565ec89, tbl_95b6c4f0fb5b5817be871593054757228694a80d8038271a890255318b6c30b9, tbl_16e88505689822625693eb48655b73ffe4cbe338ffb2e8f98b1f565ef65f77fa, tbl_85f0775f1a5f4d50d26a63a29d74e76435cbbc35e44b60742b2c75a3b78c412d, tbl_f786a786c466edea0518a270e25e3fce5b0b0a5bdf4e39dc99712dac6fea10e3, tbl_6a89e765332710b0c8ada3d87ff56fc5697f86d6fe059439706ee7b0bdd24136, tbl_f0020d7700414f371b26c5410f3aebb353894610ca00598755529f5bc7092bbe, tbl_4f7f0705a8eaca16e22481d37547c9ada7ce66ae9a021c8d33174aaa2a2140aa
- Gold table IDs: tbl_43553d14f989fbed07d18b01f093f4c6f25d89e30917cde81ae3217bb747b42e
- Missing gold table IDs: tbl_43553d14f989fbed07d18b01f093f4c6f25d89e30917cde81ae3217bb747b42e
- Eligible documents: 22
- Empty reason: (none)
- Filter counts: company_codes=2568/2568; periods=20269/294; statement_types=7745/22
- Scores and matched tokens: tbl_deecbbbc5fb25e0418697f5f2a924972d60e8c8f94ae272ac64a4f4d5e176f85=6.508770 [2024, của, hàng, năm, stb, tại, và]; tbl_14a9f0bf0f63c3dd4a4dbbb9ebd8171a56a32dda99b91f63989ab05fda26e760=5.869369 [2024, của, hàng, năm, stb, tại, và]; tbl_5f555d03dbc0fd5375c6f4834ad689fb057eb3db74d98930f9bb58c39565ec89=5.785300 [2024, của, năm, stb, tại, và]; tbl_95b6c4f0fb5b5817be871593054757228694a80d8038271a890255318b6c30b9=5.720798 [2024, của, hàng, năm, stb, tại, và]; tbl_16e88505689822625693eb48655b73ffe4cbe338ffb2e8f98b1f565ef65f77fa=5.020646 [2024, của, năm, stb, tại, và]; tbl_85f0775f1a5f4d50d26a63a29d74e76435cbbc35e44b60742b2c75a3b78c412d=4.853314 [2024, của, hàng, năm, stb, tư]; tbl_f786a786c466edea0518a270e25e3fce5b0b0a5bdf4e39dc99712dac6fea10e3=4.853314 [2024, của, hàng, năm, stb, tư]; tbl_6a89e765332710b0c8ada3d87ff56fc5697f86d6fe059439706ee7b0bdd24136=4.826817 [2024, của, hàng, năm, stb, tư]; tbl_f0020d7700414f371b26c5410f3aebb353894610ca00598755529f5bc7092bbe=4.826817 [2024, của, hàng, năm, stb, tư]; tbl_4f7f0705a8eaca16e22481d37547c9ada7ce66ae9a021c8d33174aaa2a2140aa=4.673717 [2024, của, hàng, năm, stb, tư]

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

### retq_140b9a346d497feed47dd86d6eb1a2bb0fa9c2c4ef873306cad2b631a8e347ce

- Question: Tra cứu ba bảng thuyết minh phân tích thời hạn và rủi ro của tài sản, nợ tài chính MSB năm 2018.
- Intent: lookup
- Failure: none
- Predicted table IDs: tbl_9f667f366be2ea2b5beb8ed773313ca9987fc274409f14f5c6d6199097354186, tbl_fea6fa0332eb5f626ce47910b6e5f3711f42dd1f2224ef3d3690e328a934b453, tbl_6bd285d9d9864b9ef0726f8eb5ec5de245823b9ba74ab1f9c63425237e71661f, tbl_00fbd970f4d52c037b1c28401db25e2354325d6db51f040ae491ae18a5bb254d, tbl_0ed83075a1511290fffba92549213ea1846a0e477b20b67cc827fa4fd0df144b
- Gold table IDs: tbl_0ed83075a1511290fffba92549213ea1846a0e477b20b67cc827fa4fd0df144b, tbl_6bd285d9d9864b9ef0726f8eb5ec5de245823b9ba74ab1f9c63425237e71661f, tbl_9f667f366be2ea2b5beb8ed773313ca9987fc274409f14f5c6d6199097354186
- Missing gold table IDs: (none)
- Eligible documents: 5
- Empty reason: (none)
- Filter counts: company_codes=1971/1971; periods=14691/268; statement_types=1479/5
- Scores and matched tokens: tbl_9f667f366be2ea2b5beb8ed773313ca9987fc274409f14f5c6d6199097354186=10.338318 [2018, chính, minh, msb, năm, nợ, sản, thuyết, tài]; tbl_fea6fa0332eb5f626ce47910b6e5f3711f42dd1f2224ef3d3690e328a934b453=10.046637 [2018, chính, minh, msb, năm, nợ, sản, thuyết, tài]; tbl_6bd285d9d9864b9ef0726f8eb5ec5de245823b9ba74ab1f9c63425237e71661f=8.506163 [2018, chính, minh, msb, năm, nợ, sản, thuyết, tài]; tbl_00fbd970f4d52c037b1c28401db25e2354325d6db51f040ae491ae18a5bb254d=8.217426 [2018, chính, minh, msb, năm, thuyết, tài]; tbl_0ed83075a1511290fffba92549213ea1846a0e477b20b67cc827fa4fd0df144b=7.401114 [2018, chính, minh, msb, năm, nợ, thuyết, tài]

### retq_15f357970d80fc9aa487f0da8d252b505d1217d23c365019000c60a366d1f620

- Question: Tính biến động LNST chưa phân phối trong năm của VSC năm 2019.
- Intent: growth
- Failure: none
- Predicted table IDs: tbl_38f9307818b4896664a9a4e33bb029b85ffcc5e38a94737bc517bb3cc9bff227, tbl_ef062a6d96cb28369d44348dccbf2403197702413739fc0d26447f099553ecf5, tbl_5d066d274e956e60faf308d1d5f4693b10186565b291e93c8b2194080ac63470, tbl_11b7f9809d7597e269a71dd7df92c22be65b0b5d9fb77b9f6eaba71046a2772d, tbl_77252a5a4e20fa81fd6e375e102996dd61df2e16ca9720c2d9fc52e7c77c7c33, tbl_e8aca82b1c2b3a683517a6d0915ab2a55e7250e40548280c02eed9bf12ba5d50, tbl_bf1a1a2dc38dee8e4f91de8e015f2fcbc75f64a0bad73490ae18b39836684ad1, tbl_1cf394854dec517933c796f83baad8f1d2b75ce80b830db3580f0f07a4d88331, tbl_4127c150ee8bbba394d2bad1a1b1b55da8aafa1f9fc756eeb60e95ed8e0bf2e3, tbl_317b15032ad950ec577d930feb61a6985530fe167771184c493c67f515edacaf
- Gold table IDs: tbl_ef062a6d96cb28369d44348dccbf2403197702413739fc0d26447f099553ecf5
- Missing gold table IDs: (none)
- Eligible documents: 14
- Empty reason: (none)
- Filter counts: company_codes=1101/1101; periods=15553/118; statement_types=7862/14
- Scores and matched tokens: tbl_38f9307818b4896664a9a4e33bb029b85ffcc5e38a94737bc517bb3cc9bff227=10.332428 [2019, chưa, của, earnings, phân, phối, retained, vsc]; tbl_ef062a6d96cb28369d44348dccbf2403197702413739fc0d26447f099553ecf5=9.350465 [2019, chưa, của, earnings, phân, phối, retained, vsc]; tbl_5d066d274e956e60faf308d1d5f4693b10186565b291e93c8b2194080ac63470=4.698732 [2019, của, phân, trong, vsc, động]; tbl_11b7f9809d7597e269a71dd7df92c22be65b0b5d9fb77b9f6eaba71046a2772d=4.261647 [2019, tính, vsc]; tbl_77252a5a4e20fa81fd6e375e102996dd61df2e16ca9720c2d9fc52e7c77c7c33=4.233267 [2019, tính, vsc]; tbl_e8aca82b1c2b3a683517a6d0915ab2a55e7250e40548280c02eed9bf12ba5d50=4.233267 [2019, tính, vsc]; tbl_bf1a1a2dc38dee8e4f91de8e015f2fcbc75f64a0bad73490ae18b39836684ad1=4.207652 [2019, trong, vsc, động]; tbl_1cf394854dec517933c796f83baad8f1d2b75ce80b830db3580f0f07a4d88331=4.183631 [2019, trong, vsc, động]; tbl_4127c150ee8bbba394d2bad1a1b1b55da8aafa1f9fc756eeb60e95ed8e0bf2e3=4.183631 [2019, trong, vsc, động]; tbl_317b15032ad950ec577d930feb61a6985530fe167771184c493c67f515edacaf=3.846044 [2019, của, trong, vsc, động]

### retq_184e75fa031778165edf54a7777b54c7cf874c7b1df60857be094b591c2564f1

- Question: Đối chiếu doanh thu, giá vốn và lợi nhuận giữa báo cáo riêng và hợp nhất của MPC năm 2017.
- Intent: compare
- Failure: none
- Predicted table IDs: tbl_32bd17ed83334db05b350fa3b7aa7cd4f017d1a6f7cf066df00b8873eeb918c5, tbl_112336df57e50d373f3d799b7322ab7304fc5990d1e8df45c30f368ce06d8f78, tbl_3dc006385961d98936509995ebb413e1a1afed2a8d23c825576a915d6ab3db90, tbl_f4d515f5cc354918f75b9d29ef7de64127c7fda8fe32e57ec8bed74b4cb766c2, tbl_59f6125612a069dd3fd282429b09a086fc8ebd8b2b5fcb97b7823f31562d5cb4, tbl_a3ea1a8df17a88bf5461a98e3a22d486b91670496b03edf8cfa7afd1aaae86ed, tbl_b6df727dc60b80da7b21e9d0d054840527f603d13373807e9eba193ef21ef0b1, tbl_632f1f029c6a8fa14841c6b98cd3f36733843d268520cf0a9d6bd0ab94f853cc, tbl_4ab73d68ec36d6d974560be6e478c23c22437c92fe536d40878a68a1bfee206f, tbl_cb4c43730dbcbd08b2b9299af256e1db3bf811ac4f7c8802d863a0abb7a2164e
- Gold table IDs: tbl_3dc006385961d98936509995ebb413e1a1afed2a8d23c825576a915d6ab3db90, tbl_f4d515f5cc354918f75b9d29ef7de64127c7fda8fe32e57ec8bed74b4cb766c2
- Missing gold table IDs: (none)
- Eligible documents: 17
- Empty reason: (none)
- Filter counts: company_codes=1493/1493; periods=14081/185; statement_types=7862/17
- Scores and matched tokens: tbl_32bd17ed83334db05b350fa3b7aa7cd4f017d1a6f7cf066df00b8873eeb918c5=12.288881 [2017, cost, của, doanh, giá, goods, mpc, năm, of, sold, thu, vốn]; tbl_112336df57e50d373f3d799b7322ab7304fc5990d1e8df45c30f368ce06d8f78=11.910583 [2017, cost, của, doanh, giá, goods, mpc, of, sold, thu, vốn]; tbl_3dc006385961d98936509995ebb413e1a1afed2a8d23c825576a915d6ab3db90=11.831985 [2017, cost, của, doanh, giá, goods, mpc, năm, of, sold, thu, vốn]; tbl_f4d515f5cc354918f75b9d29ef7de64127c7fda8fe32e57ec8bed74b4cb766c2=11.465837 [2017, cost, của, doanh, giá, goods, mpc, of, sold, thu, vốn]; tbl_59f6125612a069dd3fd282429b09a086fc8ebd8b2b5fcb97b7823f31562d5cb4=8.681617 [2017, báo, cáo, doanh, hợp, mpc, nhất]; tbl_a3ea1a8df17a88bf5461a98e3a22d486b91670496b03edf8cfa7afd1aaae86ed=8.681617 [2017, báo, cáo, doanh, hợp, mpc, nhất]; tbl_b6df727dc60b80da7b21e9d0d054840527f603d13373807e9eba193ef21ef0b1=7.696703 [2017, báo, cáo, doanh, mpc, riêng]; tbl_632f1f029c6a8fa14841c6b98cd3f36733843d268520cf0a9d6bd0ab94f853cc=7.071197 [2017, của, lợi, mpc, nhuận, năm, vốn]; tbl_4ab73d68ec36d6d974560be6e478c23c22437c92fe536d40878a68a1bfee206f=6.819741 [2017, của, lợi, mpc, nhuận, vốn]; tbl_cb4c43730dbcbd08b2b9299af256e1db3bf811ac4f7c8802d863a0abb7a2164e=6.325031 [2017, của, lợi, mpc, nhuận]

### retq_18d416f9aaed3300ef83d93fd5743112c68fc2f6a11d518cc4f029be24e3766a

- Question: So sánh kết quả kinh doanh riêng và hợp nhất của IJC năm 2017.
- Intent: compare
- Failure: none
- Predicted table IDs: tbl_0b502faf679af4154a8aea80e267299e5ef6f27d7f5c98ed0226d0657f7bda2c, tbl_4ba24948c737a41ca5cb00e64649777ed1849f3f5d055541f5530ec83d7cb6b5, tbl_8e6d80ed2a3b027a7b4c60d61a1e86f98d76461f35e7cf875be9f62a27fcc2e5, tbl_57fedb3fd38155dd53d7a6d6badf58a22b8b47a31d45d03665128d7da6480575, tbl_65208f0c97b44a5c608c21501e9a812bea6729ce03508ca289bd7076f8c15aca, tbl_6899e56c7bbe2c2083ff617270702d2a60a970b3ad6ae2edf14cafdf13cc1bd3, tbl_80a13beb3df521a1e6b1d61c3a3f3b235528471a7e9c6d066f05fb076bc1e902
- Gold table IDs: tbl_4ba24948c737a41ca5cb00e64649777ed1849f3f5d055541f5530ec83d7cb6b5, tbl_80a13beb3df521a1e6b1d61c3a3f3b235528471a7e9c6d066f05fb076bc1e902
- Missing gold table IDs: (none)
- Eligible documents: 7
- Empty reason: (none)
- Filter counts: company_codes=1483/1483; periods=14081/185; statement_types=7862/7
- Scores and matched tokens: tbl_0b502faf679af4154a8aea80e267299e5ef6f27d7f5c98ed0226d0657f7bda2c=4.400786 [2017, của, doanh, ijc, kinh, và]; tbl_4ba24948c737a41ca5cb00e64649777ed1849f3f5d055541f5530ec83d7cb6b5=4.400786 [2017, của, doanh, ijc, kinh, và]; tbl_8e6d80ed2a3b027a7b4c60d61a1e86f98d76461f35e7cf875be9f62a27fcc2e5=4.174366 [2017, ijc, kinh, và]; tbl_57fedb3fd38155dd53d7a6d6badf58a22b8b47a31d45d03665128d7da6480575=3.871773 [2017, doanh, ijc]; tbl_65208f0c97b44a5c608c21501e9a812bea6729ce03508ca289bd7076f8c15aca=3.871773 [2017, doanh, ijc]; tbl_6899e56c7bbe2c2083ff617270702d2a60a970b3ad6ae2edf14cafdf13cc1bd3=2.701876 [2017, ijc]; tbl_80a13beb3df521a1e6b1d61c3a3f3b235528471a7e9c6d066f05fb076bc1e902=2.701876 [2017, ijc]

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

### retq_21d0fcc00dee1bd24d014fda92df6768482761e46ff4d1b809e4a8a61a8f2e58

- Question: Tính biến động đầu năm-cuối năm của tiền gửi có kỳ hạn và rà soát các khoản đầu tư góp vốn của SCR năm 2017.
- Intent: growth
- Failure: none
- Predicted table IDs: tbl_318954f4955b3e8aa87f41b3ca424b293eeffbfabc6304b6ab4320977208fe72, tbl_ea5f4f7aa6b8fde4a4796131f7d2d55a8c7ec2336cc15d5da0510291045c2cf7, tbl_57a5438a0cf600de29b978a1ae3d7314169bfa9cbf3cf6663199219a0596c247, tbl_7329e83652425f23a28ac45107aac79e36d2facd605a9522e9f19f1af7ecef4a, tbl_e5caee0ac7cf780a2c9e4996755d728733365dd839f37c9f6a3151bf11a8153c, tbl_34e5b1a821a2f90639b9100bceaa49456b87027d264a98797ea7ba2f6e80804e, tbl_3532c3707c1ff854e19994ddc2665ab30918c5f04377c9273ce9ee55f70160aa, tbl_4aa98ca75b1ca5741b6af712d29db5f4654d1424755a322097e3ba6f8560cb33, tbl_649b66d0ab86df87d903d5fa7b82dd5191f9ab02e8fd757790073f341c7ca742
- Gold table IDs: tbl_34e5b1a821a2f90639b9100bceaa49456b87027d264a98797ea7ba2f6e80804e, tbl_4aa98ca75b1ca5741b6af712d29db5f4654d1424755a322097e3ba6f8560cb33, tbl_7329e83652425f23a28ac45107aac79e36d2facd605a9522e9f19f1af7ecef4a
- Missing gold table IDs: (none)
- Eligible documents: 9
- Empty reason: (none)
- Filter counts: company_codes=1713/1713; periods=14081/180; statement_types=1479/9
- Scores and matched tokens: tbl_318954f4955b3e8aa87f41b3ca424b293eeffbfabc6304b6ab4320977208fe72=4.272853 [2017, các, năm, scr]; tbl_ea5f4f7aa6b8fde4a4796131f7d2d55a8c7ec2336cc15d5da0510291045c2cf7=4.221764 [2017, các, năm, scr, và]; tbl_57a5438a0cf600de29b978a1ae3d7314169bfa9cbf3cf6663199219a0596c247=4.210013 [2017, năm, scr]; tbl_7329e83652425f23a28ac45107aac79e36d2facd605a9522e9f19f1af7ecef4a=3.985540 [2017, năm, scr]; tbl_e5caee0ac7cf780a2c9e4996755d728733365dd839f37c9f6a3151bf11a8153c=3.985540 [2017, năm, scr]; tbl_34e5b1a821a2f90639b9100bceaa49456b87027d264a98797ea7ba2f6e80804e=3.964969 [2017, năm, scr]; tbl_3532c3707c1ff854e19994ddc2665ab30918c5f04377c9273ce9ee55f70160aa=3.964969 [2017, năm, scr]; tbl_4aa98ca75b1ca5741b6af712d29db5f4654d1424755a322097e3ba6f8560cb33=3.964969 [2017, năm, scr]; tbl_649b66d0ab86df87d903d5fa7b82dd5191f9ab02e8fd757790073f341c7ca742=3.964969 [2017, năm, scr]

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

### retq_2bf1a7ff817063f8a7583a5809b021257faa2bf25252135d085d38ffeb21a84f

- Question: Tra cứu cơ cấu tài sản và nợ của NVB theo nhóm kỳ hạn năm 2019.
- Intent: lookup
- Failure: none
- Predicted table IDs: tbl_1b932b3f9aa03337c4a013e1b6cd1aedcb3addb9a2a2185d94426c61ecb837d2, tbl_c8fbc2b981125417a7614bc16fd45d16cd33df7864fdda0b0749df5eeac1cc2b, tbl_d85d2cd6b8b408ebc4311a55d97462a0ae569f4667315a70b0691211ab9d7b65, tbl_4a17fd07f7ab555fb0955923727d6ea5a8d9af8193ef7acb0da6ae64d111825c, tbl_33d0dfa4a33648b1fa025e0b278cea15fba3ff0eb3c4250697823bf2974baf0a, tbl_3ca30bcf1edb478fed860b2143175c5bba391be1d60883ada734b7e7ca0be801, tbl_6e40ddbd9d235ec7b0ccf666b59acdeecd605fad72717a27fc1746dda86b2d97, tbl_93b499c564189d710f734c666518d9c568aa67a204df2daee9d01d517edb21e4, tbl_243edac63d4d5660bdd69a303245bf4a62b84c37a7365e4e7e774e33b4efde03, tbl_4874887ba661fce62444484ea537e1f9e6eaf0d51394b0c172359162d60b8c99
- Gold table IDs: tbl_4a17fd07f7ab555fb0955923727d6ea5a8d9af8193ef7acb0da6ae64d111825c
- Missing gold table IDs: (none)
- Eligible documents: 16
- Empty reason: (none)
- Filter counts: company_codes=1315/1315; periods=15553/170; statement_types=7745/16
- Scores and matched tokens: tbl_1b932b3f9aa03337c4a013e1b6cd1aedcb3addb9a2a2185d94426c61ecb837d2=12.477587 [2019, của, hạn, kỳ, nhóm, nvb, năm, nợ, sản, theo, tài, và]; tbl_c8fbc2b981125417a7614bc16fd45d16cd33df7864fdda0b0749df5eeac1cc2b=12.477587 [2019, của, hạn, kỳ, nhóm, nvb, năm, nợ, sản, theo, tài, và]; tbl_d85d2cd6b8b408ebc4311a55d97462a0ae569f4667315a70b0691211ab9d7b65=12.477587 [2019, của, hạn, kỳ, nhóm, nvb, năm, nợ, sản, theo, tài, và]; tbl_4a17fd07f7ab555fb0955923727d6ea5a8d9af8193ef7acb0da6ae64d111825c=12.233433 [2019, của, hạn, kỳ, nhóm, nvb, năm, nợ, sản, theo, tài, và]; tbl_33d0dfa4a33648b1fa025e0b278cea15fba3ff0eb3c4250697823bf2974baf0a=8.649840 [2019, của, nvb, năm, nợ, sản, theo, tài, và]; tbl_3ca30bcf1edb478fed860b2143175c5bba391be1d60883ada734b7e7ca0be801=8.649840 [2019, của, nvb, năm, nợ, sản, theo, tài, và]; tbl_6e40ddbd9d235ec7b0ccf666b59acdeecd605fad72717a27fc1746dda86b2d97=8.649840 [2019, của, nvb, năm, nợ, sản, theo, tài, và]; tbl_93b499c564189d710f734c666518d9c568aa67a204df2daee9d01d517edb21e4=8.649840 [2019, của, nvb, năm, nợ, sản, theo, tài, và]; tbl_243edac63d4d5660bdd69a303245bf4a62b84c37a7365e4e7e774e33b4efde03=6.876189 [2019, của, nvb, năm, nợ, sản, tài]; tbl_4874887ba661fce62444484ea537e1f9e6eaf0d51394b0c172359162d60b8c99=6.876189 [2019, của, nvb, năm, nợ, sản, tài]

### retq_2c6e03ad7f1a391d3673e2d8148a083ee759a1c1149c6f379241323f8b20221d

- Question: So sánh LCTT trực tiếp của MBB giữa năm 2016 và 2017.
- Intent: compare
- Failure: none
- Predicted table IDs: tbl_66145ccc8b0f9a55df597dbd5be0a0840cacbd1895d2d2c4159e987df6ba4d9e, tbl_49c8c3e98158abdcaf2c89743d032f67bca709ab1b0e5af3ed5d747f3afd16c8, tbl_dd4d1cf0f655a5f109ecfb691f7a4f4eb191b3bf318591236562ca30865d8778, tbl_28cbee1f7a012d945d486243f7713de00cbcd8e1747da2e854059c161c42210c, tbl_8d823f3bb7f9dcebe414cd90967f20757858e54ea08c02a7a5a2b09360ade6a3, tbl_94ebf880f91c6440d9af7c4bfd54ff0cdf0359468871986271d2259943fd70bd, tbl_05fb80419b0107345b1e14304f5dfb94ec937a55255aea80e017559652aca7d3, tbl_44a3537ed06e43aaf5cd3a39c584c5d695124803bb5ba57fe5a6e3fd8c27aea7, tbl_44f35a9ae03cf57b6d88fe72407e981bd0d682913aafaaa0a7dceaf54c18359e, tbl_d747dfb296e4d38957d0576da58573f427a0cc64e7de5335eaff7c3f8801b074
- Gold table IDs: tbl_49c8c3e98158abdcaf2c89743d032f67bca709ab1b0e5af3ed5d747f3afd16c8
- Missing gold table IDs: (none)
- Eligible documents: 31
- Empty reason: (none)
- Filter counts: company_codes=2367/2367; periods=24087/439; statement_types=13712/31
- Scores and matched tokens: tbl_66145ccc8b0f9a55df597dbd5be0a0840cacbd1895d2d2c4159e987df6ba4d9e=7.732103 [2016, 2017, mbb, năm, tiếp, trực]; tbl_49c8c3e98158abdcaf2c89743d032f67bca709ab1b0e5af3ed5d747f3afd16c8=7.415800 [2016, 2017, mbb, năm, tiếp, trực]; tbl_dd4d1cf0f655a5f109ecfb691f7a4f4eb191b3bf318591236562ca30865d8778=6.592423 [2016, mbb, năm, tiếp, trực]; tbl_28cbee1f7a012d945d486243f7713de00cbcd8e1747da2e854059c161c42210c=5.043664 [2016, 2017, của, mbb, năm]; tbl_8d823f3bb7f9dcebe414cd90967f20757858e54ea08c02a7a5a2b09360ade6a3=5.043664 [2016, 2017, của, mbb, năm]; tbl_94ebf880f91c6440d9af7c4bfd54ff0cdf0359468871986271d2259943fd70bd=5.043664 [2016, 2017, của, mbb, năm]; tbl_05fb80419b0107345b1e14304f5dfb94ec937a55255aea80e017559652aca7d3=4.587500 [2016, của, mbb, năm]; tbl_44a3537ed06e43aaf5cd3a39c584c5d695124803bb5ba57fe5a6e3fd8c27aea7=4.587500 [2016, của, mbb, năm]; tbl_44f35a9ae03cf57b6d88fe72407e981bd0d682913aafaaa0a7dceaf54c18359e=4.587500 [2016, của, mbb, năm]; tbl_d747dfb296e4d38957d0576da58573f427a0cc64e7de5335eaff7c3f8801b074=4.587500 [2016, của, mbb, năm]

### retq_3311745c98ba5efc0f6b7e78f74fbb69d061ec16f6f266fa0d82f83b1564dee0

- Question: So sánh doanh thu, giá vốn và chi phí tài chính của VJC giữa năm 2018 và 2019.
- Intent: compare
- Failure: none
- Predicted table IDs: tbl_d9cdaad7a54d5f045f078c67f646f177068388bae77cc4cac98a4fd784d4f1a8, tbl_0bb395d7d52914af575c5f2001151fe9a13ae237f0a47776212e26167b43b199, tbl_410ca92069eb547a4ec925403319229b1ccfb2e930e994bc6495a892afd2721b, tbl_1c1ea8acc4febff119c96410441dc5c5e8cfed9570c38c66ce70722e54a9dade, tbl_2dfaa8d4824de96a56e1ce96a644f6957d6f19af7024bc8476b076b98ce33f02, tbl_8b99416d9794d944cdea8758154093eb2803a9e76e250b7a386b374536716ca4, tbl_18dc687546d1d290b9312cb0f46546cd342cf8907eab799a1fb19a608a759b7d, tbl_97ea899251b591a2c87178e4f387e8e81b933d8b00403535aa12b34e8ef9e7e9, tbl_2b02b158d7b11fafceb34d04977953b49250cf19e8546eadd9f632c019e6b4ce, tbl_940172b96ceb1c147a5b2ebc71ae79237dd855371ee922c359d5a7d3c06b1770
- Gold table IDs: tbl_1c1ea8acc4febff119c96410441dc5c5e8cfed9570c38c66ce70722e54a9dade
- Missing gold table IDs: (none)
- Eligible documents: 16
- Empty reason: (none)
- Filter counts: company_codes=1466/1466; periods=27650/319; statement_types=7862/16
- Scores and matched tokens: tbl_d9cdaad7a54d5f045f078c67f646f177068388bae77cc4cac98a4fd784d4f1a8=14.066078 [2018, 2019, chi, chính, của, doanh, expenses, financial, phí, thu, tài, vjc, và]; tbl_0bb395d7d52914af575c5f2001151fe9a13ae237f0a47776212e26167b43b199=13.476126 [2018, chi, chính, của, doanh, expenses, financial, phí, thu, tài, vjc, và]; tbl_410ca92069eb547a4ec925403319229b1ccfb2e930e994bc6495a892afd2721b=13.460346 [2018, chi, chính, của, doanh, expenses, financial, phí, thu, tài, vjc, và]; tbl_1c1ea8acc4febff119c96410441dc5c5e8cfed9570c38c66ce70722e54a9dade=13.039650 [2018, 2019, chi, chính, của, doanh, expenses, financial, phí, thu, tài, vjc, và]; tbl_2dfaa8d4824de96a56e1ce96a644f6957d6f19af7024bc8476b076b98ce33f02=12.726083 [2019, chi, chính, doanh, expenses, financial, phí, thu, tài, vjc, và]; tbl_8b99416d9794d944cdea8758154093eb2803a9e76e250b7a386b374536716ca4=12.242965 [2019, chi, chính, doanh, expenses, financial, phí, thu, tài, vjc, và]; tbl_18dc687546d1d290b9312cb0f46546cd342cf8907eab799a1fb19a608a759b7d=6.512435 [2018, chi, doanh, phí, thu, vjc]; tbl_97ea899251b591a2c87178e4f387e8e81b933d8b00403535aa12b34e8ef9e7e9=6.484489 [2018, chi, doanh, phí, thu, vjc]; tbl_2b02b158d7b11fafceb34d04977953b49250cf19e8546eadd9f632c019e6b4ce=5.542876 [2018, doanh, thu, vjc, và]; tbl_940172b96ceb1c147a5b2ebc71ae79237dd855371ee922c359d5a7d3c06b1770=5.532185 [2019, chính, của, tài, vjc, vốn]

### retq_3e6c4ae3ff4d30628320182dce41772a231e92f8cb6d740a476bccdee64f5b06

- Question: So sánh LCTT từ hoạt động kinh doanh của PLX giữa năm 2024 và 2025.
- Intent: compare
- Failure: none
- Predicted table IDs: tbl_42d86b616a29d962d15d0f9b876ce9ef6d6a6c1dc28ac364ec4ea7eb5cee50e0, tbl_2d7185765baf974a3b46ce1a4b088741ad34ee38c6058a3feee345be4ac2fb51, tbl_4b0f982e76c95dcb12eaa0db3a78fd7ff1bfc368736c7900854fbe6cbc7f59f3, tbl_905eb13193083017bcf46d6245b05cc8fee77d83be33d40a4d1deda7c469de30, tbl_0dcfc1c40c58b8bab7ff398204271ab767471b2483577a64b3efc5634bdec29e, tbl_a2723551c1e5df14bee344e97fbbc702123844fd05c2ee7104f1d9fbcb4f0f21, tbl_e778a790326cabb1b92778f9763dc5b421f8825d7c69a5426995f5c9b621ec15, tbl_85d4b68ca940544dfd080492e04421132af82a850a5441de01a812f29d90c477, tbl_e0ee13f6b58a9ed0a451d234925f00be3317b0bc40e8e8e3e60ea92297068b68, tbl_136f5d35e7ce03d6b4a11fb96aa8b3a0f267197a257631755bbcde362b6dc813
- Gold table IDs: tbl_42d86b616a29d962d15d0f9b876ce9ef6d6a6c1dc28ac364ec4ea7eb5cee50e0
- Missing gold table IDs: (none)
- Eligible documents: 31
- Empty reason: (none)
- Filter counts: company_codes=1279/1279; periods=32845/275; statement_types=13712/31
- Scores and matched tokens: tbl_42d86b616a29d962d15d0f9b876ce9ef6d6a6c1dc28ac364ec4ea7eb5cee50e0=8.204067 [2024, 2025, của, doanh, hoạt, kinh, plx, từ, động]; tbl_2d7185765baf974a3b46ce1a4b088741ad34ee38c6058a3feee345be4ac2fb51=7.810892 [2024, 2025, của, doanh, hoạt, kinh, plx, từ, động]; tbl_4b0f982e76c95dcb12eaa0db3a78fd7ff1bfc368736c7900854fbe6cbc7f59f3=7.478374 [2024, của, doanh, hoạt, kinh, plx, từ, động]; tbl_905eb13193083017bcf46d6245b05cc8fee77d83be33d40a4d1deda7c469de30=7.478374 [2024, của, doanh, hoạt, kinh, plx, từ, động]; tbl_0dcfc1c40c58b8bab7ff398204271ab767471b2483577a64b3efc5634bdec29e=7.471109 [2024, 2025, của, hoạt, plx, từ, động]; tbl_a2723551c1e5df14bee344e97fbbc702123844fd05c2ee7104f1d9fbcb4f0f21=7.057762 [2024, của, hoạt, plx, từ, động]; tbl_e778a790326cabb1b92778f9763dc5b421f8825d7c69a5426995f5c9b621ec15=7.057762 [2024, của, hoạt, plx, từ, động]; tbl_85d4b68ca940544dfd080492e04421132af82a850a5441de01a812f29d90c477=6.957601 [2024, 2025, của, hoạt, plx, từ, động]; tbl_e0ee13f6b58a9ed0a451d234925f00be3317b0bc40e8e8e3e60ea92297068b68=6.957047 [2025, hoạt, plx, từ, động]; tbl_136f5d35e7ce03d6b4a11fb96aa8b3a0f267197a257631755bbcde362b6dc813=6.028377 [2024, 2025, doanh, kinh, plx, và]

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

### retq_4af1d194037feec883a2ad351fad2d857c5a714df82de48c323b5b03da976fd4

- Question: So sánh cơ cấu công ty con, công ty liên kết và các khoản đầu tư liên kết của HSG năm 2019.
- Intent: compare
- Failure: partial_gold_hits
- Predicted table IDs: tbl_b4746a668ff55f69408298505571ab066903986fad1b56dd8f3c846093a7d34e, tbl_7f324fa48e13db105a678ddec7f0122ee5128fde4d9ae706d14d43d3869616f9, tbl_e830ced8f2b7ba65fbd036fd3af42143b1126e41edc6c505262efdd903c3eb98, tbl_9744eaf875f087c439ac0e87d3d1457545a0a33ae5c41f86db4dd31a5690cff8, tbl_0c27b7e394cdbd3f23417664dc831a22fa4ede462d8628b27bdf18588a425ce9, tbl_267031f63ffe8e9433f046800e45dd583c86901b72d4f8f6271f8d818104a3ff, tbl_4bd4f6874fa41eb156f28eaa8b412a0b8e35c270fb4e39cd9eabbcf63ee2f41e, tbl_c2b475a1371ca3ff5aa0efcd6805436905662a14992ccd6e36aa01864e6a868c, tbl_d105c6c410b5a22e264baacaaf6242b79a34736e7906c3172b9432409d3d2b7d, tbl_f82cc0e8bfc1201dfce22aa75e4a9255371168e13e629886645128298c683be6
- Gold table IDs: tbl_0c27b7e394cdbd3f23417664dc831a22fa4ede462d8628b27bdf18588a425ce9, tbl_5f59924a66120962e7825d29f63bfd67f40f97052199e33d8a0cf9284a116caf, tbl_c2b475a1371ca3ff5aa0efcd6805436905662a14992ccd6e36aa01864e6a868c
- Missing gold table IDs: tbl_5f59924a66120962e7825d29f63bfd67f40f97052199e33d8a0cf9284a116caf
- Eligible documents: 16
- Empty reason: (none)
- Filter counts: company_codes=1650/1650; periods=15553/142; statement_types=1479/16
- Scores and matched tokens: tbl_b4746a668ff55f69408298505571ab066903986fad1b56dd8f3c846093a7d34e=5.021499 [2019, hsg, kết, năm]; tbl_7f324fa48e13db105a678ddec7f0122ee5128fde4d9ae706d14d43d3869616f9=4.999047 [2019, hsg, kết, năm]; tbl_e830ced8f2b7ba65fbd036fd3af42143b1126e41edc6c505262efdd903c3eb98=4.999047 [2019, hsg, kết, năm]; tbl_9744eaf875f087c439ac0e87d3d1457545a0a33ae5c41f86db4dd31a5690cff8=4.967262 [2019, hsg, kết, năm]; tbl_0c27b7e394cdbd3f23417664dc831a22fa4ede462d8628b27bdf18588a425ce9=4.940592 [2019, hsg, kết, năm]; tbl_267031f63ffe8e9433f046800e45dd583c86901b72d4f8f6271f8d818104a3ff=4.940592 [2019, hsg, kết, năm]; tbl_4bd4f6874fa41eb156f28eaa8b412a0b8e35c270fb4e39cd9eabbcf63ee2f41e=4.940592 [2019, hsg, kết, năm]; tbl_c2b475a1371ca3ff5aa0efcd6805436905662a14992ccd6e36aa01864e6a868c=4.940592 [2019, hsg, kết, năm]; tbl_d105c6c410b5a22e264baacaaf6242b79a34736e7906c3172b9432409d3d2b7d=4.940592 [2019, hsg, kết, năm]; tbl_f82cc0e8bfc1201dfce22aa75e4a9255371168e13e629886645128298c683be6=4.940592 [2019, hsg, kết, năm]

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

### retq_518fddfe44aa2ba59c07c69fba10ccc85837f034cd9bd0f31802eefd97757c7c

- Question: So sánh lưu chuyển tiền từ hoạt động kinh doanh và đầu tư của MSR giữa năm 2024 và 2025.
- Intent: compare
- Failure: none
- Predicted table IDs: tbl_e17a99688107ac7857b1936def9cf5ab7c70a676095ef03aa3641f85d24eb207, tbl_8852e42a5aeef5f45767513b1cdbf87599dbe6c761bdc00934edec88c0442dfb, tbl_f984149a403b4b41606e7cfeb50060e932516ed20427ceb1a082c12edd1bbc53, tbl_c83e7e4cd2c9b68ec97ed05fe6bfd50739d2556645b8200e710f514da596a552, tbl_59ebadb8387f29d0d4577386a6c958df499b8b7e6cc480860fea0c95e6cab085, tbl_f685410404356622bfa32aaa2d727bea425e9b65d48cf9761fa2e0c449d57c44, tbl_4d95d5bbcbc8743cedf33cd0791a598b7c143e138727b6228bfca3f4e12ffd29, tbl_6d40c2555d152c805cebc81c21a9e406d08109e1e0b3f100754b8fd709603c84, tbl_47ab0571a657d61466051691e0bad744baf4c8c9d515f80023998e9ddd2ccd4b, tbl_791cd23dc08930885065384ae2b30fa71d9085123cc0bc0375e8c71a9829f3b4
- Gold table IDs: tbl_e17a99688107ac7857b1936def9cf5ab7c70a676095ef03aa3641f85d24eb207
- Missing gold table IDs: (none)
- Eligible documents: 27
- Empty reason: (none)
- Filter counts: company_codes=1384/1384; periods=32845/257; statement_types=13712/27
- Scores and matched tokens: tbl_e17a99688107ac7857b1936def9cf5ab7c70a676095ef03aa3641f85d24eb207=15.785539 [2024, 2025, chuyển, của, doanh, hoạt, kinh, lưu, msr, tiền, tư, từ, đầu, động]; tbl_8852e42a5aeef5f45767513b1cdbf87599dbe6c761bdc00934edec88c0442dfb=15.302201 [2025, chuyển, của, doanh, hoạt, kinh, lưu, msr, tiền, tư, từ, đầu, động]; tbl_f984149a403b4b41606e7cfeb50060e932516ed20427ceb1a082c12edd1bbc53=15.085352 [2024, chuyển, của, doanh, hoạt, kinh, lưu, msr, tiền, tư, từ, đầu, động]; tbl_c83e7e4cd2c9b68ec97ed05fe6bfd50739d2556645b8200e710f514da596a552=11.652162 [2024, chuyển, của, doanh, hoạt, lưu, msr, tiền, tư, từ, đầu, động]; tbl_59ebadb8387f29d0d4577386a6c958df499b8b7e6cc480860fea0c95e6cab085=10.605855 [2024, 2025, chuyển, của, hoạt, lưu, msr, tiền, từ, động]; tbl_f685410404356622bfa32aaa2d727bea425e9b65d48cf9761fa2e0c449d57c44=10.605855 [2024, 2025, chuyển, của, hoạt, lưu, msr, tiền, từ, động]; tbl_4d95d5bbcbc8743cedf33cd0791a598b7c143e138727b6228bfca3f4e12ffd29=9.786339 [2024, chuyển, của, hoạt, lưu, msr, tiền, từ, động]; tbl_6d40c2555d152c805cebc81c21a9e406d08109e1e0b3f100754b8fd709603c84=9.786339 [2024, chuyển, của, hoạt, lưu, msr, tiền, từ, động]; tbl_47ab0571a657d61466051691e0bad744baf4c8c9d515f80023998e9ddd2ccd4b=5.596674 [2024, msr, tư, đầu, động]; tbl_791cd23dc08930885065384ae2b30fa71d9085123cc0bc0375e8c71a9829f3b4=4.268738 [2024, 2025, doanh, msr]

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

### retq_5bc82c57026f9a9c465c6de9a5423aa4dcc47c3bcb909980e2a8b34312c723e5

- Question: Tính biến động tỷ lệ sở hữu tại các nhóm công ty con chủ chốt của MSN trong năm 2019.
- Intent: growth
- Failure: partial_gold_hits
- Predicted table IDs: tbl_a41efffc23b838943088bfa105bfd935b8587afc6212a9d635822bb44c9cadb2, tbl_e5e3e0bec5d0817623caa23d1605bc09a18c05a9f2f143ca477f03f07eeafd90, tbl_29918745c5d49e496fad0238605e40e1a8f113ebad7cfd97a96a6450217210be, tbl_94aed7cc0d3eba60a4fd6088c94ffc2d48eeea21a2aec0edba71cd47a25abf16, tbl_1da562d2f881f6961c036f97219e00c615a517bf5bc79a3af85ef64009f16bd8, tbl_30c45beb5ae410b12ece4be9022a0472557e5440698403d64c8ed048f840bdad, tbl_46bf1fc9965efefef41680e93bb52542f2017ad580ee7488afd2f36a532add32, tbl_8288bad492d25c1b22691f870b3bc1ad9d9d5427e06083ddc0136a7917cfb8f8, tbl_9c9d1bb0b9374cfd1a0c9ead1f989d11d23c6921009cfce54f22f984861371ad, tbl_bab88f9d1a554509454e97c234bb653aa223e639d9b62f52ca8a2dd21ca0e73d
- Gold table IDs: tbl_1da562d2f881f6961c036f97219e00c615a517bf5bc79a3af85ef64009f16bd8, tbl_30c45beb5ae410b12ece4be9022a0472557e5440698403d64c8ed048f840bdad, tbl_9c9d1bb0b9374cfd1a0c9ead1f989d11d23c6921009cfce54f22f984861371ad, tbl_d1a533562e37f19983730375b041745f25652491d0544a9e5bc8fe6acf8085c1
- Missing gold table IDs: tbl_d1a533562e37f19983730375b041745f25652491d0544a9e5bc8fe6acf8085c1
- Eligible documents: 17
- Empty reason: (none)
- Filter counts: company_codes=1502/1502; periods=15553/148; statement_types=1479/17
- Scores and matched tokens: tbl_a41efffc23b838943088bfa105bfd935b8587afc6212a9d635822bb44c9cadb2=4.008796 [2019, msn, năm]; tbl_e5e3e0bec5d0817623caa23d1605bc09a18c05a9f2f143ca477f03f07eeafd90=4.008796 [2019, msn, năm]; tbl_29918745c5d49e496fad0238605e40e1a8f113ebad7cfd97a96a6450217210be=3.987905 [2019, msn, năm]; tbl_94aed7cc0d3eba60a4fd6088c94ffc2d48eeea21a2aec0edba71cd47a25abf16=3.987905 [2019, msn, năm]; tbl_1da562d2f881f6961c036f97219e00c615a517bf5bc79a3af85ef64009f16bd8=3.967237 [2019, msn, năm]; tbl_30c45beb5ae410b12ece4be9022a0472557e5440698403d64c8ed048f840bdad=3.967237 [2019, msn, năm]; tbl_46bf1fc9965efefef41680e93bb52542f2017ad580ee7488afd2f36a532add32=3.967237 [2019, msn, năm]; tbl_8288bad492d25c1b22691f870b3bc1ad9d9d5427e06083ddc0136a7917cfb8f8=3.967237 [2019, msn, năm]; tbl_9c9d1bb0b9374cfd1a0c9ead1f989d11d23c6921009cfce54f22f984861371ad=3.967237 [2019, msn, năm]; tbl_bab88f9d1a554509454e97c234bb653aa223e639d9b62f52ca8a2dd21ca0e73d=3.967237 [2019, msn, năm]

### retq_5e5a7482abba820a4e8819833d85d90cbb45fcd33bd23e5214a29f9f4e861f5d

- Question: Tính tăng trưởng LNST chưa phân phối của MBS từ cuối năm 2024 đến cuối năm 2025.
- Intent: growth
- Failure: none
- Predicted table IDs: tbl_26e3659b2fcfeb6b902514126b37f1b78e1225cb068c73714f2d43bc02ad6855, tbl_27de9916b0b57c635f3fafe361c0c25b6231348b9789b936ad72a92c76a4b636, tbl_ca7f7621474df65ae98c21170d5e5e76a362879df12558da83957bdd109cd66f, tbl_2876293c695efac9ae4dcb5a8dc03804a249dc13a955016c26682a7ef9175a8e, tbl_1e350035cbb945d683acb85e3329a854d0ce8a4db888a3768ff30ba5c0e2f56e, tbl_bd1902e27b93cebbe68e673ad3c0961d8ddf3c64934ed96b7c7d55dbe4ae0a70, tbl_4a78fa9fbad9d09da85bb936f6adc8014c029ea794dbc347c992b9dec3813a49
- Gold table IDs: tbl_26e3659b2fcfeb6b902514126b37f1b78e1225cb068c73714f2d43bc02ad6855
- Missing gold table IDs: (none)
- Eligible documents: 7
- Empty reason: (none)
- Filter counts: company_codes=994/994; periods=32845/224; statement_types=7862/7
- Scores and matched tokens: tbl_26e3659b2fcfeb6b902514126b37f1b78e1225cb068c73714f2d43bc02ad6855=10.334927 [2024, 2025, chưa, của, earnings, mbs, phân, phối, retained]; tbl_27de9916b0b57c635f3fafe361c0c25b6231348b9789b936ad72a92c76a4b636=9.650703 [2024, chưa, của, earnings, mbs, phân, phối, retained]; tbl_ca7f7621474df65ae98c21170d5e5e76a362879df12558da83957bdd109cd66f=4.503851 [2024, mbs, tăng]; tbl_2876293c695efac9ae4dcb5a8dc03804a249dc13a955016c26682a7ef9175a8e=4.338051 [2025, mbs, tăng]; tbl_1e350035cbb945d683acb85e3329a854d0ce8a4db888a3768ff30ba5c0e2f56e=3.939799 [2024, 2025, mbs, năm]; tbl_bd1902e27b93cebbe68e673ad3c0961d8ddf3c64934ed96b7c7d55dbe4ae0a70=3.859154 [2024, mbs, năm]; tbl_4a78fa9fbad9d09da85bb936f6adc8014c029ea794dbc347c992b9dec3813a49=3.856970 [2025, mbs, năm]

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

### retq_70fa3ba240c02f67e25d7f74451a303b71bdab96a0a660807d3f1d48621e2318

- Question: Tính tăng trưởng kết quả hoạt động theo lĩnh vực kinh doanh của BVH từ năm 2020 đến 2021.
- Intent: growth
- Failure: none
- Predicted table IDs: tbl_500627e346f8d3a40e31814980415615b891d37206c8606ec27313fb6104f736, tbl_0363cc5b912a7156ac7b5ce05c5d40f9b443d176f936a8602164e25f6cc98bd1, tbl_4573bd7421ac2a781e78e44bbb4dcc1a3fa208e33593653b291b27adc959f0c5, tbl_c2b47b1ebffad3d654a96cdc28d4c9001fadc37486bf22db350827619ea075e6, tbl_7f278ec3fe1a1567ea57e84a6bb9361210639b8016bd835c3b0fe93f2555aee4, tbl_6aaf6bbc2179d9b10452a4b55cda8778f58e3f22d89bfca994c4d4002a0847ba, tbl_37b9c6fb241f58069ed3a46a619c937ee05279ba690799307e1d9da495d5a7c9, tbl_b7894eff30a09f78b9e44719aa90ddf05acc958cc4c19f3134bbbe7c934c0357, tbl_4427b6aa03e3c53fdd4703afea9547c48506905d28ff3c7faeeb0b29afe655ca, tbl_882d2788c29fe3ced2bd91c7b8d322039d58ef875dfee284fba6bae873a152d7
- Gold table IDs: tbl_0363cc5b912a7156ac7b5ce05c5d40f9b443d176f936a8602164e25f6cc98bd1, tbl_500627e346f8d3a40e31814980415615b891d37206c8606ec27313fb6104f736
- Missing gold table IDs: (none)
- Eligible documents: 13
- Empty reason: (none)
- Filter counts: company_codes=1956/1956; periods=17746/172; statement_types=7862/13
- Scores and matched tokens: tbl_500627e346f8d3a40e31814980415615b891d37206c8606ec27313fb6104f736=15.065587 [2021, bvh, của, doanh, hoạt, kinh, kết, lĩnh, năm, quả, theo, vực, động]; tbl_0363cc5b912a7156ac7b5ce05c5d40f9b443d176f936a8602164e25f6cc98bd1=15.051662 [2020, 2021, bvh, của, doanh, hoạt, kinh, kết, lĩnh, năm, quả, theo, vực, động]; tbl_4573bd7421ac2a781e78e44bbb4dcc1a3fa208e33593653b291b27adc959f0c5=4.890581 [2021, bvh, doanh, hoạt, động]; tbl_c2b47b1ebffad3d654a96cdc28d4c9001fadc37486bf22db350827619ea075e6=3.455157 [2021, bvh, tính]; tbl_7f278ec3fe1a1567ea57e84a6bb9361210639b8016bd835c3b0fe93f2555aee4=3.415250 [2021, bvh, tính]; tbl_6aaf6bbc2179d9b10452a4b55cda8778f58e3f22d89bfca994c4d4002a0847ba=3.035697 [2021, bvh, của]; tbl_37b9c6fb241f58069ed3a46a619c937ee05279ba690799307e1d9da495d5a7c9=2.846454 [2021, bvh]; tbl_b7894eff30a09f78b9e44719aa90ddf05acc958cc4c19f3134bbbe7c934c0357=2.846454 [2021, bvh]; tbl_4427b6aa03e3c53fdd4703afea9547c48506905d28ff3c7faeeb0b29afe655ca=2.730401 [2021, bvh]; tbl_882d2788c29fe3ced2bd91c7b8d322039d58ef875dfee284fba6bae873a152d7=2.730401 [2021, bvh]

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

### retq_7a23e39cc00a015db310ae9865241133d6c2bebebb760ab4a9b0341c96de6628

- Question: Tra cứu các bảng thuyết minh biến động TSCĐ hữu hình, TSCĐ vô hình và chi phí trả trước của PLX năm 2018.
- Intent: lookup
- Failure: none
- Predicted table IDs: tbl_a532aa0cc496fbb2e5813cea0dfef690bcdac0cc6a976cd32675e110113bd255, tbl_7142232bdfa1b465cacd949ff4e52578ab0548b4c953584675ca2874c666d0fb, tbl_87e423b11644b28868ab5e6430600e02a0e4846b3737fb08d6b9079bb49087e0, tbl_de69236b99be4486d86172fe7789038978b5ff0b7352e120aec6fd0e805f771f
- Gold table IDs: tbl_7142232bdfa1b465cacd949ff4e52578ab0548b4c953584675ca2874c666d0fb, tbl_a532aa0cc496fbb2e5813cea0dfef690bcdac0cc6a976cd32675e110113bd255, tbl_de69236b99be4486d86172fe7789038978b5ff0b7352e120aec6fd0e805f771f
- Missing gold table IDs: (none)
- Eligible documents: 4
- Empty reason: (none)
- Filter counts: company_codes=1279/1279; periods=14691/121; statement_types=1479/4
- Scores and matched tokens: tbl_a532aa0cc496fbb2e5813cea0dfef690bcdac0cc6a976cd32675e110113bd255=20.413765 [2018, assets, equipment, hình, hữu, intangible, minh, năm, plant, plx, property, thuyết, tscđ, vô]; tbl_7142232bdfa1b465cacd949ff4e52578ab0548b4c953584675ca2874c666d0fb=6.983835 [2018, minh, năm, plx, thuyết]; tbl_87e423b11644b28868ab5e6430600e02a0e4846b3737fb08d6b9079bb49087e0=6.983835 [2018, minh, năm, plx, thuyết]; tbl_de69236b99be4486d86172fe7789038978b5ff0b7352e120aec6fd0e805f771f=6.983835 [2018, minh, năm, plx, thuyết]

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

### retq_7d8b6d321910394b5f0eaeeb22a747545e1d297ae6f08f8a3f9243c11c557125

- Question: Tính tăng trưởng lưu chuyển tiền thuần từ hoạt động kinh doanh của SJG từ năm 2019 đến 2020.
- Intent: growth
- Failure: none
- Predicted table IDs: tbl_4c4f383342c4070a31ba76f7a3a5e659ac499e642453a3ed65d904b3682e16dc, tbl_7dcce7007e9fd10f1645c88b97f368ff59a1fc5c0ed00c8d491260393f17c629, tbl_a401bf4d50732629ff33a26ee916137181289f1c3eae9213ff634e0782ed4475, tbl_6b12edf08e6646a7f775375f38db867153a1c84b2fd559387ae6beb90d2b5158, tbl_d0328d5a2765aa2a646474c978fb2c99280bead991260925a8ce306357df94ed, tbl_8439bbdd7ca14e22731ca349cd665ae80c32cfbcb7ebc704a6b2dd7ebf1da8fb, tbl_de5f80e58645414a8dad886f078f6da9b877935e59611615369a46a0f42bb55b, tbl_7197881ae9b87b558a9d9b5dfc3fd0bfc61b18ddde4e77c380d6871084915512, tbl_ee7c8ec772c8ab7b7a27a211958c478bc17628112b0b0af0d57b7f78fedfc57b, tbl_b4a4d84562d80fd144c486ea95026e28f9727693bd8e0417bdd7a8e0bf142196
- Gold table IDs: tbl_7dcce7007e9fd10f1645c88b97f368ff59a1fc5c0ed00c8d491260393f17c629
- Missing gold table IDs: (none)
- Eligible documents: 22
- Empty reason: (none)
- Filter counts: company_codes=1056/1056; periods=29768/333; statement_types=13712/22
- Scores and matched tokens: tbl_4c4f383342c4070a31ba76f7a3a5e659ac499e642453a3ed65d904b3682e16dc=21.085560 [2019, cash, chuyển, doanh, flow, hoạt, kinh, lưu, operating, sjg, thuần, tiền, tính, từ, động]; tbl_7dcce7007e9fd10f1645c88b97f368ff59a1fc5c0ed00c8d491260393f17c629=21.055084 [2019, 2020, cash, chuyển, doanh, flow, hoạt, kinh, lưu, operating, sjg, thuần, tiền, tính, từ, động]; tbl_a401bf4d50732629ff33a26ee916137181289f1c3eae9213ff634e0782ed4475=21.048992 [2020, cash, chuyển, doanh, flow, hoạt, kinh, lưu, operating, sjg, thuần, tiền, tính, từ, động]; tbl_6b12edf08e6646a7f775375f38db867153a1c84b2fd559387ae6beb90d2b5158=20.618132 [2020, cash, chuyển, doanh, flow, hoạt, kinh, lưu, năm, operating, sjg, thuần, tiền, từ, động]; tbl_d0328d5a2765aa2a646474c978fb2c99280bead991260925a8ce306357df94ed=20.312199 [2020, cash, chuyển, doanh, flow, hoạt, kinh, lưu, operating, sjg, thuần, tiền, tính, từ, động]; tbl_8439bbdd7ca14e22731ca349cd665ae80c32cfbcb7ebc704a6b2dd7ebf1da8fb=15.543224 [2019, 2020, cash, chuyển, flow, hoạt, lưu, sjg, thuần, tiền, từ, động]; tbl_de5f80e58645414a8dad886f078f6da9b877935e59611615369a46a0f42bb55b=15.543224 [2019, 2020, cash, chuyển, flow, hoạt, lưu, sjg, thuần, tiền, từ, động]; tbl_7197881ae9b87b558a9d9b5dfc3fd0bfc61b18ddde4e77c380d6871084915512=14.768373 [2019, cash, chuyển, flow, hoạt, lưu, sjg, thuần, tiền, từ, động]; tbl_ee7c8ec772c8ab7b7a27a211958c478bc17628112b0b0af0d57b7f78fedfc57b=14.768373 [2019, cash, chuyển, flow, hoạt, lưu, sjg, thuần, tiền, từ, động]; tbl_b4a4d84562d80fd144c486ea95026e28f9727693bd8e0417bdd7a8e0bf142196=14.730074 [2020, cash, chuyển, flow, hoạt, lưu, sjg, thuần, tiền, từ, động]

### retq_843c4f4d5dbc6284787d601a04e41c8bd5e98f2374d96a5432d69ad85daedbe8

- Question: Tra cứu TSCĐ, tài sản vô hình và đầu tư dài hạn của DNH cuối năm 2021.
- Intent: lookup
- Failure: none
- Predicted table IDs: tbl_8753fe06e2aa42744da649f13763a363f61d332106e4aff0e74112faedd67536, tbl_02a8737a518d1e93fa9a49317f20cace6c294fda30ee3a96cd9c25979a65138a, tbl_6891a61c0715c3be5be343454830f253e506a9961661eef4abf35c1418094e3c, tbl_eb8ff0db92c14affcc0d5ab53d1b6f4fdec23fd667227980928caf97f5fdd7ac
- Gold table IDs: tbl_8753fe06e2aa42744da649f13763a363f61d332106e4aff0e74112faedd67536
- Missing gold table IDs: (none)
- Eligible documents: 4
- Empty reason: (none)
- Filter counts: company_codes=1002/1002; periods=17746/141; statement_types=7745/4
- Scores and matched tokens: tbl_8753fe06e2aa42744da649f13763a363f61d332106e4aff0e74112faedd67536=15.746036 [2021, assets, của, dnh, dài, fixed, hình, hạn, intangible, năm, sản, tài, tư, vô, đầu]; tbl_02a8737a518d1e93fa9a49317f20cace6c294fda30ee3a96cd9c25979a65138a=15.484982 [2021, assets, của, dnh, dài, fixed, hình, hạn, intangible, năm, sản, tài, tư, vô, đầu]; tbl_6891a61c0715c3be5be343454830f253e506a9961661eef4abf35c1418094e3c=7.765837 [2021, của, dnh, hạn, năm, tài, tư, và, đầu]; tbl_eb8ff0db92c14affcc0d5ab53d1b6f4fdec23fd667227980928caf97f5fdd7ac=7.765837 [2021, của, dnh, hạn, năm, tài, tư, và, đầu]

### retq_859f3db5be0b80c43b04417d2369df915845fd9cfec70503dbe753f23ea43fe6

- Question: So sánh lưu chuyển tiền từ hoạt động tài chính của HHV giữa năm 2019 và 2020.
- Intent: compare
- Failure: none
- Predicted table IDs: tbl_2786e6e2a51f20e539cfe0593bd7048822d88c80c38bae4d9b3552ab6aa057d4, tbl_682680b26d0234dc9b125ada987295046d9ab6a0d9046ddb0d13633e86199173, tbl_3ab171fbf4f843538e0073a7a7922571ea5a79793eb05ee2076571eb1760ccec, tbl_0f377915e70efa05f6a2456b8c340e9aae06e5d7612c4453661b1ce148db4e71, tbl_2b687da25f30836dc2572aaf651dd5498e541fde5369c367acdb2bd2e86a02b7, tbl_8844461237d04a17baec78eca1c9992fb0be2e34701f488e62d8a911d0928a81, tbl_64dfc9f5208831ee0bb46c562eee5b5993623e72208d2acb98fca5546cea4f38, tbl_c7e29a9a4657e7db49564818b21cd1419a4593d928a12d585d29d9b985ac1042, tbl_2346e199daca18e8699ed372a34567f8c7bb7df27c005b51d73027039589af94, tbl_56826d6151dc9b94081f26ae1a48a2542760312b2ba9ce80e71244a933b6dbb9
- Gold table IDs: tbl_2b687da25f30836dc2572aaf651dd5498e541fde5369c367acdb2bd2e86a02b7
- Missing gold table IDs: (none)
- Eligible documents: 38
- Empty reason: (none)
- Filter counts: company_codes=1181/1181; periods=29768/303; statement_types=13712/38
- Scores and matched tokens: tbl_2786e6e2a51f20e539cfe0593bd7048822d88c80c38bae4d9b3552ab6aa057d4=13.216446 [2019, 2020, chuyển, hhv, hoạt, lưu, tiền, từ, động]; tbl_682680b26d0234dc9b125ada987295046d9ab6a0d9046ddb0d13633e86199173=13.216446 [2019, 2020, chuyển, hhv, hoạt, lưu, tiền, từ, động]; tbl_3ab171fbf4f843538e0073a7a7922571ea5a79793eb05ee2076571eb1760ccec=13.087027 [2019, 2020, chuyển, chính, hhv, hoạt, lưu, tiền, tài, từ, và, động]; tbl_0f377915e70efa05f6a2456b8c340e9aae06e5d7612c4453661b1ce148db4e71=12.870530 [2019, chuyển, hhv, hoạt, lưu, tiền, từ, động]; tbl_2b687da25f30836dc2572aaf651dd5498e541fde5369c367acdb2bd2e86a02b7=12.865372 [2019, 2020, chuyển, chính, của, hhv, hoạt, lưu, tiền, tài, từ, và, động]; tbl_8844461237d04a17baec78eca1c9992fb0be2e34701f488e62d8a911d0928a81=12.786301 [2019, chuyển, chính, hhv, hoạt, lưu, tiền, tài, từ, và, động]; tbl_64dfc9f5208831ee0bb46c562eee5b5993623e72208d2acb98fca5546cea4f38=12.564855 [2019, chuyển, hhv, hoạt, lưu, tiền, từ, động]; tbl_c7e29a9a4657e7db49564818b21cd1419a4593d928a12d585d29d9b985ac1042=12.532647 [2020, chuyển, hhv, hoạt, lưu, tiền, từ, động]; tbl_2346e199daca18e8699ed372a34567f8c7bb7df27c005b51d73027039589af94=12.404918 [2019, chuyển, chính, hhv, hoạt, lưu, tiền, tài, từ, và, động]; tbl_56826d6151dc9b94081f26ae1a48a2542760312b2ba9ce80e71244a933b6dbb9=10.779645 [2020, chuyển, chính, hhv, hoạt, lưu, tiền, tài, từ, động]

### retq_865b76cc9a17d3afd764551b50a9de5b68384b62c2366a2273804e80d1d999a1

- Question: So sánh doanh thu thuần và lợi nhuận sau thuế trên báo cáo riêng với báo cáo hợp nhất của MML năm 2017.
- Intent: compare
- Failure: none
- Predicted table IDs: tbl_5a90d02a5bf9201c29ce5f1740082810de9f94846459ec4f4435dc839a749b27, tbl_5d21772104dad21f5eca0c72d316345781d87f43e3149c82b93183540f08521b, tbl_4998d6e6580d53da760fb0612a85f08d421ec0bf0ba6eb3128a9fc781f93211f, tbl_a4dec3c29d390aaf368acdbedb040a9cd97b0e1dc545c146bdcc7ba599c6fdc1, tbl_510a5f4e98ad927193abe8651a4d9e5884a6a0d9172628c73877a3626995e59f, tbl_2c265aeb95145be44c6ee3e3e1aa2bb616f0d8dc64c72e4f0480c5c077a97fad, tbl_3d6bc5471b5a1af7886e4c404c4c612f1c95b17edf356e706fe2ddeda5b4b20f, tbl_439f68f95ad8fcfbcb3addd9fb64b38a4608100789bddf708b897775878e3b87, tbl_9123ebde93bec6e5902b7ef5a5f75199f39e740e6d187d0f80a742c93c008b9f, tbl_63b202d2110eb2be2fb03b22cf9225401b5f555fea76d0b89dc78857f5dcba57
- Gold table IDs: tbl_2c265aeb95145be44c6ee3e3e1aa2bb616f0d8dc64c72e4f0480c5c077a97fad, tbl_3d6bc5471b5a1af7886e4c404c4c612f1c95b17edf356e706fe2ddeda5b4b20f
- Missing gold table IDs: (none)
- Eligible documents: 12
- Empty reason: (none)
- Filter counts: company_codes=1009/1009; periods=14081/124; statement_types=7862/12
- Scores and matched tokens: tbl_5a90d02a5bf9201c29ce5f1740082810de9f94846459ec4f4435dc839a749b27=10.593638 [2017, doanh, mml, net, revenue, thu, thuần]; tbl_5d21772104dad21f5eca0c72d316345781d87f43e3149c82b93183540f08521b=10.593638 [2017, doanh, mml, net, revenue, thu, thuần]; tbl_4998d6e6580d53da760fb0612a85f08d421ec0bf0ba6eb3128a9fc781f93211f=8.836980 [2017, báo, cáo, doanh, hợp, mml, nhất]; tbl_a4dec3c29d390aaf368acdbedb040a9cd97b0e1dc545c146bdcc7ba599c6fdc1=8.836980 [2017, báo, cáo, doanh, hợp, mml, nhất]; tbl_510a5f4e98ad927193abe8651a4d9e5884a6a0d9172628c73877a3626995e59f=8.118380 [2017, của, doanh, mml, năm, revenue, tax, thu, thuế]; tbl_2c265aeb95145be44c6ee3e3e1aa2bb616f0d8dc64c72e4f0480c5c077a97fad=7.830904 [2017, của, doanh, mml, năm, revenue, tax, thu, thuế]; tbl_3d6bc5471b5a1af7886e4c404c4c612f1c95b17edf356e706fe2ddeda5b4b20f=7.601128 [2017, của, doanh, mml, năm, tax, thu, thuế]; tbl_439f68f95ad8fcfbcb3addd9fb64b38a4608100789bddf708b897775878e3b87=7.497606 [2017, của, lợi, mml, nhuận, năm, sau, thuế]; tbl_9123ebde93bec6e5902b7ef5a5f75199f39e740e6d187d0f80a742c93c008b9f=6.424598 [2017, của, lợi, mml, năm, trên]; tbl_63b202d2110eb2be2fb03b22cf9225401b5f555fea76d0b89dc78857f5dcba57=6.079493 [2017, của, lợi, mml, năm, trên]

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

### retq_9024e98663eb4d6d935ed2bd5c014bdeff9ba0d94a7e62038833deb6ff1b464a

- Question: Tính tăng trưởng doanh thu bán hàng và cung cấp dịch vụ của FIT từ năm 2024 đến 2025.
- Intent: growth
- Failure: none
- Predicted table IDs: tbl_bd98bfd7439ef6111116fe4390189950b1aff86465ebd2d2a9d467fe01acde31, tbl_52a74667cce7621c6ec505ad5f97dd053d6dec2d042f7cf754710ef514faa48c, tbl_bb10624e01297b3207cf628b3eec06624b5190018446a5c4068a35e9cdc304ac, tbl_02640840347e9a2b67c1414be981f2f145aab122ca351b17b91071b196be3bd5, tbl_f799743fc89c14554f18792b5e35895f2204b7d6cd276368f1fa77ba1726b9d9, tbl_5834fbf6cdb01460a5bac43c8ef0075f829e8dec1c573f8639658e45d106295e
- Gold table IDs: tbl_bd98bfd7439ef6111116fe4390189950b1aff86465ebd2d2a9d467fe01acde31
- Missing gold table IDs: (none)
- Eligible documents: 6
- Empty reason: (none)
- Filter counts: company_codes=1319/1319; periods=32845/281; statement_types=7862/6
- Scores and matched tokens: tbl_bd98bfd7439ef6111116fe4390189950b1aff86465ebd2d2a9d467fe01acde31=13.777048 [2024, 2025, bán, cung, cấp, doanh, dịch, fit, hàng, thu, và, vụ]; tbl_52a74667cce7621c6ec505ad5f97dd053d6dec2d042f7cf754710ef514faa48c=12.866572 [2024, bán, cung, cấp, doanh, dịch, fit, hàng, thu, và, vụ]; tbl_bb10624e01297b3207cf628b3eec06624b5190018446a5c4068a35e9cdc304ac=4.505755 [2025, của, doanh, fit, thu]; tbl_02640840347e9a2b67c1414be981f2f145aab122ca351b17b91071b196be3bd5=3.043317 [2025, của, fit]; tbl_f799743fc89c14554f18792b5e35895f2204b7d6cd276368f1fa77ba1726b9d9=2.855229 [2025, của, fit]; tbl_5834fbf6cdb01460a5bac43c8ef0075f829e8dec1c573f8639658e45d106295e=2.763450 [2024, của, fit]

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

### retq_940e4706d5eea84a3bc3a90d109ddc7298d497fb90cbf4899fb043edb2268064

- Question: Tính tăng trưởng DT thuần và lợi nhuận sau thuế riêng của HPG từ năm 2023 đến 2024.
- Intent: growth
- Failure: none
- Predicted table IDs: tbl_ebf044b5d74111ba642297bd34835a38a2e61cd6f3b5ba35454eb1a1e6dd3662, tbl_10d616af10c4c2b269850189c4747d4b021bdfb2eae59ffa70b4b1d36336203f, tbl_0709c3a49537075eb867362e87fa7b645a731909c49a06e297ce3478a1129e65, tbl_a882fa7d7865e7f149f8c20b7db4ddc6b79b77a4c309521ba76ef959b9f567ad, tbl_64bcd1cf9772cf24111053804d5874a693ffd19ff05524bd57befdf1a1598c71, tbl_dc1f37752da0c492d024fca36863cfaaa8bef71855b685d91162c30a3f40004c, tbl_da881c3648f1e1f6b08f7472115783be88e9c6897599150610dced059bc714a0, tbl_e194d2067d5f70ff7515171334d20eb1018e44934ca676699c5d144d94b2b9e2, tbl_561934584b646691bfd7e4b23f676d050e3dcb57eac15312d4ec7a6cff1c73b9, tbl_3bdfa9668eb8ef820bd6521c9bef5975cf71a20ef90c67beb1612a1ce3ab2f50
- Gold table IDs: tbl_3bdfa9668eb8ef820bd6521c9bef5975cf71a20ef90c67beb1612a1ce3ab2f50
- Missing gold table IDs: (none)
- Eligible documents: 18
- Empty reason: (none)
- Filter counts: company_codes=1299/1299; periods=35143/262; statement_types=7862/18
- Scores and matched tokens: tbl_ebf044b5d74111ba642297bd34835a38a2e61cd6f3b5ba35454eb1a1e6dd3662=11.264411 [2024, hpg, lợi, nhuận, năm, profit, sau, tax, thuế, tính]; tbl_10d616af10c4c2b269850189c4747d4b021bdfb2eae59ffa70b4b1d36336203f=9.120173 [2023, 2024, của, hpg, lợi, nhuận, năm, sau, thuế]; tbl_0709c3a49537075eb867362e87fa7b645a731909c49a06e297ce3478a1129e65=7.500430 [2024, của, hpg, lợi, nhuận, sau, thuế]; tbl_a882fa7d7865e7f149f8c20b7db4ddc6b79b77a4c309521ba76ef959b9f567ad=7.220186 [2023, của, hpg, lợi, nhuận, sau, thuế]; tbl_64bcd1cf9772cf24111053804d5874a693ffd19ff05524bd57befdf1a1598c71=6.875414 [2024, của, hpg, lợi, nhuận, sau, thuế]; tbl_dc1f37752da0c492d024fca36863cfaaa8bef71855b685d91162c30a3f40004c=6.735204 [2023, của, hpg, lợi, nhuận, sau, thuế]; tbl_da881c3648f1e1f6b08f7472115783be88e9c6897599150610dced059bc714a0=5.418602 [2023, 2024, hpg, tax, thuế]; tbl_e194d2067d5f70ff7515171334d20eb1018e44934ca676699c5d144d94b2b9e2=5.316854 [2023, 2024, hpg, thuần]; tbl_561934584b646691bfd7e4b23f676d050e3dcb57eac15312d4ec7a6cff1c73b9=5.228072 [2023, 2024, của, hpg, tax, thuế, và]; tbl_3bdfa9668eb8ef820bd6521c9bef5975cf71a20ef90c67beb1612a1ce3ab2f50=5.047141 [2023, 2024, của, hpg, tax, thuế, và]

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

### retq_9a05b36a4b132789fe7ddd37f32f3ea8f6d14b2f51bef2ca73991c166adfa1d5

- Question: Đối chiếu tỷ lệ sở hữu và giá trị đầu tư tại các công ty liên kết của VSF năm 2019.
- Intent: compare
- Failure: zero_gold_hits
- Predicted table IDs: tbl_4be1f97d8c84732baa4e4cc5389388c8a34e99f2c9d8878baa26456174fd7845, tbl_916e9082786974a0cfe1315f678c30a3b06f17be4607b5f43e76cf80d96c4389, tbl_56cbcb8ac1a87eec41efff9d983110645468e67b3521d3c1d36d6d76a222973d, tbl_3af54f7053fa6b59c4fc7d459a42e765f6e463a11b6a33bafa26c32c5262b02f, tbl_4809a2aae1ef555902270dfc380be3c7d0983a97b2e17fd9404678ff611bbf16, tbl_21a88164c203323e200236ecce1691c5cf2ab7334d5282cc33ccd147e83cae39, tbl_57cdf5d3869f9f30d68912f3125b1c7d03755d7087da72e9e079761a16437049, tbl_5f2528a814e3613568a1803c666a3e2a85127ad0c49084eaa076ee7629137f12, tbl_86cdac87d86f4ee3767f27db3299d81db237acfd1d9a638b223e20cc8c44ca75, tbl_0a5f2c4e9fbbedc73d8cce9347158eee39cf32a28064d535a67b11bd79ad9df0
- Gold table IDs: tbl_14356d221adccea2fca9acbeee9d317b6f47f366293dac5df7af6e949304a3a7, tbl_7178f397f374720112e3934ca2ab11c3be497c59c51d32701d91706486988430, tbl_b2157506dc11a7cd6a1f3a6e30090d1f696a96da3c3325119b4489a2adb7f0d1
- Missing gold table IDs: tbl_14356d221adccea2fca9acbeee9d317b6f47f366293dac5df7af6e949304a3a7, tbl_7178f397f374720112e3934ca2ab11c3be497c59c51d32701d91706486988430, tbl_b2157506dc11a7cd6a1f3a6e30090d1f696a96da3c3325119b4489a2adb7f0d1
- Eligible documents: 24
- Empty reason: (none)
- Filter counts: company_codes=1194/1194; periods=15553/183; statement_types=1479/24
- Scores and matched tokens: tbl_4be1f97d8c84732baa4e4cc5389388c8a34e99f2c9d8878baa26456174fd7845=7.088854 [2019, các, kết, năm, tư, vsf, đầu]; tbl_916e9082786974a0cfe1315f678c30a3b06f17be4607b5f43e76cf80d96c4389=5.382601 [2019, kết, năm, tư, vsf, và, đầu]; tbl_56cbcb8ac1a87eec41efff9d983110645468e67b3521d3c1d36d6d76a222973d=5.282284 [2019, kết, năm, vsf]; tbl_3af54f7053fa6b59c4fc7d459a42e765f6e463a11b6a33bafa26c32c5262b02f=5.257406 [2019, kết, năm, vsf]; tbl_4809a2aae1ef555902270dfc380be3c7d0983a97b2e17fd9404678ff611bbf16=5.120902 [2019, kết, năm, vsf]; tbl_21a88164c203323e200236ecce1691c5cf2ab7334d5282cc33ccd147e83cae39=5.093154 [2019, kết, năm, vsf]; tbl_57cdf5d3869f9f30d68912f3125b1c7d03755d7087da72e9e079761a16437049=5.093154 [2019, kết, năm, vsf]; tbl_5f2528a814e3613568a1803c666a3e2a85127ad0c49084eaa076ee7629137f12=5.093154 [2019, kết, năm, vsf]; tbl_86cdac87d86f4ee3767f27db3299d81db237acfd1d9a638b223e20cc8c44ca75=5.093154 [2019, kết, năm, vsf]; tbl_0a5f2c4e9fbbedc73d8cce9347158eee39cf32a28064d535a67b11bd79ad9df0=5.065712 [2019, kết, năm, vsf]

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

### retq_9cf253d87eebc6cff165ab32f1d759d9779d11bd67331bb56806fcf6af78a8c4

- Question: Tra cứu lợi nhuận trước thuế, chi phí lãi vay và thuế đã nộp trên LCTT hợp nhất của SCR năm 2018.
- Intent: lookup
- Failure: none
- Predicted table IDs: tbl_260423c1280906e134a91ad617e9ebae2252116efa62122ef422900be97cdeda, tbl_9d635ae56f4836121b5e1279caec5030cfd18ae7bfff1c52cf0a57495ec9ade1, tbl_4d6ddfe5cc4f67a48cb8ef739ec7c1c8a0e1983de400a1b42050643728d74b31, tbl_189f5eab8df9ed4c62deab6e34a3e32f07d2d511d0f5305ffe3c8e5fd05d52a3, tbl_4ddb6b2041e0ad4722e84986eb158fcc9fc057e7bd497f908b3e5771114c04bb, tbl_92eeaf15aa374a9e083fe27e25c4d5c90921bd863b5f1a0abe19572e0478ae7d, tbl_2678e4c3d491a21d1cee6d5ce54276a73d50f38347b4750f2e3e2a19c0c8c4be, tbl_67e1154dcf4dc0c56df6713bfaa359e48b4cc98fbb15361c8dc70926d02be442, tbl_121c3bd22232138d17feb8496a5a71d0b3d6fa9a10e2fed01b9b0c3a9d5510e3, tbl_0aa7357e8adb88d9897184efdb9b52a7eba7079fd01bd6b872b140194e3369a2
- Gold table IDs: tbl_260423c1280906e134a91ad617e9ebae2252116efa62122ef422900be97cdeda
- Missing gold table IDs: (none)
- Eligible documents: 25
- Empty reason: (none)
- Filter counts: company_codes=1713/1713; periods=14691/197; statement_types=13712/25
- Scores and matched tokens: tbl_260423c1280906e134a91ad617e9ebae2252116efa62122ef422900be97cdeda=15.393176 [2018, chi, của, expense, interest, lãi, nộp, phí, scr, tax, thuế, vay, đã]; tbl_9d635ae56f4836121b5e1279caec5030cfd18ae7bfff1c52cf0a57495ec9ade1=15.393176 [2018, chi, của, expense, interest, lãi, nộp, phí, scr, tax, thuế, vay, đã]; tbl_4d6ddfe5cc4f67a48cb8ef739ec7c1c8a0e1983de400a1b42050643728d74b31=11.121840 [2018, interest, lãi, nộp, scr, tax, thuế, vay, đã]; tbl_189f5eab8df9ed4c62deab6e34a3e32f07d2d511d0f5305ffe3c8e5fd05d52a3=11.001179 [2018, interest, lãi, nộp, scr, tax, thuế, vay, đã]; tbl_4ddb6b2041e0ad4722e84986eb158fcc9fc057e7bd497f908b3e5771114c04bb=5.665901 [2018, chi, phí, scr, và]; tbl_92eeaf15aa374a9e083fe27e25c4d5c90921bd863b5f1a0abe19572e0478ae7d=5.540954 [2018, chi, phí, scr, và]; tbl_2678e4c3d491a21d1cee6d5ce54276a73d50f38347b4750f2e3e2a19c0c8c4be=4.621686 [2018, chi, phí, scr]; tbl_67e1154dcf4dc0c56df6713bfaa359e48b4cc98fbb15361c8dc70926d02be442=4.163571 [2018, scr, vay, và]; tbl_121c3bd22232138d17feb8496a5a71d0b3d6fa9a10e2fed01b9b0c3a9d5510e3=4.131574 [2018, lợi, scr]; tbl_0aa7357e8adb88d9897184efdb9b52a7eba7079fd01bd6b872b140194e3369a2=3.907936 [2018, scr, và]

### retq_9ecbb4727d458b5de69ed6ff3355d237b4482e86765f3b9d339c37f86c66c3d7

- Question: Tra cứu tài sản dài hạn và chi phí xây dựng cơ bản dở dang của BAF năm 2020.
- Intent: lookup
- Failure: none
- Predicted table IDs: tbl_65e83bd84d9fd7f8c0a6f279a4551429aa68c51cfe163ed6830dc83429d086e6, tbl_71cb97766e11b4629c94a2971ffa73f770ba7523b3ef386a1a61b8b845859a41, tbl_e2417fa9f302d236cd1be81ebc27d5465860ae40be7cd33d87d5a401ef5081cf, tbl_0ca33b5bfbbf6767171bc221bd837bdb8958562c619a511385a20b653ba69212, tbl_57987264eebcbb382c68401ce0b4241c4b3b71fd5f0ad1ce39a01c7e6d96a7f5, tbl_0e8657449414d71fc328145b728eb0b1ea298034d3194747f0fb0a6629024dbd, tbl_f6171c50d2c8410cc6ee1d46ff059b289105cdad468dd4851078030121b68eca
- Gold table IDs: tbl_65e83bd84d9fd7f8c0a6f279a4551429aa68c51cfe163ed6830dc83429d086e6
- Missing gold table IDs: (none)
- Eligible documents: 7
- Empty reason: (none)
- Filter counts: company_codes=872/872; periods=17285/171; statement_types=7745/7
- Scores and matched tokens: tbl_65e83bd84d9fd7f8c0a6f279a4551429aa68c51cfe163ed6830dc83429d086e6=8.337070 [2020, assets, baf, dài, hạn, sản, tài]; tbl_71cb97766e11b4629c94a2971ffa73f770ba7523b3ef386a1a61b8b845859a41=7.275649 [2020, assets, baf, của, sản, tài, và]; tbl_e2417fa9f302d236cd1be81ebc27d5465860ae40be7cd33d87d5a401ef5081cf=7.275649 [2020, assets, baf, của, sản, tài, và]; tbl_0ca33b5bfbbf6767171bc221bd837bdb8958562c619a511385a20b653ba69212=5.653392 [2020, baf, của, sản, tài, và]; tbl_57987264eebcbb382c68401ce0b4241c4b3b71fd5f0ad1ce39a01c7e6d96a7f5=5.653392 [2020, baf, của, sản, tài, và]; tbl_0e8657449414d71fc328145b728eb0b1ea298034d3194747f0fb0a6629024dbd=5.455953 [2020, baf, hạn, năm, tài, và]; tbl_f6171c50d2c8410cc6ee1d46ff059b289105cdad468dd4851078030121b68eca=5.455953 [2020, baf, hạn, năm, tài, và]

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

### retq_a51c4303ed6e770889e289f07ef1704ee7ccd896f35768a69a203118062b9e16

- Question: So sánh các thuyết minh về chi phí trả trước, thuế phải thu và biến động vốn chủ sở hữu của VJC năm 2017.
- Intent: compare
- Failure: none
- Predicted table IDs: tbl_3dcd9133865a91dc880fdcd0c4eab641a5513105c45c855e5895e847bc24acc0, tbl_392f5a4a29996f4c54bbb57081e4fa55dc06e6af5bc08d736b522a2185ee4389, tbl_67ec2e2aec79a913445d537849ac11cd5bcc3497b2a19c0680f7d69179739af7, tbl_0260eb0565a3720de1bef4d21219c361795259f5c54d05e06f8598318ef6dd6d, tbl_304bdb02a768953b9fecc84ee52f6e8b4e66f315b6d1c656011ecc3127266cd9, tbl_5499346b3d25fb0acdcdbf2a26bcd9d5a6b274ab7bcf4641d444d86b198b8964, tbl_251d7b87ed1883a8927077400d0b4f28a1a4005d5a2cfd47dfe678c408e107e2, tbl_e2e425dda95176443c4f7d9d74ee7ad16901714d5fce764b900b44f3c73394d8
- Gold table IDs: tbl_251d7b87ed1883a8927077400d0b4f28a1a4005d5a2cfd47dfe678c408e107e2, tbl_3dcd9133865a91dc880fdcd0c4eab641a5513105c45c855e5895e847bc24acc0, tbl_67ec2e2aec79a913445d537849ac11cd5bcc3497b2a19c0680f7d69179739af7
- Missing gold table IDs: (none)
- Eligible documents: 8
- Empty reason: (none)
- Filter counts: company_codes=1466/1466; periods=14081/148; statement_types=1479/8
- Scores and matched tokens: tbl_3dcd9133865a91dc880fdcd0c4eab641a5513105c45c855e5895e847bc24acc0=7.086496 [2017, minh, năm, thuyết, vjc]; tbl_392f5a4a29996f4c54bbb57081e4fa55dc06e6af5bc08d736b522a2185ee4389=7.065467 [2017, minh, năm, thuyết, vjc]; tbl_67ec2e2aec79a913445d537849ac11cd5bcc3497b2a19c0680f7d69179739af7=7.025600 [2017, minh, năm, thuyết, vjc]; tbl_0260eb0565a3720de1bef4d21219c361795259f5c54d05e06f8598318ef6dd6d=6.986190 [2017, minh, năm, thuyết, vjc]; tbl_304bdb02a768953b9fecc84ee52f6e8b4e66f315b6d1c656011ecc3127266cd9=6.986190 [2017, minh, năm, thuyết, vjc]; tbl_5499346b3d25fb0acdcdbf2a26bcd9d5a6b274ab7bcf4641d444d86b198b8964=6.986190 [2017, minh, năm, thuyết, vjc]; tbl_251d7b87ed1883a8927077400d0b4f28a1a4005d5a2cfd47dfe678c408e107e2=6.947228 [2017, minh, năm, thuyết, vjc]; tbl_e2e425dda95176443c4f7d9d74ee7ad16901714d5fce764b900b44f3c73394d8=6.947228 [2017, minh, năm, thuyết, vjc]

### retq_a5af50731f92719510085938eae0f29f7cb39b3b0457e15e99748f5f61a22a86

- Question: Tính biến động tài sản, nợ phải trả và kết quả kinh doanh theo bộ phận của HDG từ năm 2017 đến 2018.
- Intent: growth
- Failure: none
- Predicted table IDs: tbl_10e2627cc80c0ae300f38fd7b1b1e0f1f23ffe640606641301deb67833390bc5, tbl_950f5caa42e93487fccf8e1dfc4533f22880cf0515ae5f57474573aacfa9796f, tbl_a1d39a5e3bf01fd2769aab4554569eb42a8b2ce0d550632f55976c298c9fa0e4, tbl_73551cbf4b5c9f3634dadc06bd0d562d1991eca9e6580bb7a1fa325674df6b2f, tbl_dcc520f36510dbfafcdaf1aea521eaf8d613448a128e33a8665374048b5897ba, tbl_733c8054e667194a05bf25d97ab891a21350bb72f3aa93a1c9a9192d6296897c, tbl_580c303b72b12c90934739321cb8038baf8c774268eb2c84f53ccd4b38d7b525, tbl_aa0f0699e29b0b3e6367fe707ffc6c227e5117d6545d0fb1f3dbccf5db749a35, tbl_fc61781b3405bf5693981ea2dcaccfbcc4cd2d9cd36686b5cdad00c54311efbb, tbl_59862a9462c13207b1b2b186681812f7e2b7306753d3e8d6b349d73f0635e62e
- Gold table IDs: tbl_10e2627cc80c0ae300f38fd7b1b1e0f1f23ffe640606641301deb67833390bc5, tbl_733c8054e667194a05bf25d97ab891a21350bb72f3aa93a1c9a9192d6296897c, tbl_73551cbf4b5c9f3634dadc06bd0d562d1991eca9e6580bb7a1fa325674df6b2f
- Missing gold table IDs: (none)
- Eligible documents: 13
- Empty reason: (none)
- Filter counts: company_codes=1465/1465; periods=14691/161; statement_types=1479/13
- Scores and matched tokens: tbl_10e2627cc80c0ae300f38fd7b1b1e0f1f23ffe640606641301deb67833390bc5=12.971701 [2017, 2018, hdg, kết, liabilities, năm, nợ, phải, sản, theo, total, trả, tài]; tbl_950f5caa42e93487fccf8e1dfc4533f22880cf0515ae5f57474573aacfa9796f=12.427208 [2018, hdg, kết, liabilities, năm, nợ, phải, sản, theo, total, trả, tài]; tbl_a1d39a5e3bf01fd2769aab4554569eb42a8b2ce0d550632f55976c298c9fa0e4=12.427208 [2018, hdg, kết, liabilities, năm, nợ, phải, sản, theo, total, trả, tài]; tbl_73551cbf4b5c9f3634dadc06bd0d562d1991eca9e6580bb7a1fa325674df6b2f=12.399591 [2018, hdg, kết, liabilities, năm, nợ, phải, sản, theo, total, trả, tài]; tbl_dcc520f36510dbfafcdaf1aea521eaf8d613448a128e33a8665374048b5897ba=7.288027 [2017, 2018, hdg, kết, năm, theo, tài]; tbl_733c8054e667194a05bf25d97ab891a21350bb72f3aa93a1c9a9192d6296897c=7.052778 [2017, 2018, doanh, hdg, kết, năm, theo, tài]; tbl_580c303b72b12c90934739321cb8038baf8c774268eb2c84f53ccd4b38d7b525=6.519936 [2018, hdg, kết, năm, theo, tài]; tbl_aa0f0699e29b0b3e6367fe707ffc6c227e5117d6545d0fb1f3dbccf5db749a35=6.519936 [2018, hdg, kết, năm, theo, tài]; tbl_fc61781b3405bf5693981ea2dcaccfbcc4cd2d9cd36686b5cdad00c54311efbb=6.447330 [2018, hdg, kết, năm, theo, tài]; tbl_59862a9462c13207b1b2b186681812f7e2b7306753d3e8d6b349d73f0635e62e=6.411643 [2018, hdg, kết, năm, theo, tài]

### retq_a92e064c27aea88ce6b2d4d81bae32e94fc23ac351036e8fd88058ed484d3d62

- Question: Tra cứu biến động tiền gửi, tiền vay và kinh doanh chứng khoán trên LCTT riêng của SHB năm 2020.
- Intent: lookup
- Failure: zero_gold_hits
- Predicted table IDs: tbl_6cd9280fb1ccf696546c3105309cfb03d36b793aaf34c19ae2ee7182bbf03b5a, tbl_9c0ecd8483c744dbde1a8c63b889cab1d316a8f93cc89ef1e712e5166a1e2aee, tbl_bcc298850727ce47ffb486a816084a898b0248c262d6a5e30879a60d88ed4abd, tbl_db9f28f018705e45aaf7c2f0612afde43f2d8fd1f5662778e8862e14dbc5ece0, tbl_e72e098266759ee6cb9aee74fc2a06dc0c8d62d78b4dd3cf6f3b5ecb7fd9f2f1, tbl_1b86ffe5d9531b3725f7a6b178811c39161ef247190fac181131af8ba1cb7aa3, tbl_1a542b2c26ce1ea6f0de6a612c13a8ea2087cefb4f8e2d69f5ac236fc436d0a0, tbl_300e02a3be9aa06b1837ac909ec75afd0a182d5d6e7d6871aa119800d1ff262c, tbl_5ae75c42f82fb0199678428ba30d633feed0b3c86ecd343a549bd022e88fb0d7, tbl_d7cabbe6198ab299af4d4fe56a5254815974df9e294ddb307750a43bb36c3543
- Gold table IDs: tbl_4a47c080f3d554db3a4571de05947364e88cc56c9d3c4072f8ec2ba0360ac805
- Missing gold table IDs: tbl_4a47c080f3d554db3a4571de05947364e88cc56c9d3c4072f8ec2ba0360ac805
- Eligible documents: 13
- Empty reason: (none)
- Filter counts: company_codes=2238/2238; periods=17285/234; statement_types=13712/13
- Scores and matched tokens: tbl_6cd9280fb1ccf696546c3105309cfb03d36b793aaf34c19ae2ee7182bbf03b5a=4.015346 [2020, của, shb, động]; tbl_9c0ecd8483c744dbde1a8c63b889cab1d316a8f93cc89ef1e712e5166a1e2aee=3.917804 [2020, năm, shb]; tbl_bcc298850727ce47ffb486a816084a898b0248c262d6a5e30879a60d88ed4abd=3.917804 [2020, năm, shb]; tbl_db9f28f018705e45aaf7c2f0612afde43f2d8fd1f5662778e8862e14dbc5ece0=3.917804 [2020, năm, shb]; tbl_e72e098266759ee6cb9aee74fc2a06dc0c8d62d78b4dd3cf6f3b5ecb7fd9f2f1=3.917804 [2020, năm, shb]; tbl_1b86ffe5d9531b3725f7a6b178811c39161ef247190fac181131af8ba1cb7aa3=3.561680 [2020, năm, shb]; tbl_1a542b2c26ce1ea6f0de6a612c13a8ea2087cefb4f8e2d69f5ac236fc436d0a0=3.217655 [2020, shb, động]; tbl_300e02a3be9aa06b1837ac909ec75afd0a182d5d6e7d6871aa119800d1ff262c=3.217655 [2020, shb, động]; tbl_5ae75c42f82fb0199678428ba30d633feed0b3c86ecd343a549bd022e88fb0d7=3.217655 [2020, shb, động]; tbl_d7cabbe6198ab299af4d4fe56a5254815974df9e294ddb307750a43bb36c3543=3.217655 [2020, shb, động]

### retq_af86db7ee5e169a65e2211bd3a20af63c4ecd906bf97d4572b15d0b6ea01859b

- Question: Tính biến động và rà soát cơ cấu các công ty con, công ty liên kết của SJG trong năm 2018.
- Intent: growth
- Failure: zero_gold_hits
- Predicted table IDs: tbl_5ebe2562368be829ee0b5991b52d554678e603bf4a2b68eb550c9c374a998614, tbl_5e89cdff7d6cfd9656789049d96779e76703bdde43572ff2bc50a55a9c78b713, tbl_08a1347ad21595e39ab185c569eb19e269e096c4852286de3e8944772edd2d6f, tbl_1cc4620fdd365e8d31cc7d5f6e8f8e6ca4429053f73c2ce0d56858d9bd73d92e, tbl_33dbbb8195d92fcaa176db24fa018ba583e685e5b455d50d8e3c74016f528efb, tbl_3fd94a4efa2f9d2388b4d86f2ac386f6f759dee49abdb5697cb6213f7dd13e8f, tbl_4cbaf89da1b07de2ea55b96315e56f3335ee2bc01b1c65a3111d34d9dd4a5a8f, tbl_60c9a00ca389d599ce985fe01ed9efb6098966ea8577c6f3041fcd38dcc42c4e, tbl_7f8e44bc632a30b6fc4b91fd28fbf6f72379b73241c41bd1623aa97b155d814c, tbl_8fe8f08786aa417bdae87a61e6095e7374823c5c19eca9ce8d3f7c17db8ecc5d
- Gold table IDs: tbl_48dc6bdecfc04c5fece016f1a2e710e71b7576fb8ed92e482ec71ce61a6345fb, tbl_9eb0512b316a80754491a8cfb2acac52b79c8cd1c89e7c366f272a27f846a470, tbl_ff659cea1878c04aa499e1360803b41e95dcf12aea9bee5859dbbb483a115bbf
- Missing gold table IDs: tbl_48dc6bdecfc04c5fece016f1a2e710e71b7576fb8ed92e482ec71ce61a6345fb, tbl_9eb0512b316a80754491a8cfb2acac52b79c8cd1c89e7c366f272a27f846a470, tbl_ff659cea1878c04aa499e1360803b41e95dcf12aea9bee5859dbbb483a115bbf
- Eligible documents: 21
- Empty reason: (none)
- Filter counts: company_codes=1056/1056; periods=14691/174; statement_types=1479/21
- Scores and matched tokens: tbl_5ebe2562368be829ee0b5991b52d554678e603bf4a2b68eb550c9c374a998614=3.087439 [2018, sjg]; tbl_5e89cdff7d6cfd9656789049d96779e76703bdde43572ff2bc50a55a9c78b713=2.978044 [2018, năm, sjg]; tbl_08a1347ad21595e39ab185c569eb19e269e096c4852286de3e8944772edd2d6f=2.972740 [2018, sjg]; tbl_1cc4620fdd365e8d31cc7d5f6e8f8e6ca4429053f73c2ce0d56858d9bd73d92e=2.972740 [2018, sjg]; tbl_33dbbb8195d92fcaa176db24fa018ba583e685e5b455d50d8e3c74016f528efb=2.972740 [2018, sjg]; tbl_3fd94a4efa2f9d2388b4d86f2ac386f6f759dee49abdb5697cb6213f7dd13e8f=2.972740 [2018, sjg]; tbl_4cbaf89da1b07de2ea55b96315e56f3335ee2bc01b1c65a3111d34d9dd4a5a8f=2.972740 [2018, sjg]; tbl_60c9a00ca389d599ce985fe01ed9efb6098966ea8577c6f3041fcd38dcc42c4e=2.972740 [2018, sjg]; tbl_7f8e44bc632a30b6fc4b91fd28fbf6f72379b73241c41bd1623aa97b155d814c=2.972740 [2018, sjg]; tbl_8fe8f08786aa417bdae87a61e6095e7374823c5c19eca9ce8d3f7c17db8ecc5d=2.972740 [2018, sjg]

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

### retq_b8f5f17c7fa19ee92df6e919eb20b531855fc9c9fc46294da23b13d9025070de

- Question: Tra cứu bốn bảng thuyết minh về các công ty con sở hữu trực tiếp và gián tiếp của MML năm 2017.
- Intent: lookup
- Failure: partial_gold_hits
- Predicted table IDs: tbl_44693fe2169ec4a6ec878a16b49020ea67db1e0ca5d74550f28f38800aced3c6, tbl_13f0470d32f60ce94d27dfb4b77c31d43aa79cd3d8611637a95a53692c70f4aa, tbl_43ba8c1f31b79107cbdf159db4f7759cc8df7bf3e7c5583dfa2dd22226566cb7, tbl_8d8d49f4b6105c225a3f1cacc34c467d1c6b5e1396fb2433340c6c37e10afa73, tbl_d2552a60408c5ff017c8edd044a062084f2b1177da56166d7606886d72934405, tbl_12a31093994bf2daec54a30bdba2701871cb3ae1d8c68440b5b4df19c7d0ffcd, tbl_2eb67f9545187a1fc7d09e33a49e4931e947495652fa1c44b9b0ec1e5bc444dc, tbl_35d46ff75faa777ad42d5d631eb9e8f0a8b0e36d7db597f60ca86474fb904197, tbl_69019bfb20d65abbdb7fa27ec16ccc50e1e4029121baf200e326696bcdbede8b, tbl_8ed2a2dab81ff19019090ff87a8d6ddd71bf5301f37ac46c3aa93f9d0d4294d9
- Gold table IDs: tbl_13f0470d32f60ce94d27dfb4b77c31d43aa79cd3d8611637a95a53692c70f4aa, tbl_43ba8c1f31b79107cbdf159db4f7759cc8df7bf3e7c5583dfa2dd22226566cb7, tbl_69019bfb20d65abbdb7fa27ec16ccc50e1e4029121baf200e326696bcdbede8b, tbl_add6276f804803e43654a26340b64c08c31cb2a83b931af1819d8d94d41aebf1
- Missing gold table IDs: tbl_add6276f804803e43654a26340b64c08c31cb2a83b931af1819d8d94d41aebf1
- Eligible documents: 14
- Empty reason: (none)
- Filter counts: company_codes=1009/1009; periods=14081/124; statement_types=1479/14
- Scores and matched tokens: tbl_44693fe2169ec4a6ec878a16b49020ea67db1e0ca5d74550f28f38800aced3c6=8.396941 [2017, minh, mml, năm, thuyết, tiếp]; tbl_13f0470d32f60ce94d27dfb4b77c31d43aa79cd3d8611637a95a53692c70f4aa=8.348814 [2017, minh, mml, năm, thuyết, tiếp]; tbl_43ba8c1f31b79107cbdf159db4f7759cc8df7bf3e7c5583dfa2dd22226566cb7=8.348814 [2017, minh, mml, năm, thuyết, tiếp]; tbl_8d8d49f4b6105c225a3f1cacc34c467d1c6b5e1396fb2433340c6c37e10afa73=8.301243 [2017, minh, mml, năm, thuyết, tiếp]; tbl_d2552a60408c5ff017c8edd044a062084f2b1177da56166d7606886d72934405=8.301243 [2017, minh, mml, năm, thuyết, tiếp]; tbl_12a31093994bf2daec54a30bdba2701871cb3ae1d8c68440b5b4df19c7d0ffcd=8.254222 [2017, minh, mml, năm, thuyết, tiếp]; tbl_2eb67f9545187a1fc7d09e33a49e4931e947495652fa1c44b9b0ec1e5bc444dc=8.254222 [2017, minh, mml, năm, thuyết, tiếp]; tbl_35d46ff75faa777ad42d5d631eb9e8f0a8b0e36d7db597f60ca86474fb904197=8.254222 [2017, minh, mml, năm, thuyết, tiếp]; tbl_69019bfb20d65abbdb7fa27ec16ccc50e1e4029121baf200e326696bcdbede8b=8.254222 [2017, minh, mml, năm, thuyết, tiếp]; tbl_8ed2a2dab81ff19019090ff87a8d6ddd71bf5301f37ac46c3aa93f9d0d4294d9=8.254222 [2017, minh, mml, năm, thuyết, tiếp]

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

### retq_c3b64537544ba45a111dd7a0aced43190d1073ed76f5927986e770e74c0c08b5

- Question: Tra cứu tổng tài sản và tổng nợ phải trả trên bảng CĐKT hợp nhất của DIG năm 2024.
- Intent: lookup
- Failure: none
- Predicted table IDs: tbl_77efd10cc63a9377c7a685c4baf0f3a263b5957a57b32b5b58075c006abed915, tbl_222afcb819b2133c481f142d28f749571c9cf19647b15ae586a45387ffd20dc9, tbl_66ecf10837be5a61044b1be7338a5f0c709fa00891f394c1d71e2b98d4065add, tbl_7939fa489e4857e053390dfb96e99fb7b70391b4e9850c5634e22cc457606482, tbl_c9eac2323e8b59e5cbe2da5c7926c2aa4074efce81834bf5c1c0b230793447d8, tbl_5a3e43ce4afcfe9465f5a0b168cf607502fb16174016bfc6c47759d788cca4a0, tbl_07b08e957b3a5313d84c4f1fa30874a579178fbf78ea50933869e2111b941d3a, tbl_dd189aa1e46776919aed2248799d5c9de6e35514eabdc3a05925f50a39ecdafc, tbl_ab9636162330bc624647dc995daa4bbffb0addd28e09f6f21e69dcdeb0ce065e, tbl_0a6f8be915002f1c6215df23755f6ffe4a2fa4585e8589f9b2d1fc1c1af49895
- Gold table IDs: tbl_222afcb819b2133c481f142d28f749571c9cf19647b15ae586a45387ffd20dc9
- Missing gold table IDs: (none)
- Eligible documents: 14
- Empty reason: (none)
- Filter counts: company_codes=1765/1765; periods=20269/334; statement_types=7745/14
- Scores and matched tokens: tbl_77efd10cc63a9377c7a685c4baf0f3a263b5957a57b32b5b58075c006abed915=10.852092 [2024, của, dig, hợp, liabilities, nhất, nợ, phải, sản, total, trả, tài, và]; tbl_222afcb819b2133c481f142d28f749571c9cf19647b15ae586a45387ffd20dc9=8.290724 [2024, assets, dig, liabilities, total]; tbl_66ecf10837be5a61044b1be7338a5f0c709fa00891f394c1d71e2b98d4065add=8.017351 [2024, assets, dig, liabilities, total]; tbl_7939fa489e4857e053390dfb96e99fb7b70391b4e9850c5634e22cc457606482=7.801106 [2024, assets, dig, sản, total, tài, tổng]; tbl_c9eac2323e8b59e5cbe2da5c7926c2aa4074efce81834bf5c1c0b230793447d8=7.801106 [2024, assets, dig, sản, total, tài, tổng]; tbl_5a3e43ce4afcfe9465f5a0b168cf607502fb16174016bfc6c47759d788cca4a0=7.123710 [2024, bảng, dig, hợp, nhất]; tbl_07b08e957b3a5313d84c4f1fa30874a579178fbf78ea50933869e2111b941d3a=6.573375 [2024, assets, dig, total]; tbl_dd189aa1e46776919aed2248799d5c9de6e35514eabdc3a05925f50a39ecdafc=6.573375 [2024, assets, dig, total]; tbl_ab9636162330bc624647dc995daa4bbffb0addd28e09f6f21e69dcdeb0ce065e=2.720730 [2024, dig]; tbl_0a6f8be915002f1c6215df23755f6ffe4a2fa4585e8589f9b2d1fc1c1af49895=2.702367 [2024, dig]

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

### retq_d18b935bc13fd5a2ddaae8ac584243d78c0d55b66468f26d18e8d92bd5af1dcd

- Question: So sánh chi phí thuế hiện hành của VPB giữa năm 2019 và 2020.
- Intent: compare
- Failure: none
- Predicted table IDs: tbl_018af6d64be6d6ae7c556447c8de8078a47931247aa2f460c6aaff65898fb27b, tbl_a5de91eb29ab68649affcac432f917abb85c5cfd7d0cdc1139dc50d912af1695, tbl_5f8b0769c03c95b45da5c43e13211159f49cfb0d6e99723500834e27037726ce, tbl_34accf8c1649dd2b35947c2f47139bad6701085372201a140ecf33cc503e17da, tbl_473e40800a0a1dc2007f98bbd168606ccff12c55d0417be97283447b4dce2017, tbl_5b536201e4860f63d8b5546f500a571349a7ae75a15118c1fd66fe5b347059a8, tbl_66668d116cc9ce791220f65ee6997b53d5bba84e5ad8b12e7ea5825f54d0e2da, tbl_7cce08f58bc3a564c4054bcdc32874e8663c70bea34732ac67656ab81d6d36bb, tbl_f2cbf23adcd45272a87252368bc765fa9ba195205901457e7ec6c98713e7673a, tbl_affa8fa8d30fd9a45d769485640bc324e478ccc228c329cbf5e76b0ab476b52a
- Gold table IDs: tbl_34accf8c1649dd2b35947c2f47139bad6701085372201a140ecf33cc503e17da
- Missing gold table IDs: (none)
- Eligible documents: 11
- Empty reason: (none)
- Filter counts: company_codes=1952/1952; periods=29768/456; statement_types=7862/11
- Scores and matched tokens: tbl_018af6d64be6d6ae7c556447c8de8078a47931247aa2f460c6aaff65898fb27b=8.616546 [2019, chi, hiện, hành, năm, phí, thuế, vpb]; tbl_a5de91eb29ab68649affcac432f917abb85c5cfd7d0cdc1139dc50d912af1695=8.249692 [2019, chi, hiện, hành, năm, phí, thuế, vpb]; tbl_5f8b0769c03c95b45da5c43e13211159f49cfb0d6e99723500834e27037726ce=3.701142 [2019, thuế, vpb]; tbl_34accf8c1649dd2b35947c2f47139bad6701085372201a140ecf33cc503e17da=3.399247 [2019, 2020, vpb]; tbl_473e40800a0a1dc2007f98bbd168606ccff12c55d0417be97283447b4dce2017=3.378023 [2019, 2020, vpb]; tbl_5b536201e4860f63d8b5546f500a571349a7ae75a15118c1fd66fe5b347059a8=2.843713 [2019, vpb]; tbl_66668d116cc9ce791220f65ee6997b53d5bba84e5ad8b12e7ea5825f54d0e2da=2.843713 [2019, vpb]; tbl_7cce08f58bc3a564c4054bcdc32874e8663c70bea34732ac67656ab81d6d36bb=2.798771 [2020, vpb]; tbl_f2cbf23adcd45272a87252368bc765fa9ba195205901457e7ec6c98713e7673a=2.798771 [2020, vpb]; tbl_affa8fa8d30fd9a45d769485640bc324e478ccc228c329cbf5e76b0ab476b52a=2.535007 [2020, vpb]

### retq_d232835d75041d3a6a2f1b8c143f5cb34e83df0ce9dbfd82e70510b073045214

- Question: Tra cứu tài sản và nợ phải trả theo bộ phận của IJC năm 2025.
- Intent: lookup
- Failure: none
- Predicted table IDs: tbl_9f4f401306be9be98dedcdece66fc54dd28529c2e767377cbceb11a9fbf488b9, tbl_df99b6d5f2a6a577adf7d0396e5761b14d8b13c5e1a273a81477154498d99be8, tbl_f7e68bcf49a2f955a317d09d03b032f8b9b39d9de564b699d2fa031b9630fb5b, tbl_461da8ecc0294e9391b1e5bd0b9b3b4fa7ee3c8e7c88a2f2c2540f282b96258e, tbl_38d0c2d66332bc4070da2a13442e1a64268f2a791358a5ef4909095b291e8428, tbl_d6a8a0ca3f0a7a9c5ea63165242f5f17312c78ee8d12ab6aa10d1bfb0218ee28, tbl_10c64da4e4d9219de068920aa1e3407d5d209e335b88fb661d56e0b65f11d482, tbl_fc0869b0a33348089f4180860aeb89e5cf80b83a25621d9234e618b548178f64
- Gold table IDs: tbl_9f4f401306be9be98dedcdece66fc54dd28529c2e767377cbceb11a9fbf488b9
- Missing gold table IDs: (none)
- Eligible documents: 8
- Empty reason: (none)
- Filter counts: company_codes=1483/1483; periods=16033/154; statement_types=7745/8
- Scores and matched tokens: tbl_9f4f401306be9be98dedcdece66fc54dd28529c2e767377cbceb11a9fbf488b9=16.003561 [2025, bộ, của, ijc, liabilities, nợ, phải, phận, sản, theo, total, trả, tài, và]; tbl_df99b6d5f2a6a577adf7d0396e5761b14d8b13c5e1a273a81477154498d99be8=15.650434 [2025, bộ, của, ijc, liabilities, nợ, phải, phận, sản, theo, total, trả, tài, và]; tbl_f7e68bcf49a2f955a317d09d03b032f8b9b39d9de564b699d2fa031b9630fb5b=6.187387 [2025, ijc, liabilities, theo, total]; tbl_461da8ecc0294e9391b1e5bd0b9b3b4fa7ee3c8e7c88a2f2c2540f282b96258e=3.777209 [2025, ijc, theo]; tbl_38d0c2d66332bc4070da2a13442e1a64268f2a791358a5ef4909095b291e8428=3.727234 [2025, ijc, theo]; tbl_d6a8a0ca3f0a7a9c5ea63165242f5f17312c78ee8d12ab6aa10d1bfb0218ee28=3.727234 [2025, ijc, theo]; tbl_10c64da4e4d9219de068920aa1e3407d5d209e335b88fb661d56e0b65f11d482=2.896798 [2025, ijc]; tbl_fc0869b0a33348089f4180860aeb89e5cf80b83a25621d9234e618b548178f64=2.896798 [2025, ijc]

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

### retq_de3c339b336e378295c8a5298d140ca41aa623690c6aed3616d4b9abc3dbe919

- Question: Tính tăng trưởng lưu chuyển tiền thuần từ hoạt động kinh doanh của HDG từ năm 2018 đến 2019.
- Intent: growth
- Failure: none
- Predicted table IDs: tbl_939681459754e7b620d3e34c2ba227122d76bf6011bdf57124c8d40a0e5027cc, tbl_ccd837d4df050f68677f4f5102ddd2b6f9f155eb2260fc80394bc9fdc007863d, tbl_6d9654a9f136c0e91a999523d7db03b442d3a3964581050b194d75917b68c4ae, tbl_641b92c54b05300e297d04c01bfb40f582ceaa86da25a9e952ca5f4c9e997067, tbl_de4a1c75e148991c98a82c2fefa69de73fb76173b82d0164ca4d779fde92c718, tbl_a33cbb7cbef322e13b01382644cc1d035d851235f413eb612edc4c87bff6b95f, tbl_983b6b9fe93e869f09a01f211f89d933d3b682f1d304533f1044f866f14bb9f3, tbl_c87836e8a5434a83d210cf3df1eaaea243211a29ab9a246dd0f8c8e7371f342a, tbl_fde9e19d6fcb2ee80d5ba6dd34d02f61f7466f9f6c05eba15884b84245686d65, tbl_88d4777d3fa44d3d1f06cd27a3409431a6b1a1f8567bc4b3cae7507c0e0a1ce7
- Gold table IDs: tbl_ccd837d4df050f68677f4f5102ddd2b6f9f155eb2260fc80394bc9fdc007863d
- Missing gold table IDs: (none)
- Eligible documents: 32
- Empty reason: (none)
- Filter counts: company_codes=1465/1465; periods=27650/270; statement_types=13712/32
- Scores and matched tokens: tbl_939681459754e7b620d3e34c2ba227122d76bf6011bdf57124c8d40a0e5027cc=19.032642 [2018, 2019, cash, chuyển, của, doanh, flow, hdg, hoạt, kinh, lưu, operating, thuần, tiền, từ, động]; tbl_ccd837d4df050f68677f4f5102ddd2b6f9f155eb2260fc80394bc9fdc007863d=19.032642 [2018, 2019, cash, chuyển, của, doanh, flow, hdg, hoạt, kinh, lưu, operating, thuần, tiền, từ, động]; tbl_6d9654a9f136c0e91a999523d7db03b442d3a3964581050b194d75917b68c4ae=18.986237 [2019, cash, chuyển, doanh, flow, hdg, hoạt, kinh, lưu, operating, thuần, tiền, tính, từ, động]; tbl_641b92c54b05300e297d04c01bfb40f582ceaa86da25a9e952ca5f4c9e997067=18.956783 [2018, cash, chuyển, của, doanh, flow, hdg, hoạt, kinh, lưu, operating, thuần, tiền, từ, động]; tbl_de4a1c75e148991c98a82c2fefa69de73fb76173b82d0164ca4d779fde92c718=18.956783 [2018, cash, chuyển, của, doanh, flow, hdg, hoạt, kinh, lưu, operating, thuần, tiền, từ, động]; tbl_a33cbb7cbef322e13b01382644cc1d035d851235f413eb612edc4c87bff6b95f=16.379925 [2019, cash, chuyển, flow, hdg, hoạt, lưu, thuần, tiền, tính, từ, động]; tbl_983b6b9fe93e869f09a01f211f89d933d3b682f1d304533f1044f866f14bb9f3=14.469789 [2018, 2019, cash, chuyển, của, flow, hdg, hoạt, lưu, thuần, tiền, từ, động]; tbl_c87836e8a5434a83d210cf3df1eaaea243211a29ab9a246dd0f8c8e7371f342a=14.383466 [2019, cash, chuyển, flow, hdg, hoạt, lưu, thuần, tiền, tính, từ, động]; tbl_fde9e19d6fcb2ee80d5ba6dd34d02f61f7466f9f6c05eba15884b84245686d65=13.698565 [2018, 2019, cash, chuyển, của, flow, hdg, hoạt, lưu, thuần, tiền, từ, động]; tbl_88d4777d3fa44d3d1f06cd27a3409431a6b1a1f8567bc4b3cae7507c0e0a1ce7=13.656638 [2018, cash, chuyển, của, flow, hdg, hoạt, lưu, thuần, tiền, từ, động]

### retq_df6f68634d600a5378892f46c619a80464948abd11da8a233fe225e749590992

- Question: Tra cứu cho vay khách hàng và chứng khoán đầu tư của SSB tại cuối năm 2020.
- Intent: lookup
- Failure: none
- Predicted table IDs: tbl_b150862fc77a0962f9d21d894a19365a9737fda5473d7f46cc6c433907a82cb6, tbl_f2f914b3245b5b231a6c545af3f0b12334924b9acb58d925ab3cd19efab4d3be, tbl_6435529651b713bf8a3c6bf97f4e356702cbeac6d545bc8e02c6c2f2e8235637, tbl_f51aa33063f280150facd95cb3ee65602bf5128975b37bf410c41232c028cdc4, tbl_c76bc3b845eafe28624a765b41d9ea6efc81424e657f77fe63f6dd918c9f4cf0, tbl_d89b30b076f9f9a59ba8585ff0b4ef2c420127051dd89095d040be353b1d84a4, tbl_f2ede9706dc7631530173a1950fb2ba1b58dd835c2bdb7bb26a0cb4eb0f24f87, tbl_b1d37e67c30ca0bb8326839842fec42add69ae39043ae2729439d08f8e3ec61e, tbl_daa22065b072f7f0fa41b2b44436aeb3aa5e492fb252f86c06ffe8029a3fd8af, tbl_b2ceef9ffce732b149423fe4a29f7211235a71366050e064d91db87dde02cac6
- Gold table IDs: tbl_c76bc3b845eafe28624a765b41d9ea6efc81424e657f77fe63f6dd918c9f4cf0
- Missing gold table IDs: (none)
- Eligible documents: 12
- Empty reason: (none)
- Filter counts: company_codes=1235/1235; periods=17285/220; statement_types=7745/12
- Scores and matched tokens: tbl_b150862fc77a0962f9d21d894a19365a9737fda5473d7f46cc6c433907a82cb6=5.918186 [2020, của, hàng, năm, ssb, tại]; tbl_f2f914b3245b5b231a6c545af3f0b12334924b9acb58d925ab3cd19efab4d3be=5.612142 [2020, của, năm, ssb, tại, và]; tbl_6435529651b713bf8a3c6bf97f4e356702cbeac6d545bc8e02c6c2f2e8235637=5.216080 [2020, của, hàng, năm, ssb, tư]; tbl_f51aa33063f280150facd95cb3ee65602bf5128975b37bf410c41232c028cdc4=5.065505 [2020, năm, ssb, tại, và]; tbl_c76bc3b845eafe28624a765b41d9ea6efc81424e657f77fe63f6dd918c9f4cf0=4.918683 [2020, của, hàng, năm, ssb, tư]; tbl_d89b30b076f9f9a59ba8585ff0b4ef2c420127051dd89095d040be353b1d84a4=4.918683 [2020, của, hàng, năm, ssb, tư]; tbl_f2ede9706dc7631530173a1950fb2ba1b58dd835c2bdb7bb26a0cb4eb0f24f87=4.717181 [2020, năm, ssb, tại, và]; tbl_b1d37e67c30ca0bb8326839842fec42add69ae39043ae2729439d08f8e3ec61e=4.148553 [2020, của, hàng, ssb]; tbl_daa22065b072f7f0fa41b2b44436aeb3aa5e492fb252f86c06ffe8029a3fd8af=4.145269 [2020, của, hàng, ssb]; tbl_b2ceef9ffce732b149423fe4a29f7211235a71366050e064d91db87dde02cac6=3.448791 [2020, hàng, ssb]

### retq_e2059ca4277998aa17e8e41c3881e93960b982416b06a73b520628e942e263d5

- Question: Tính tăng trưởng lợi nhuận sau thuế của OCB từ năm 2019 đến 2020.
- Intent: growth
- Failure: none
- Predicted table IDs: tbl_be52d4a6572cb0385ef6b9660fcd7c79628839258f0cd621709ae7ea0d45ee53, tbl_3e7aaa675a2a1389aa1e56ec4e7c993e3f3fdba5bae6db8c2613e0f282ee4342, tbl_d65e95512e51c95be8827a0844077a899babbe21a5914b1a0aa69f714354cdd3, tbl_0ed9cf46ebecbdba294f38af4b0f2444409711e2547ee15c5ea75259e4522d13, tbl_45b2031a8116a40ea6adfd0e0449431ebe3bf666b059c1dde86f46c8cdba3dba, tbl_b697330d19f5f1f6c66708c908074bcc3fdbba0d05f5472a418eafbb37e15503, tbl_f25dfadecac1c22b33615b4b894959fa0f926a3cf0c158b09f470ae743d32824, tbl_fb8bab1e95a59b7dce4b4d78aba6798c26c745bd8283e12981a87d4ae4f2e2fb, tbl_44c1306a1460046297408259d04e68c06ec9da0ccabdb81eb552a7179111676a, tbl_8c563f97e4456b69e3bc3121eb14d2045d54aad9fc96d20188b4c0d44dd47107
- Gold table IDs: tbl_d65e95512e51c95be8827a0844077a899babbe21a5914b1a0aa69f714354cdd3
- Missing gold table IDs: (none)
- Eligible documents: 12
- Empty reason: (none)
- Filter counts: company_codes=2572/2572; periods=29768/442; statement_types=7862/12
- Scores and matched tokens: tbl_be52d4a6572cb0385ef6b9660fcd7c79628839258f0cd621709ae7ea0d45ee53=13.233921 [2019, 2020, after, lợi, nhuận, ocb, profit, sau, tax, thuế]; tbl_3e7aaa675a2a1389aa1e56ec4e7c993e3f3fdba5bae6db8c2613e0f282ee4342=12.903092 [2019, 2020, after, lợi, nhuận, ocb, profit, sau, tax, thuế]; tbl_d65e95512e51c95be8827a0844077a899babbe21a5914b1a0aa69f714354cdd3=12.833059 [2019, 2020, after, lợi, nhuận, ocb, profit, sau, tax, thuế]; tbl_0ed9cf46ebecbdba294f38af4b0f2444409711e2547ee15c5ea75259e4522d13=12.094721 [2020, after, lợi, nhuận, ocb, profit, sau, tax, thuế]; tbl_45b2031a8116a40ea6adfd0e0449431ebe3bf666b059c1dde86f46c8cdba3dba=12.029411 [2020, after, lợi, nhuận, ocb, profit, sau, tax, thuế]; tbl_b697330d19f5f1f6c66708c908074bcc3fdbba0d05f5472a418eafbb37e15503=9.363446 [2019, lợi, nhuận, ocb, profit, tax, thuế]; tbl_f25dfadecac1c22b33615b4b894959fa0f926a3cf0c158b09f470ae743d32824=9.363446 [2019, lợi, nhuận, ocb, profit, tax, thuế]; tbl_fb8bab1e95a59b7dce4b4d78aba6798c26c745bd8283e12981a87d4ae4f2e2fb=5.242129 [2019, ocb, tax, thuế]; tbl_44c1306a1460046297408259d04e68c06ec9da0ccabdb81eb552a7179111676a=5.218592 [2019, ocb, tax, thuế]; tbl_8c563f97e4456b69e3bc3121eb14d2045d54aad9fc96d20188b4c0d44dd47107=2.715675 [2020, ocb]

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

### retq_ea502802099c055da44f3cf5e1352444a2df08eed91049e1a8e463f3277ba886

- Question: Tính biến động chi phí thuế thu nhập doanh nghiệp của NAB từ năm 2019 đến 2020.
- Intent: growth
- Failure: none
- Predicted table IDs: tbl_7088783024659eb52e393b75b2b6f69f72665b0b34d8ff803f13298d9c215089, tbl_98274abe066c6e2824c8f6c607d7059b13cf617433dfd22a7b4684724d7d3f06, tbl_d0649a598920b54cde96d94e39a4a4f8ce1e10784687f10af691a02511cf2ea7, tbl_2af32a540e6cac56be340d7abb7d42eb617f31e960b55256fd41a787d41239d1, tbl_32fd657cb61d67d8b2ae6a5b17735e83e71a1b4f1aa3234fd2e0dda66dcf27b8, tbl_85a238500b92c47a0379c1f9dffd510e6818b128ecfd23f6df02f2ed389640bc, tbl_160897096825b92e1109ea0f87da4d0018a18834f5f8a5d236087f44c7e13e69, tbl_9ff265a0e9dc856d6e4e515114bbadb0b2bb3fd4c4e80a37d10aaf0b1dc69f89, tbl_d1bbddc89d0ae6998d635cfd49ff3cb8fba66ec1d6ce38bda2c2b4454ea5f0c6, tbl_6b3f05fb39f9ba9b73aeb27fb2e66bd6545d2149934273ae1dda196443ef57b8
- Gold table IDs: tbl_7088783024659eb52e393b75b2b6f69f72665b0b34d8ff803f13298d9c215089
- Missing gold table IDs: (none)
- Eligible documents: 10
- Empty reason: (none)
- Filter counts: company_codes=1621/1621; periods=29768/271; statement_types=7862/10
- Scores and matched tokens: tbl_7088783024659eb52e393b75b2b6f69f72665b0b34d8ff803f13298d9c215089=12.285687 [2019, 2020, chi, doanh, nab, nghiệp, nhập, phí, thu, thuế, động]; tbl_98274abe066c6e2824c8f6c607d7059b13cf617433dfd22a7b4684724d7d3f06=12.233160 [2019, 2020, chi, doanh, nab, nghiệp, nhập, phí, thu, thuế, động]; tbl_d0649a598920b54cde96d94e39a4a4f8ce1e10784687f10af691a02511cf2ea7=9.553471 [2020, chi, doanh, nab, nghiệp, nhập, phí, thu, thuế, động]; tbl_2af32a540e6cac56be340d7abb7d42eb617f31e960b55256fd41a787d41239d1=9.500963 [2020, chi, doanh, nab, nghiệp, nhập, phí, thu, thuế, động]; tbl_32fd657cb61d67d8b2ae6a5b17735e83e71a1b4f1aa3234fd2e0dda66dcf27b8=6.205923 [2019, 2020, nab, nhập, thu]; tbl_85a238500b92c47a0379c1f9dffd510e6818b128ecfd23f6df02f2ed389640bc=6.205923 [2019, 2020, nab, nhập, thu]; tbl_160897096825b92e1109ea0f87da4d0018a18834f5f8a5d236087f44c7e13e69=4.754332 [2019, 2020, nab, thuế]; tbl_9ff265a0e9dc856d6e4e515114bbadb0b2bb3fd4c4e80a37d10aaf0b1dc69f89=4.752300 [2019, 2020, nab, thuế]; tbl_d1bbddc89d0ae6998d635cfd49ff3cb8fba66ec1d6ce38bda2c2b4454ea5f0c6=3.943928 [2020, nab, thuế]; tbl_6b3f05fb39f9ba9b73aeb27fb2e66bd6545d2149934273ae1dda196443ef57b8=3.856236 [2020, nab, thuế]

### retq_f4d7995ffd78410b783739fab33df211f00ce268cea4a5403c2b33e0ccb7832f

- Question: Tra cứu tổng tài sản và các khoản lãi, phí phải thu của MSB cuối năm 2020.
- Intent: lookup
- Failure: none
- Predicted table IDs: tbl_5e760f5b7e2baaabf6152d51c336bea64333a4a9ed4c29cbd79fffab4c6c2c39, tbl_64b0af2b0f97895188a2ec0cdab36c18855e93d3d057f27094705a9dcf863d23, tbl_4aa032ad779978b8876db6b3ef96c5243931183424507a936a88542ea8d02faa, tbl_700b3ea304cba23eb8491b0ac45f37d3ba83ef252590aca3e8359212d101638f, tbl_f8289c3b97e3810d9971578ca83eb829ae852e05af5cff67a6615f02e94cb7f2, tbl_fe5d016f6b3bbada2866e8a64e0315a968561ebabe60be3faca84485642ebe7d, tbl_b5fd646ae4b634e2fbc6a0f3aee47ca25c84de12274bf98d0f7389d0bc858e82, tbl_1c0d5cc09bbbcdeada8c20257159d7789266223d4b34e4643e9e6dab1d76fae9, tbl_7e7641eddb7496c2140b210909c0f8fc4ee8535055b69b96e624927061622a69, tbl_d856f4b499d8f43bd6fd2331cd3dc81214884c7197e1fed5b2d97037a53548b3
- Gold table IDs: tbl_64b0af2b0f97895188a2ec0cdab36c18855e93d3d057f27094705a9dcf863d23
- Missing gold table IDs: (none)
- Eligible documents: 14
- Empty reason: (none)
- Filter counts: company_codes=1971/1971; periods=17285/252; statement_types=7745/14
- Scores and matched tokens: tbl_5e760f5b7e2baaabf6152d51c336bea64333a4a9ed4c29cbd79fffab4c6c2c39=10.173290 [2020, assets, các, khoản, msb, phải, sản, thu, total, tài, tổng]; tbl_64b0af2b0f97895188a2ec0cdab36c18855e93d3d057f27094705a9dcf863d23=10.173290 [2020, assets, các, khoản, msb, phải, sản, thu, total, tài, tổng]; tbl_4aa032ad779978b8876db6b3ef96c5243931183424507a936a88542ea8d02faa=6.850955 [2020, của, msb, năm, phải, sản, total, tài]; tbl_700b3ea304cba23eb8491b0ac45f37d3ba83ef252590aca3e8359212d101638f=6.850955 [2020, của, msb, năm, phải, sản, total, tài]; tbl_f8289c3b97e3810d9971578ca83eb829ae852e05af5cff67a6615f02e94cb7f2=6.850955 [2020, của, msb, năm, phải, sản, total, tài]; tbl_fe5d016f6b3bbada2866e8a64e0315a968561ebabe60be3faca84485642ebe7d=6.142357 [2020, msb, phải, sản, total, tài]; tbl_b5fd646ae4b634e2fbc6a0f3aee47ca25c84de12274bf98d0f7389d0bc858e82=5.501192 [2020, của, msb, năm, phải, total]; tbl_1c0d5cc09bbbcdeada8c20257159d7789266223d4b34e4643e9e6dab1d76fae9=5.314177 [2020, msb, năm, phải, total]; tbl_7e7641eddb7496c2140b210909c0f8fc4ee8535055b69b96e624927061622a69=5.227680 [2020, msb, phải, sản, tài]; tbl_d856f4b499d8f43bd6fd2331cd3dc81214884c7197e1fed5b2d97037a53548b3=4.645851 [2020, msb, phải, total]

