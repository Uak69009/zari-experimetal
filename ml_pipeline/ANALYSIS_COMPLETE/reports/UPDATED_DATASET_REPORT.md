# Updated Dataset Report

## Overview
- Total rows: 142600
- Total classes: 153
- Healthy rows: 18032
- Healthy classes: 13
- Unknown rows: 377 rows in class `Unknown_20`, plus 21749 rows with `Unknown` crop labels

## Dataset Sources
| Source dataset | Rows |
| --- | --- |
| plantvillage | 73604 |
| plantcity | 52273 |
| nwrd | 14150 |
| plantdoc | 2573 |

## Data Types
| Domain | Rows |
| --- | --- |
| Lab | 73604 |
| Mixed | 52273 |
| Field | 16723 |

| Annotation type | Rows |
| --- | --- |
| classification | 142600 |

| Split | Rows |
| --- | --- |
| train | 115753 |
| test | 13632 |
| val | 13215 |

## Pathogen Type Breakdown
| Pathogen type | Rows |
| --- | --- |
| Unknown | 54080 |
| Fungal | 49611 |
| Healthy | 18032 |
| Viral | 9124 |
| Bacterial | 7236 |
| Pest | 4517 |

## Crop Breakdown
| Crop | Rows |
| --- | --- |
| Tomato | 34686 |
| Unknown | 21749 |
| Wheat | 14150 |
| Grape | 12076 |
| Cherry | 6958 |
| Corn | 6549 |
| Apple | 6370 |
| Orange | 5507 |
| Soybean | 5155 |
| Walnut | 5030 |
| Bean | 3207 |
| Fig | 2864 |
| Peach | 2768 |
| Pepper | 2608 |
| Pear | 2560 |
| Potato | 2373 |
| Apricot | 2257 |
| Squash | 1965 |
| Strawberry | 1661 |
| Blueberry | 1617 |
| Raspberry | 490 |

## Class Breakdown
| Class | Crop | Disease | Pathogen type | Source dataset | Domain | Rows | Train | Val | Test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Orange_Haunglongbing_Greening | Orange | Haunglongbing_Greening | Unknown | plantvillage | Lab | 5507 | 4405 | 559 | 543 |
| Tomato_Yellow_Curl_Virus | Tomato | Yellow_Curl_Virus | Viral | plantvillage | Lab | 5357 | 4278 | 551 | 528 |
| Soybean_Healthy | Soybean | Healthy | Healthy | plantvillage | Lab | 5090 | 4016 | 533 | 541 |
| Tomato_Bacterial_Spot | Tomato | Bacterial_Spot | Bacterial | plantvillage | Lab | 3942 | 3146 | 394 | 402 |
| Tomato_Healthy | Tomato | Healthy | Healthy | plantcity | Mixed | 3469 | 2786 | 341 | 342 |
| Tomato_Late_Blight | Tomato | Late_Blight | Fungal | plantvillage | Lab | 3426 | 2741 | 347 | 338 |
| Tomato_Early_Blight | Tomato | Early_Blight | Fungal | plantcity | Mixed | 3013 | 2406 | 311 | 296 |
| Tomato_Mold | Tomato | Mold | Fungal | plantcity | Mixed | 2953 | 2334 | 310 | 309 |
| Peach_Bacterial_Spot | Peach | Bacterial_Spot | Bacterial | plantvillage | Lab | 2297 | 1829 | 250 | 218 |
| Unknown_15 | Unknown | 15 | Unknown | plantvillage | Lab | 2013 | 1610 | 202 | 201 |
| Unknown_35 | Unknown | 35 | Unknown | plantvillage | Lab | 2000 | 1600 | 200 | 200 |
| Squash_Powdery_Mildew | Squash | Powdery_Mildew | Fungal | plantvillage | Lab | 1965 | 1573 | 193 | 199 |
| Grape_Unknown | Grape | Unknown | Unknown | plantcity | Mixed | 1927 | 1542 | 193 | 192 |
| Tomato_Septoria_Spot | Tomato | Septoria_Spot | Fungal | plantvillage | Lab | 1922 | 1524 | 190 | 208 |
| Tomato_Miner | Tomato | Miner | Pest | plantcity | Mixed | 1897 | 1518 | 189 | 190 |
| Tomato_Curl | Tomato | Curl | Viral | plantcity | Mixed | 1893 | 1514 | 189 | 190 |
| Unknown_24 | Unknown | 24 | Unknown | plantvillage | Lab | 1784 | 1427 | 179 | 178 |
| Walnut_Blotch | Walnut | Blotch | Unknown | plantcity | Mixed | 1693 | 1354 | 170 | 169 |
| Tomato_Spider_Mites_Two_Spotted_Spider_Mite | Tomato | Spider_Mites_Two_Spotted_Spider_Mite | Fungal | plantvillage | Lab | 1676 | 1344 | 169 | 163 |
| Apple_Healthy | Apple | Healthy | Healthy | plantvillage | Lab | 1645 | 1321 | 151 | 173 |
| Tomato_Septoria | Tomato | Septoria | Unknown | plantcity | Mixed | 1599 | 1279 | 160 | 160 |
| Grape_Powdery_Mildew | Grape | Powdery_Mildew | Fungal | plantcity | Mixed | 1509 | 1207 | 151 | 151 |
| Blueberry_Healthy | Blueberry | Healthy | Healthy | plantvillage | Lab | 1502 | 1182 | 164 | 156 |
| Pepper_Healthy | Pepper | Healthy | Healthy | plantvillage | Lab | 1478 | 1197 | 148 | 133 |
| Tomato_Target_Spot | Tomato | Target_Spot | Fungal | plantvillage | Lab | 1404 | 1109 | 154 | 141 |
| Cherry_Scorch | Cherry | Scorch | Unknown | plantcity | Mixed | 1383 | 1106 | 138 | 139 |
| Grape_Esca_Black_Measles | Grape | Esca_Black_Measles | Unknown | plantvillage | Lab | 1383 | 1096 | 142 | 145 |
| Wheat_Smut | Wheat | Smut | Fungal | nwrd | Field | 1380 | 1310 | 20 | 50 |
| Wheat_Yellow_Rust | Wheat | Yellow_Rust | Viral | nwrd | Field | 1371 | 1301 | 20 | 50 |
| Wheat_Brown_Rust | Wheat | Brown_Rust | Fungal | nwrd | Field | 1341 | 1271 | 20 | 50 |
| Apple_Brown_Spot | Apple | Brown_Spot | Fungal | plantcity | Mixed | 1325 | 1060 | 132 | 133 |
| Grape_Anthracnose | Grape | Anthracnose | Unknown | plantcity | Mixed | 1245 | 996 | 124 | 125 |
| Grape_Black_Rot | Grape | Black_Rot | Fungal | plantvillage | Lab | 1244 | 995 | 128 | 121 |
| Cherry_Purple_Spot | Cherry | Purple_Spot | Fungal | plantcity | Mixed | 1227 | 982 | 122 | 123 |
| Wheat_Septoria | Wheat | Septoria | Unknown | nwrd | Field | 1214 | 1144 | 20 | 50 |
| Cherry_Brown_Spot | Cherry | Brown_Spot | Fungal | plantcity | Mixed | 1210 | 968 | 121 | 121 |
| Corn_Common_Rust | Corn | Common_Rust | Fungal | plantvillage | Lab | 1192 | 963 | 110 | 119 |
| Corn_Healthy | Corn | Healthy | Healthy | plantvillage | Lab | 1162 | 913 | 123 | 126 |
| Wheat_Mildew | Wheat | Mildew | Fungal | nwrd | Field | 1151 | 1081 | 20 | 50 |
| Walnut_Shot_Hole | Walnut | Shot_Hole | Unknown | plantcity | Mixed | 1147 | 918 | 114 | 115 |
| Pear_Black_Spot | Pear | Black_Spot | Fungal | plantcity | Mixed | 1130 | 904 | 113 | 113 |
| Bean_Unknown | Bean | Unknown | Unknown | plantcity | Mixed | 1123 | 898 | 112 | 113 |
| Apple_Unknown | Apple | Unknown | Unknown | plantcity | Mixed | 1121 | 897 | 112 | 112 |
| Potato_Early_Blight | Potato | Early_Blight | Fungal | plantvillage | Lab | 1116 | 883 | 108 | 125 |
| Strawberry_Scorch | Strawberry | Scorch | Unknown | plantvillage | Lab | 1109 | 894 | 101 | 114 |
| Potato_Late_Blight | Potato | Late_Blight | Fungal | plantvillage | Lab | 1105 | 879 | 112 | 114 |
| Grape_Blight_Isariopsis_Spot | Grape | Blight_Isariopsis_Spot | Fungal | plantvillage | Lab | 1076 | 854 | 104 | 118 |
| Wheat_Healthy | Wheat | Healthy | Healthy | nwrd | Field | 1070 | 1000 | 20 | 50 |
| Grape_Shot_Hole | Grape | Shot_Hole | Unknown | plantcity | Mixed | 1065 | 852 | 106 | 107 |
| Cherry_Powdery_Mildew | Cherry | Powdery_Mildew | Fungal | plantvillage | Lab | 1052 | 842 | 112 | 98 |
| Apricot_Shot_Hole | Apricot | Shot_Hole | Unknown | plantcity | Mixed | 1040 | 832 | 104 | 104 |
| Pepper_Bacterial_Spot | Pepper | Bacterial_Spot | Bacterial | plantvillage | Lab | 997 | 810 | 96 | 91 |
| Unknown_Lokat_Spot | Unknown | Lokat_Spot | Fungal | plantcity | Mixed | 990 | 792 | 99 | 99 |
| Corn_Northern_Blight | Corn | Northern_Blight | Fungal | plantvillage | Lab | 985 | 799 | 92 | 94 |
| Wheat_Aphid | Wheat | Aphid | Unknown | nwrd | Field | 973 | 903 | 20 | 50 |
| Bean_Fungal | Bean | Fungal | Unknown | plantcity | Mixed | 970 | 776 | 97 | 97 |
| Pear_Unknown | Pear | Unknown | Unknown | plantcity | Mixed | 955 | 764 | 96 | 95 |
| Fig_Blight | Fig | Blight | Fungal | plantcity | Mixed | 954 | 763 | 96 | 95 |
| Wheat_Leaf_Blight | Wheat | Leaf_Blight | Fungal | nwrd | Field | 912 | 842 | 20 | 50 |
| Walnut_Unknown | Walnut | Unknown | Unknown | plantcity | Mixed | 890 | 712 | 89 | 89 |
| Wheat_Mite | Wheat | Mite | Pest | nwrd | Field | 870 | 800 | 20 | 50 |
| Cherry_Healthy | Cherry | Healthy | Healthy | plantvillage | Lab | 854 | 687 | 88 | 79 |
| Wheat_Tan_Spot | Wheat | Tan_Spot | Fungal | nwrd | Field | 840 | 770 | 20 | 50 |
| Unknown_Persimmons_Brown_Spot | Unknown | Persimmons_Brown_Spot | Fungal | plantcity | Mixed | 829 | 663 | 83 | 83 |
| Unknown_16 | Unknown | 16 | Unknown | plantvillage | Lab | 826 | 661 | 82 | 83 |
| Apricot_Unknown | Apricot | Unknown | Unknown | plantcity | Mixed | 815 | 652 | 82 | 81 |
| Walnut_Anthracnose | Walnut | Anthracnose | Unknown | plantcity | Mixed | 810 | 648 | 81 | 81 |
| Grape_Brown_Spot | Grape | Brown_Spot | Fungal | plantcity | Mixed | 800 | 640 | 80 | 80 |
| Grape_Downy_Mildew | Grape | Downy_Mildew | Fungal | plantcity | Mixed | 779 | 623 | 78 | 78 |
| Unknown_28 | Unknown | 28 | Unknown | plantvillage | Lab | 760 | 608 | 76 | 76 |
| Fig_Unknown | Fig | Unknown | Unknown | plantcity | Mixed | 750 | 600 | 75 | 75 |
| Corn_Gray_Spot | Corn | Gray_Spot | Fungal | plantcity | Mixed | 728 | 582 | 73 | 73 |
| Apple_Scab | Apple | Scab | Fungal | plantvillage | Lab | 724 | 581 | 69 | 74 |
| Wheat_Blast | Wheat | Blast | Unknown | nwrd | Field | 717 | 647 | 20 | 50 |
| Corn_Unknown | Corn | Unknown | Unknown | plantcity | Mixed | 710 | 568 | 71 | 71 |
| Wheat_Common_Root_Rot | Wheat | Common_Root_Rot | Fungal | nwrd | Field | 684 | 614 | 20 | 50 |
| Cherry_Unknown | Cherry | Unknown | Unknown | plantcity | Mixed | 682 | 546 | 68 | 68 |
| Wheat_Fusarium_Head_Blight | Wheat | Fusarium_Head_Blight | Fungal | nwrd | Field | 681 | 611 | 20 | 50 |
| Unknown_30 | Unknown | 30 | Unknown | plantvillage | Lab | 660 | 528 | 66 | 66 |
| Unknown_25 | Unknown | 25 | Unknown | plantvillage | Lab | 649 | 519 | 65 | 65 |
| Wheat_Black_Rust | Wheat | Black_Rust | Fungal | nwrd | Field | 642 | 572 | 20 | 50 |
| Unknown_33 | Unknown | 33 | Unknown | plantvillage | Lab | 640 | 512 | 64 | 64 |
| Tomato_Spider_Mites | Tomato | Spider_Mites | Pest | plantcity | Mixed | 635 | 508 | 64 | 63 |
| Unknown_Lokat | Unknown | Lokat | Unknown | plantcity | Mixed | 630 | 504 | 63 | 63 |
| Fig_Rust | Fig | Rust | Fungal | plantcity | Mixed | 625 | 500 | 63 | 62 |
| Grape_Mites | Grape | Mites | Pest | plantcity | Mixed | 625 | 500 | 62 | 63 |
| Bean_Rust | Bean | Rust | Fungal | plantcity | Mixed | 624 | 499 | 63 | 62 |
| Apple_Black_Rot | Apple | Black_Rot | Fungal | plantvillage | Lab | 621 | 511 | 56 | 54 |
| Unknown_32 | Unknown | 32 | Unknown | plantvillage | Lab | 587 | 470 | 58 | 59 |
| Apple_Black_Spot | Apple | Black_Spot | Fungal | plantcity | Mixed | 571 | 457 | 57 | 57 |
| Unknown_37 | Unknown | 37 | Unknown | plantvillage | Lab | 560 | 448 | 56 | 56 |
| Unknown_3 | Unknown | 3 | Unknown | plantvillage | Lab | 554 | 443 | 56 | 55 |
| Cherry_Shot_Hole | Cherry | Shot_Hole | Unknown | plantcity | Mixed | 550 | 440 | 55 | 55 |
| Unknown_19 | Unknown | 19 | Unknown | plantvillage | Lab | 544 | 435 | 54 | 55 |
| Corn_Holcus_Spot | Corn | Holcus_Spot | Fungal | plantcity | Mixed | 540 | 432 | 54 | 54 |
| Fig_Brown_Spot | Fig | Brown_Spot | Fungal | plantcity | Mixed | 535 | 428 | 53 | 54 |
| Tomato_Verticillium_Wilt | Tomato | Verticillium_Wilt | Unknown | plantcity | Mixed | 525 | 420 | 52 | 53 |
| Corn_Cercospora_Spot_Gray_Spot | Corn | Cercospora_Spot_Gray_Spot | Fungal | plantvillage | Lab | 513 | 418 | 44 | 51 |
| Bean_Shot_Hole | Bean | Shot_Hole | Unknown | plantcity | Mixed | 490 | 392 | 49 | 49 |
| Walnut_Gall_Mite | Walnut | Gall_Mite | Pest | plantcity | Mixed | 490 | 392 | 49 | 49 |
| Pear_Fire_Blight | Pear | Fire_Blight | Fungal | plantcity | Mixed | 475 | 380 | 47 | 48 |
| Unknown_34 | Unknown | 34 | Unknown | plantvillage | Lab | 468 | 374 | 47 | 47 |
| Unknown_4 | Unknown | 4 | Unknown | plantvillage | Lab | 468 | 374 | 47 | 47 |
| Unknown_12 | Unknown | 12 | Unknown | plantvillage | Lab | 465 | 372 | 46 | 47 |
| Strawberry_Healthy | Strawberry | Healthy | Healthy | plantvillage | Lab | 456 | 368 | 42 | 46 |
| Unknown_10 | Unknown | 10 | Unknown | plantvillage | Lab | 433 | 346 | 44 | 43 |
| Unknown_8 | Unknown | 8 | Unknown | plantvillage | Lab | 432 | 346 | 43 | 43 |
| Tomato_Mosaic_Virus | Tomato | Mosaic_Virus | Viral | plantvillage | Lab | 427 | 343 | 35 | 49 |
| Grape_Healthy | Grape | Healthy | Healthy | plantvillage | Lab | 423 | 335 | 51 | 37 |
| Corn_Fungal | Corn | Fungal | Unknown | plantcity | Mixed | 415 | 332 | 42 | 41 |
| Unknown_5 | Unknown | 5 | Unknown | plantvillage | Lab | 409 | 327 | 41 | 41 |
| Tomato_Fusarium_Wilt | Tomato | Fusarium_Wilt | Unknown | plantcity | Mixed | 407 | 326 | 41 | 40 |
| Unknown_11 | Unknown | 11 | Unknown | plantvillage | Lab | 407 | 326 | 40 | 41 |
| Apricot_Blight | Apricot | Blight | Fungal | plantcity | Mixed | 402 | 322 | 40 | 40 |
| Unknown_26 | Unknown | 26 | Unknown | plantvillage | Lab | 380 | 304 | 38 | 38 |
| Unknown_13 | Unknown | 13 | Unknown | plantvillage | Lab | 379 | 303 | 38 | 38 |
| Unknown_20 | Unknown | 20 | Unknown | plantvillage | Lab | 377 | 302 | 38 | 37 |
| Unknown_21 | Unknown | 21 | Unknown | plantvillage | Lab | 372 | 298 | 37 | 37 |
| Raspberry_Healthy | Raspberry | Healthy | Healthy | plantvillage | Lab | 371 | 302 | 33 | 36 |
| Peach_Healthy | Peach | Healthy | Healthy | plantvillage | Lab | 360 | 281 | 38 | 41 |
| Unknown_9 | Unknown | 9 | Unknown | plantvillage | Lab | 354 | 283 | 36 | 35 |
| Unknown_6 | Unknown | 6 | Unknown | plantvillage | Lab | 334 | 267 | 34 | 33 |
| Unknown_29 | Unknown | 29 | Unknown | plantvillage | Lab | 331 | 265 | 33 | 33 |
| Wheat_Stem_Fly | Wheat | Stem_Fly | Unknown | nwrd | Field | 304 | 234 | 20 | 50 |
| Unknown_18 | Unknown | 18 | Unknown | plantvillage | Lab | 287 | 230 | 29 | 28 |
| Unknown_31 | Unknown | 31 | Unknown | plantvillage | Lab | 286 | 229 | 29 | 28 |
| Apple_Cedar_Rust | Apple | Cedar_Rust | Fungal | plantvillage | Lab | 275 | 221 | 26 | 28 |
| Unknown_23 | Unknown | 23 | Unknown | plantvillage | Lab | 224 | 179 | 22 | 23 |
| Unknown_0 | Unknown | 0 | Unknown | plantvillage | Lab | 221 | 177 | 22 | 22 |
| Unknown_1 | Unknown | 1 | Unknown | plantvillage | Lab | 199 | 159 | 20 | 20 |
| Corn_Blight | Corn | Blight | Fungal | plantdoc | Field | 188 | 150 | 19 | 19 |
| Unknown_7 | Unknown | 7 | Unknown | plantvillage | Lab | 187 | 150 | 19 | 18 |
| Unknown_27 | Unknown | 27 | Unknown | plantvillage | Lab | 184 | 147 | 19 | 18 |
| Potato_Healthy | Potato | Healthy | Healthy | plantvillage | Lab | 152 | 121 | 19 | 12 |
| Unknown_14 | Unknown | 14 | Unknown | plantvillage | Lab | 146 | 117 | 14 | 15 |
| Unknown_36 | Unknown | 36 | Unknown | plantvillage | Lab | 143 | 114 | 14 | 15 |
| Raspberry_Unknown | Raspberry | Unknown | Unknown | plantdoc | Field | 119 | 95 | 12 | 12 |
| Corn_Rust | Corn | Rust | Fungal | plantdoc | Field | 116 | 93 | 12 | 11 |
| Blueberry_Unknown | Blueberry | Unknown | Unknown | plantdoc | Field | 115 | 92 | 12 | 11 |
| Peach_Unknown | Peach | Unknown | Unknown | plantdoc | Field | 111 | 89 | 11 | 11 |
| Unknown_2 | Unknown | 2 | Unknown | plantvillage | Lab | 99 | 79 | 10 | 10 |
| Strawberry_Unknown | Strawberry | Unknown | Unknown | plantdoc | Field | 96 | 77 | 10 | 9 |
| Unknown_17 | Unknown | 17 | Unknown | plantvillage | Lab | 95 | 76 | 10 | 9 |
| Apple_Rust | Apple | Rust | Fungal | plantdoc | Field | 88 | 70 | 9 | 9 |
| Tomato_Yellow_Virus | Tomato | Yellow_Virus | Viral | plantdoc | Field | 76 | 61 | 7 | 8 |
| Pepper_Spot | Pepper | Spot | Fungal | plantdoc | Field | 71 | 57 | 7 | 7 |
| Soybean_Unknown | Soybean | Unknown | Unknown | plantdoc | Field | 65 | 52 | 6 | 7 |
| Tomato_Unknown | Tomato | Unknown | Unknown | plantdoc | Field | 63 | 50 | 6 | 7 |
| Pepper_Unknown | Pepper | Unknown | Unknown | plantdoc | Field | 62 | 50 | 6 | 6 |
| Unknown_22 | Unknown | 22 | Unknown | plantvillage | Lab | 41 | 33 | 4 | 4 |
| Tomato_Two_Spotted_Spider_Mites | Tomato | Two_Spotted_Spider_Mites | Fungal | plantdoc | Field | 2 | 2 | 0 | 0 |
| Unknown_Generated_For_Paper | Unknown | Generated_For_Paper | Unknown | plantvillage | Lab | 1 | 1 | 0 | 0 |
| Unknown_Plantdoc | Unknown | Plantdoc | Unknown | plantdoc | Field | 1 | 1 | 0 | 0 |

## Healthy Classes
| Healthy class | Rows |
| --- | --- |
| Soybean_Healthy | 5090 |
| Tomato_Healthy | 3469 |
| Apple_Healthy | 1645 |
| Blueberry_Healthy | 1502 |
| Pepper_Healthy | 1478 |
| Corn_Healthy | 1162 |
| Wheat_Healthy | 1070 |
| Cherry_Healthy | 854 |
| Strawberry_Healthy | 456 |
| Grape_Healthy | 423 |
| Raspberry_Healthy | 371 |
| Peach_Healthy | 360 |
| Potato_Healthy | 152 |

## CSV Consistency Notes
- The enriched CSV has 142600 rows; the clean analysis CSV has 142596 rows.
- Rows present in the enriched CSV but missing from the clean CSV: 4
  - /home/hammad/Desktop/project zari - experimental/ml_pipeline/data/raw/plantdoc/PlantDoc_Examples.png
  - /home/hammad/Desktop/project zari - experimental/ml_pipeline/data/raw/plantdoc/train/Tomato two spotted spider mites leaf/SpotSpeckBlightMite-1l4v879.jpg
  - /home/hammad/Desktop/project zari - experimental/ml_pipeline/data/raw/plantdoc/train/Tomato two spotted spider mites leaf/comparing-diseases-4-canker-tomato-1y51ejs.jpg
  - /home/hammad/Desktop/project zari - experimental/ml_pipeline/data/raw/plantvillage/generated_for_paper/plantvillage.jpg

## Key Observations
- The dataset is now fully Linux-path based in the refreshed CSVs.
- The deleted grayscale and segmented folders are no longer referenced in the refreshed master CSVs.
- PlantVillage remains the largest source, with PlantCity as the next largest source, followed by NWRD and PlantDoc.
- Healthy samples are concentrated in Soybean, Tomato, Apple, Pepper, and Blueberry.
- The class set is still highly imbalanced, so model training should continue to use class weighting or balanced sampling.