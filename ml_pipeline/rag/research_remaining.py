"""ZARI.ai — Remaining Crop Evidence Research & Knowledgebase Chunking Engine.

Gathers evidence and generates RAG chunks for all remaining 41 classes across:
- Grape (7 classes: Cornell Grape Program, OIV)
- Apple (3 classes: WSU, APS)
- Pear (3 classes: APS, Cornell)
- Stone Fruits: Cherry & Apricot (8 classes: APS, UC IPM)
- Corn (4 classes: CIMMYT, Purdue Extension)
- Bean (4 classes: CABI, CIAT)
- Fig (4 classes: CABI)
- Walnut (5 classes: UC IPM, CABI)
- Persimmon (1 class: CABI)
- Lokat (2 classes: CABI)

SPECIAL RULES:
- UNKNOWN CLASSES (9 remaining unknown classes): Do NOT create treatment chunks.
  Create ONE chunk: "This classification is uncertain. Request clearer image."
- HEALTHY CLASSES (e.g. Lokat_Healthy):
  Create maintenance chunks: "This crop appears healthy. Maintain with..."

Output:
- ml_pipeline/data/chunks_remaining.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
OUTPUT_JSON = DATA_DIR / "chunks_remaining.json"

# List of Unknown Classes in 67 Head Registry
UNKNOWN_CLASSES = {
    "Apple_Unknown", "Apricot_Unknown", "Bean_Unknown", "Cherry_Unknown",
    "Corn_Unknown", "Fig_Unknown", "Grape_Unknown", "Pear_Unknown", "Walnut_Unknown"
}

# Master Evidence Database for Remaining Classes
REMAINING_EVIDENCE_DATA: dict[str, dict] = {
    # -------------------------------------------------------------------------
    # GRAPE (7 Classes)
    # -------------------------------------------------------------------------
    "Grape_Anthracnose": {
        "crop": "Grape",
        "scientific_name": "Elsinoë ampelina",
        "pathogen_type": "Fungal",
        "common_name": {"en": "Grape Anthracnose / Bird's Eye Rot", "ur": "انگور کا اینتھراکنوز", "ps": "د انګورو انترکنوز"},
        "identity": "Grape_Anthracnose is caused by Elsinoë ampelina. Common names: Anthracnose / Bird's Eye Rot (en), انگور کا اینتھراکنوز (ur), د انګورو انترکنوز (ps). Severe foliar and fruit disease of grapevine.",
        "symptoms": "Circular sunken spots on shoots and fruits with gray centers and reddish-brown to dark purple raised margins ('bird's eye' pattern). Shot-hole lesions on leaves.",
        "epidemiology": "Favored by warm wet weather during early shoot growth. Conidia spread by rain splash.",
        "cultural_control": "Prune out infected canes during dormant winter season. Destroy cane prunings. Improve canopy airflow.",
        "biological_control": "Apply Trichoderma harzianum or Copper bio-rational sprays.",
        "chemical_control": "Apply Liquid Lime Sulfur during dormant stage, followed by Copper Hydroxide or Mancozeb sprays at early shoot growth. Check product label for rates.",
        "prevention": "Dormant cane sanitation, canopy pruning, preventive copper sprays.",
        "safety": "Wear eye protection when spraying lime sulfur or copper.",
        "pakistan": "Common in grape orchards of KP (Swat, Peshawar) and Balochistan (Quetta, Pishin).",
        "sources": "Cornell Grape Pathology / OIV / CABI Plantwise.",
    },
    "Grape_Brown_Spot": {
        "crop": "Grape",
        "scientific_name": "Pseudocercospora vitis",
        "pathogen_type": "Fungal",
        "common_name": {"en": "Grape Leaf Spot / Brown Spot", "ur": "انگور کے بھورے دھبے", "ps": "د انګورو نسواري ټاپي"},
        "identity": "Grape_Brown_Spot is caused by Pseudocercospora vitis. Common names: Grape Leaf Spot (en), انگور کے بھورے دھبے (ur), د انګورو نسواري ټاپي (ps). Late-season foliar disease.",
        "symptoms": "Irregular dark brown lesions on leaves with dark borders. Severe leaf yellowing and early defoliation in late summer.",
        "epidemiology": "Favored by high humidity and warm temperatures late in the growing season.",
        "cultural_control": "Prune lower leaves, collect fallen leaf litter post-harvest, practice summer canopy management.",
        "biological_control": "Bacillus subtilis foliar spray.",
        "chemical_control": "Post-harvest or late summer Mancozeb or Copper Oxychloride spray.",
        "prevention": "Fall leaf cleanup, canopy management.",
        "safety": "Standard protective clothing during spraying.",
        "pakistan": "Occurs in rainfed grape vineyards in Punjab and KP.",
        "sources": "Cornell Grape Program / CABI Plantwise.",
    },
    "Grape_Downy_Mildew": {
        "crop": "Grape",
        "scientific_name": "Plasmopara viticola",
        "pathogen_type": "Fungal",
        "common_name": {"en": "Grape Downy Mildew", "ur": "انگور کا ڈاؤن ہی ملڈیو", "ps": "د انګورو ډاوني ملډیو"},
        "identity": "Grape_Downy_Mildew is caused by Plasmopara viticola. Common names: Downy Mildew (en), انگور کا ڈاؤن ہی ملڈیو (ur), د انګورو ډاوني ملډیو (ps). Highly destructive oomycete disease.",
        "symptoms": "Yellowish oily spots on upper leaf surface ('oil spots'). Dense white cottony downy growth on undersides of leaves during wet morning hours. Brown leathery fruit rot.",
        "epidemiology": "Favored by 10-10-10 rule (10mm rain, 10°C temp, 10cm shoot growth). Primary oospores overwinter in leaf litter.",
        "cultural_control": "Avoid overhead irrigation, prune lower leaves near ground, destroy fallen leaves.",
        "biological_control": "Copper hydroxide bio-rational spray.",
        "chemical_control": "Preventive Metalaxyl-M + Mancozeb, Cymoxanil, or Fosetyl-Al sprays prior to rain events. Check label for rates.",
        "prevention": "Scout after warm spring rains, apply preventive copper protectants.",
        "safety": "Wear PPE during spraying.",
        "pakistan": "Major disease in humid grape growing areas of KP and Punjab.",
        "sources": "Cornell Grape Pathology / OIV.",
    },
    "Grape_Mites": {
        "crop": "Grape",
        "scientific_name": "Colomerus vitis",
        "pathogen_type": "Pest",
        "common_name": {"en": "Grape Erinose Mite", "ur": "انگور کی مائٹس", "ps": "د انګورو ژوي یا مایټس"},
        "identity": "Grape_Mites is caused by Colomerus vitis (Grape Erinose Mite). Common names: Grape Blister Mite (en), انگور کی مائٹس (ur), د انګورو مایټس (ps). PEST: Do NOT use fungicides.",
        "symptoms": "Upper leaf surface bulges into prominent green galls/blisters. White to reddish felt-like dense hair patches (erineum) inside cavities on lower leaf underside.",
        "epidemiology": "Overwinters under grape bud scales. Mites feed inside leaf tissues during spring growth.",
        "cultural_control": "Prune out heavily infested shoots in spring.",
        "biological_control": "Release predatory phytoseiid mites. Apply Neem oil spray.",
        "chemical_control": "Apply Liquid Lime Sulfur or Wettable Sulfur (FRAC M02) during bud break. Acaricide spray if severe.",
        "prevention": "Dormant sulfur application at bud burst.",
        "safety": "Wear eye protection during sulfur application.",
        "pakistan": "Common in grape orchards across Balochistan (Pishin, Mastung) and KP.",
        "sources": "UC IPM Grape Pest Management / CABI.",
    },
    "Grape_Powdery_Mildew": {
        "crop": "Grape",
        "scientific_name": "Erysiphe necator",
        "pathogen_type": "Fungal",
        "common_name": {"en": "Grape Powdery Mildew", "ur": "انگور کا پاؤڈری ملڈیو", "ps": "د انګورو پاوډري ملډیو"},
        "identity": "Grape_Powdery_Mildew is caused by Erysiphe necator. Common names: Powdery Mildew (en), انگور کا پاؤڈری ملڈیو (ur), د انګورو پاوډري ملډیو (ps). Primary foliar and berry disease.",
        "symptoms": "White to gray dusty powdery patches on leaves, green shoots, and berries. Infected berries split, dry up, or fail to ripen properly. Musty odor in vineyard.",
        "epidemiology": "Favored by dry warm weather (20-28°C) with low sunlight (shaded canopy). High humidity promotes spore formation.",
        "cultural_control": "Open canopy pruning to increase sunlight penetration and air circulation. Remove infected shoots early.",
        "biological_control": "Spray Potassium Bicarbonate, Neem oil (1-2%), or Ampelomyces quisqualis bio-fungicide.",
        "chemical_control": "Apply Sulfur (FRAC M02), Tebuconazole (FRAC 3), or Azoxystrobin (FRAC 11) from pre-bloom to fruit set.",
        "prevention": "Canopy management, early sulfur dusting.",
        "safety": "Avoid sulfur application when ambient temperature exceeds 32°C to prevent fruit burn.",
        "pakistan": "Widespread across all grape-growing districts of Balochistan and KP.",
        "sources": "Cornell Grape Pathology / OIV.",
    },
    "Grape_Shot_Hole": {
        "crop": "Grape",
        "scientific_name": "Phyllosticta ampelicida",
        "pathogen_type": "Fungal",
        "common_name": {"en": "Grape Shot Hole / Black Rot Spot", "ur": "انگور کے پتے کے سوراخ", "ps": "د انګورو سوري لرونکي ټاپي"},
        "identity": "Grape_Shot_Hole is caused by Phyllosticta ampelicida (teleomorph Guignardia bidwellii). Common names: Black Rot Leaf Spot (en), انگور کے پتے کے سوراخ (ur), د انګورو سوري لرونکي ټاپي (ps).",
        "symptoms": "Small reddish-brown circular spots on leaves with dark brown borders. Center of spots drop out leaving shot-holes.",
        "epidemiology": "Favored by warm spring rain. Pycnidia overwinter in mummified berries and cane lesions.",
        "cultural_control": "Prune out mummified berries and dead wood in winter.",
        "biological_control": "Bacillus subtilis foliar application.",
        "chemical_control": "Mancozeb or Difenoconazole spray at early shoot growth.",
        "prevention": "Sanitation pruning, preventive spring protectant spray.",
        "safety": "Standard protective gear during spray.",
        "pakistan": "Occurs in upland grape vineyards in KP and Balochistan.",
        "sources": "Cornell Grape Program / APS.",
    },

    # -------------------------------------------------------------------------
    # APPLE (2 Diseases)
    # -------------------------------------------------------------------------
    "Apple_Black_Spot": {
        "crop": "Apple",
        "scientific_name": "Venturia inaequalis",
        "pathogen_type": "Fungal",
        "common_name": {"en": "Apple Black Spot / Scab", "ur": "سیب کے سیاہ دھبے / اسکایب", "ps": "د مڼو تور ټاپي"},
        "identity": "Apple_Black_Spot is caused by Venturia inaequalis. Common names: Apple Scab / Black Spot (en), سیب کے سیاہ دھبے (ur), د مڼو تور ټاپي (ps). Major apple disease.",
        "symptoms": "Olive-green to black velvety spots on leaves and fruit. Leaf spots become corky and distorted. Fruit develops dark scabby cracks.",
        "epidemiology": "Ascospores released from fallen leaves during spring rain events (Mills infection period).",
        "cultural_control": "Shred or plow fallen leaves in autumn to reduce ascospore carryover. Prune trees for open canopy airflow.",
        "biological_control": "Bio-spray of Bacillus subtilis or Neem extract.",
        "chemical_control": "Captan, Mancozeb (FRAC M03/M04) or Difenoconazole (FRAC 3) sprayed from green tip to petal fall stage.",
        "prevention": "Fall leaf sanitation, spring scab forecasting, open canopy pruning.",
        "safety": "Standard spray PPE.",
        "pakistan": "Major threat in commercial apple valleys of Balochistan (Kalat, Ziarat), KP (Swat), and Gilgit-Baltistan.",
        "sources": "WSU Apple Pathology / APS.",
    },
    "Apple_Brown_Spot": {
        "crop": "Apple",
        "scientific_name": "Marssonina coronaria",
        "pathogen_type": "Fungal",
        "common_name": {"en": "Marssonina Leaf Blotch / Brown Spot", "ur": "سیب کے بھورے دھبے", "ps": "د مڼو نسواري ټاپي"},
        "identity": "Apple_Brown_Spot is caused by Marssonina coronaria. Common names: Marssonina Blotch (en), سیب کے بھورے دھبے (ur), د مڼو نسواري ټاپي (ps). Late summer defoliator.",
        "symptoms": "Dark brown circular spots on leaves surrounded by yellowing chlorotic tissue. Severe summer defoliation leading to undersized fruit.",
        "epidemiology": "Favored by summer rainfall and high relative humidity (July-August).",
        "cultural_control": "Collect fallen leaf debris in autumn. Improve canopy drainage.",
        "biological_control": "Trichoderma foliar treatment.",
        "chemical_control": "Tebuconazole or Mancozeb foliar spray during summer monsoon period.",
        "prevention": "Summer leaf monitoring, post-harvest orchard floor sanitation.",
        "safety": "Standard protective clothing.",
        "pakistan": "Prevalent in high-rainfall apple zones of Swat, Murree, and GB.",
        "sources": "APS / CABI Plantwise.",
    },

    # -------------------------------------------------------------------------
    # PEAR (2 Diseases)
    # -------------------------------------------------------------------------
    "Pear_Black_Spot": {
        "crop": "Pear",
        "scientific_name": "Alternaria gaisen / Venturia pirina",
        "pathogen_type": "Fungal",
        "common_name": {"en": "Pear Black Spot / Scab", "ur": "ناشپاتی کے سیاہ دھبے", "ps": "د ناک تور ټاپي"},
        "identity": "Pear_Black_Spot is caused by Alternaria gaisen / Venturia pirina. Common names: Pear Black Spot (en), ناشپاتی کے سیاہ دھبے (ur), د ناک تور ټاپي (ps).",
        "symptoms": "Black velvety spots on pear leaves, petioles, and young fruit. Fruit lesions become corky and cracked.",
        "epidemiology": "Favored by cool moist spring weather. Spores spread by wind and rain splash.",
        "cultural_control": "Prune infected twigs in winter. Clean up orchard floor leaf litter.",
        "biological_control": "Bacillus subtilis bio-fungicide.",
        "chemical_control": "Mancozeb or Difenoconazole spray at bud break and petal fall.",
        "prevention": "Winter orchard sanitation, spring preventive sprays.",
        "safety": "Standard protective gear.",
        "pakistan": "Prevalent in pear orchards of KP (Peshawar, Hazara) and Punjab.",
        "sources": "Cornell Fruit IPM / APS.",
    },
    "Pear_Fire_Blight": {
        "crop": "Pear",
        "scientific_name": "Erwinia amylovora",
        "pathogen_type": "Bacterial",
        "common_name": {"en": "Fire Blight of Pear", "ur": "ناشپاتی کا فائر بلائٹ", "ps": "د ناک د اور سوځیدنه"},
        "identity": "Pear_Fire_Blight is caused by Erwinia amylovora. Common names: Fire Blight (en), ناشپاتی کا فائر بلائٹ (ur), د ناک د اور سوځیدنه (ps). Destructive bacterial disease.",
        "symptoms": "Blossoms and shoots suddenly wilt, blacken, and die, looking as if scorched by fire. Affected shoot tips curve into a characteristic 'shepherd's crook'. Bacterial ooze droplets under humid conditions.",
        "epidemiology": "Bacteria multiply in blossoms during warm spring weather (18-28°C) with rain/dew. Spread by bees and wind.",
        "cultural_control": "Prune out blighted strikes 30cm below visible margin during dry weather. Sanitize pruners between cuts with 70% alcohol.",
        "biological_control": "Apply Aureobasidium pullulans or Bacillus subtilis during bloom.",
        "chemical_control": "Apply Fixed Copper or Agricultural Streptomycin during bloom period. Check local registration rules.",
        "prevention": "Bloom temperature risk monitoring, immediate pruning of strikes, disinfectant tools.",
        "safety": "Sterilize tools with bleach or alcohol to prevent mechanical spread.",
        "pakistan": "Occurs in northern pear-growing valleys of KP and Gilgit-Baltistan.",
        "sources": "Cornell / APS / CABI Plantwise.",
    },

    # -------------------------------------------------------------------------
    # STONE FRUITS: CHERRY & APRICOT (6 Diseases)
    # -------------------------------------------------------------------------
    "Cherry_Brown_Spot": {
        "crop": "Cherry",
        "scientific_name": "Blumeriella jaapii",
        "pathogen_type": "Fungal",
        "common_name": {"en": "Cherry Leaf Spot", "ur": "چیری کے بھورے دھبے", "ps": "د چیری نسواري ټاپي"},
        "identity": "Cherry_Brown_Spot is caused by Blumeriella jaapii. Common names: Cherry Leaf Spot (en), چیری کے بھورے دھبے (ur), د چیری نسواري ټاپي (ps).",
        "symptoms": "Small purple circular spots on upper leaf surface turning brown. Centers may drop out. Leaves turn yellow and drop prematurely in summer.",
        "epidemiology": "Favored by warm wet spring weather. Overwinters on fallen leaves.",
        "cultural_control": "Rake and destroy fallen leaves in autumn. Prune trees for sunlight penetration.",
        "biological_control": "Bacillus subtilis foliar application.",
        "chemical_control": "Captan or Tebuconazole spray starting at petal fall.",
        "prevention": "Autumn leaf cleanup, post-bloom protective sprays.",
        "safety": "Standard pesticide safety.",
        "pakistan": "Common in cherry orchards of Swat, Gilgit, and Quetta.",
        "sources": "APS / UC IPM.",
    },
    "Cherry_Purple_Spot": {
        "crop": "Cherry",
        "scientific_name": "Cercospora circumscissa",
        "pathogen_type": "Fungal",
        "common_name": {"en": "Cherry Purple Spot", "ur": "چیری کے جامنی دھبے", "ps": "د چیری ارغواني ټاپي"},
        "identity": "Cherry_Purple_Spot is caused by Cercospora circumscissa. Common names: Purple Leaf Spot (en), چیری کے جامنی دھبے (ur), د چیری ارغواني ټاپي (ps).",
        "symptoms": "Distinct purple to reddish circular spots on leaves. Shot-hole effect when center tissue falls out.",
        "epidemiology": "Favored by high humidity and rain splash.",
        "cultural_control": "Prune tree canopy for ventilation, remove leaf litter.",
        "biological_control": "Neem oil bio-spray.",
        "chemical_control": "Copper Oxychloride or Mancozeb spray.",
        "prevention": "Pruning, orchard sanitation.",
        "safety": "Standard protective gear.",
        "pakistan": "Prevalent in upland cherry growing areas.",
        "sources": "APS / CABI.",
    },
    "Cherry_Scorch": {
        "crop": "Cherry",
        "scientific_name": "Gnomonia erythrostoma / Xylella fastidiosa",
        "pathogen_type": "Fungal / Bacterial",
        "common_name": {"en": "Cherry Leaf Scorch", "ur": "چیری کے پتے کا جلساؤ", "ps": "د چیری د پاڼو سوځیدنه"},
        "identity": "Cherry_Scorch is caused by Gnomonia erythrostoma or bacterial leaf scorch. Common names: Cherry Leaf Scorch (en), چیری کے پتے کا جلساؤ (ur), د چیری د پاڼو سوځیدنه (ps).",
        "symptoms": "Marginal yellowing and browning of leaves giving a scorched appearance. Leaves remain attached to shoots through winter.",
        "epidemiology": "Fungal spores discharge from overwintered leaves attached to trees.",
        "cultural_control": "Prune out dead shoots holding dried leaves during winter.",
        "biological_control": "Trichoderma bio-spray.",
        "chemical_control": "Dormant Copper Hydroxide spray followed by spring protectant.",
        "prevention": "Winter pruning of dead shoots.",
        "safety": "Standard spray safety.",
        "pakistan": "Occurs in northern high-altitude cherry orchards.",
        "sources": "APS / UC IPM.",
    },
    "Cherry_Shot_Hole": {
        "crop": "Cherry",
        "scientific_name": "Wilsonomyces carpophilus",
        "pathogen_type": "Fungal",
        "common_name": {"en": "Cherry Shot Hole", "ur": "چیری کے پتے کے سوراخ", "ps": "د چیری شاټ هول"},
        "identity": "Cherry_Shot_Hole is caused by Wilsonomyces carpophilus. Common names: Shot Hole (en), چیری کے پتے کے سوراخ (ur), د چیری شاټ هول (ps).",
        "symptoms": "Small reddish spots on leaves that drop out leaving clean holes ('shot hole'). Rough sunken spots on fruits.",
        "epidemiology": "Favored by cool wet spring weather.",
        "cultural_control": "Prune out infected twigs during winter. Avoid sprinkler irrigation.",
        "biological_control": "Bio-copper spray.",
        "chemical_control": "Apply Fixed Copper at autumn leaf fall and early spring delayed-dormant stage.",
        "prevention": "Dormant copper sprays, winter pruning.",
        "safety": "Wear eye protection.",
        "pakistan": "Widespread across stone fruit orchards in KP and Balochistan.",
        "sources": "UC IPM / APS.",
    },
    "Apricot_Blight": {
        "crop": "Apricot",
        "scientific_name": "Monilinia laxa",
        "pathogen_type": "Fungal",
        "common_name": {"en": "Apricot Brown Rot / Blight", "ur": "خوبانی کا جلاؤ", "ps": "د زردالو وسوځیدنه"},
        "identity": "Apricot_Blight is caused by Monilinia laxa. Common names: Brown Rot / Blossom Blight (en), خوبانی کا جلاؤ (ur), د زردالو وسوځیدنه (ps). Major blossom and fruit rot disease.",
        "symptoms": "Blossoms collapse, turn brown, and cling to twigs. Gummy cankers on twigs. Soft brown rot on ripening fruit covered with gray spore tufts.",
        "epidemiology": "Favored by rain during bloom. Overwinters in mummified fruits on trees or orchard floor.",
        "cultural_control": "Remove and destroy mummified fruits in winter. Prune out cankered twigs.",
        "biological_control": "Bacillus subtilis foliar spray at bloom.",
        "chemical_control": "Apply Tebuconazole or Captan at pink bud and full bloom stages.",
        "prevention": "Mummy removal, bloom protective sprays.",
        "safety": "Standard protective wear.",
        "pakistan": "Severe issue in apricot growing regions of Gilgit-Baltistan, KP (Swat), and Balochistan.",
        "sources": "UC IPM / CABI Plantwise.",
    },
    "Apricot_Shot_Hole": {
        "crop": "Apricot",
        "scientific_name": "Wilsonomyces carpophilus",
        "pathogen_type": "Fungal",
        "common_name": {"en": "Apricot Shot Hole", "ur": "خوبانی کے پتے کے سوراخ", "ps": "د زردالو شاټ هول"},
        "identity": "Apricot_Shot_Hole is caused by Wilsonomyces carpophilus. Common names: Shot Hole Disease (en), خوبانی کے پتے کے سوراخ (ur), د زردالو شاټ هول (ps).",
        "symptoms": "Purplish spots on young leaves that fall out leaving perforated holes. Raised corky spots on developing fruit surface.",
        "epidemiology": "Favored by spring rain and overhead moisture.",
        "cultural_control": "Prune out diseased wood. Avoid overhead irrigation.",
        "biological_control": "Bio-copper or sulfur spray.",
        "chemical_control": "Dormant Fixed Copper spray in autumn/early spring.",
        "prevention": "Autumn copper application, canopy pruning.",
        "safety": "Standard protective gear.",
        "pakistan": "Widespread across KP, GB, and Balochistan apricot orchards.",
        "sources": "UC IPM / APS.",
    },

    # -------------------------------------------------------------------------
    # CORN (3 Diseases)
    # -------------------------------------------------------------------------
    "Corn_Fungal": {
        "crop": "Corn",
        "scientific_name": "Bipolaris / Exserohilum spp.",
        "pathogen_type": "Fungal",
        "common_name": {"en": "Corn Fungal Leaf Blight", "ur": "مکئی کا فنگس عارضہ", "ps": "د جوارو فنګسي ناروغي"},
        "identity": "Corn_Fungal is caused by Bipolaris or Exserohilum species (Northern/Southern Corn Leaf Blight). Common names: Corn Leaf Blight (en), مکئی کا فنگس عارضہ (ur), د جوارو فنګسي ناروغي (ps).",
        "symptoms": "Long, elliptical grayish-green to tan lesions on leaves. Lesions coalesce causing broad leaf desiccation.",
        "epidemiology": "Favored by moderate temperatures (18-27°C) and high humidity or dew.",
        "cultural_control": "Rotate with non-cereal crops. Tillage to incorporate crop stubble. Plant resistant corn hybrids.",
        "biological_control": "Trichoderma seed treatment.",
        "chemical_control": "Azoxystrobin or Propiconazole foliar spray if threshold reached prior to tasseling.",
        "prevention": "Hybrid selection, crop rotation, stubble management.",
        "safety": "Standard protective equipment.",
        "pakistan": "Prevalent in spring and autumn maize crops in Punjab and KP.",
        "sources": "CIMMYT / Purdue Extension.",
    },
    "Corn_Gray_Spot": {
        "crop": "Corn",
        "scientific_name": "Cercospora zeae-maydis",
        "pathogen_type": "Fungal",
        "common_name": {"en": "Gray Leaf Spot of Corn", "ur": "مکئی کے خاکستری دھبے", "ps": "د جوارو خړ ټاپي"},
        "identity": "Corn_Gray_Spot is caused by Cercospora zeae-maydis. Common names: Gray Leaf Spot (en), مکئی کے خاکستری دھبے (ur), د جوارو خړ ټاپي (ps).",
        "symptoms": "Rectangular, tan to gray lesions strictly restricted by parallel leaf veins. Severe leaf blighting from lower leaves upward.",
        "epidemiology": "Favored by continuous warm humid weather (25-30°C) and conservation tillage.",
        "cultural_control": "2-year crop rotation, deep tillage of maize residue, planting resistant hybrids.",
        "biological_control": "Bacillus subtilis foliar treatment.",
        "chemical_control": "Pyraclostrobin or Tebuconazole spray at tasseling (VT) stage.",
        "prevention": "Resistant hybrids, crop rotation.",
        "safety": "Standard protective gear.",
        "pakistan": "Common in irrigated corn growing belts of Punjab (Sahiwal, Okara, Pakpattan) and KP (Mardan, Swabi).",
        "sources": "CIMMYT / Purdue Extension.",
    },
    "Corn_Holcus_Spot": {
        "crop": "Corn",
        "scientific_name": "Pseudomonas syringae pv. lapsa",
        "pathogen_type": "Bacterial",
        "common_name": {"en": "Holcus Bacterial Spot", "ur": "ہولکس بیکٹیریل دھبے", "ps": "د جوارو ہولکس باکتریایي ټاپي"},
        "identity": "Corn_Holcus_Spot is caused by Pseudomonas syringae pv. lapsa. Common names: Holcus Spot (en), ہولکس بیکٹیریل دھبے (ur), د جوارو ہولکس باکتریایي ټاپي (ps). Bacterial spot disease.",
        "symptoms": "Round to elliptical pale cream to tan spots (2-10mm) surrounded by a reddish-purple margin or yellow halo. Lower leaves affected first after early season storms.",
        "epidemiology": "Favored by early season rainstorms with strong winds that cause sandblasting or leaf micro-wounds.",
        "cultural_control": "Usually minor disease; crops outgrow early infection. Maintain balanced NPK fertility.",
        "biological_control": "Bio-copper drench.",
        "chemical_control": "Chemical sprays rarely economically justified.",
        "prevention": "Avoid early cultivation damage during wet weather.",
        "safety": "General safety hygiene.",
        "pakistan": "Occurs early in spring corn crops after thunderstorms in Punjab.",
        "sources": "Purdue Extension / CABI.",
    },

    # -------------------------------------------------------------------------
    # BEAN (3 Diseases)
    # -------------------------------------------------------------------------
    "Bean_Fungal": {
        "crop": "Bean",
        "scientific_name": "Colletotrichum lindemuthianum / Cercospora spp.",
        "pathogen_type": "Fungal",
        "common_name": {"en": "Bean Fungal Leaf Spot / Anthracnose", "ur": "پھلی کا فنگس عارضہ", "ps": "د لوبیا فنګسي ناروغي"},
        "identity": "Bean_Fungal is caused by Colletotrichum lindemuthianum or Cercospora species. Common names: Bean Anthracnose / Leaf Spot (en), پھلی کا فنگس عارضہ (ur), د لوبیا فنګسي ناروغي (ps).",
        "symptoms": "Dark reddish-brown to black sunken lesions along leaf veins on underside. Circular dark lesions on pods carrying pinkish spore masses.",
        "epidemiology": "Favored by cool to moderate temperatures (17-24°C) and high rainfall. Seed-borne pathogen.",
        "cultural_control": "Use certified disease-free seed. Practice 2-3 year crop rotation. Avoid working in fields when foliage is wet.",
        "biological_control": "Trichoderma seed treatment.",
        "chemical_control": "Mancozeb or Copper Oxychloride foliar spray.",
        "prevention": "Clean seed, seed dressing, crop rotation.",
        "safety": "Standard protective clothing.",
        "pakistan": "Prevalent in pulse/bean growing areas of KP (Swat, Hazara) and Punjab.",
        "sources": "CABI Plantwise / CIAT.",
    },
    "Bean_Rust": {
        "crop": "Bean",
        "scientific_name": "Uromyces appendiculatus",
        "pathogen_type": "Fungal",
        "common_name": {"en": "Bean Rust", "ur": "پھلی کا زنگ", "ps": "د لوبیا زنګ"},
        "identity": "Bean_Rust is caused by Uromyces appendiculatus. Common names: Bean Rust (en), پھلی کا زنگ (ur), د لوبیا زنګ (ps). Foliar fungal rust.",
        "symptoms": "Small reddish-brown powdery pustules surrounded by yellow chlorotic halos on leaves and pods. Severe defoliation.",
        "epidemiology": "Favored by cool night temperatures with high humidity and warm daytime conditions.",
        "cultural_control": "Plant resistant cultivars, destroy bean straw post-harvest, practice crop rotation.",
        "biological_control": "Bacillus subtilis foliar application.",
        "chemical_control": "Propiconazole or Sulfur spray at early rust detection.",
        "prevention": "Resistant varieties, crop residue destruction.",
        "safety": "Standard protective gear.",
        "pakistan": "Common in autumn bean crops in northern Punjab and KP.",
        "sources": "CABI Plantwise / CIAT.",
    },
    "Bean_Shot_Hole": {
        "crop": "Bean",
        "scientific_name": "Pseudocercospora / Stigmina spp.",
        "pathogen_type": "Fungal",
        "common_name": {"en": "Bean Shot Hole", "ur": "پھلی کے پتے کے سوراخ", "ps": "د لوبیا سوري لرونکي ټاپي"},
        "identity": "Bean_Shot_Hole is caused by Pseudocercospora or Stigmina species. Common names: Bean Shot Hole (en), پھلی کے پتے کے سوراخ (ur), د لوبیا سوري لرونکي ټاپي (ps).",
        "symptoms": "Small brown spots that drop out leaving clean holes in leaf blades.",
        "epidemiology": "Favored by warm humid rain splash conditions.",
        "cultural_control": "Remove crop debris, rotate crops.",
        "biological_control": "Neem oil spray.",
        "chemical_control": "Copper Hydroxide spray.",
        "prevention": "Clean seed, field sanitation.",
        "safety": "Standard safety equipment.",
        "pakistan": "Occurs in rainfed bean crops in northern hilly regions.",
        "sources": "CABI Plantwise / CIAT.",
    },

    # -------------------------------------------------------------------------
    # FIG (3 Diseases)
    # -------------------------------------------------------------------------
    "Fig_Blight": {
        "crop": "Fig",
        "scientific_name": "Pellicularia koleroga",
        "pathogen_type": "Fungal",
        "common_name": {"en": "Fig Thread Blight", "ur": "انجیر کا بلاسٹ یا جلاؤ", "ps": "د انځر سوځیدنه"},
        "identity": "Fig_Blight is caused by Pellicularia koleroga. Common names: Fig Blight (en), انجیر کا جلاؤ (ur), د انځر سوځیدنه (ps).",
        "symptoms": "Large water-soaked leaf spots that turn silvery white to tan. Mycelial threads bind dead leaves to branches.",
        "epidemiology": "Favored by high humidity and dense unpruned tree canopy.",
        "cultural_control": "Prune fig tree canopy for ventilation. Remove dead leaves clinging to branches.",
        "biological_control": "Trichoderma spray.",
        "chemical_control": "Copper Oxychloride spray.",
        "prevention": "Canopy pruning, sanitation.",
        "safety": "Standard protective gear.",
        "pakistan": "Common in fig orchards of KP and Balochistan.",
        "sources": "CABI Plantwise.",
    },
    "Fig_Brown_Spot": {
        "crop": "Fig",
        "scientific_name": "Cercospora fici",
        "pathogen_type": "Fungal",
        "common_name": {"en": "Fig Leaf Spot / Brown Spot", "ur": "انجیر کے بھورے دھبے", "ps": "د انځر نسواري ټاپي"},
        "identity": "Fig_Brown_Spot is caused by Cercospora fici. Common names: Fig Leaf Spot (en), انجیر کے بھورے دھبے (ur), د انځر نسواري ټاپي (ps).",
        "symptoms": "Reddish-brown circular spots on leaf surface. Severe infection causes early leaf defoliation.",
        "epidemiology": "Favored by warm humid rain splash.",
        "cultural_control": "Prune lower branches, collect fallen leaves.",
        "biological_control": "Neem oil bio-spray.",
        "chemical_control": "Mancozeb spray.",
        "prevention": "Leaf sanitation.",
        "safety": "Standard protective clothing.",
        "pakistan": "Prevalent in fig growing areas of KP and Sindh.",
        "sources": "CABI Plantwise.",
    },
    "Fig_Rust": {
        "crop": "Fig",
        "scientific_name": "Cerotelium fici",
        "pathogen_type": "Fungal",
        "common_name": {"en": "Fig Rust", "ur": "انجیر کا زنگ", "ps": "د انځر زنگ"},
        "identity": "Fig_Rust is caused by Cerotelium fici. Common names: Fig Rust (en), انجیر کا زنگ (ur), د انځر زنگ (ps).",
        "symptoms": "Tiny yellowish-orange pustules on leaf underside. Leaves turn brown and drop prematurely.",
        "epidemiology": "Favored by warm humid weather late in season.",
        "cultural_control": "Rake and burn fallen leaves.",
        "biological_control": "Sulfur dusting.",
        "chemical_control": "Wettable Sulfur spray.",
        "prevention": "Clean orchard floor.",
        "safety": "Eye protection for sulfur.",
        "pakistan": "Common in late summer across fig orchards.",
        "sources": "CABI Plantwise.",
    },

    # -------------------------------------------------------------------------
    # WALNUT (4 Diseases/Pests)
    # -------------------------------------------------------------------------
    "Walnut_Anthracnose": {
        "crop": "Walnut",
        "scientific_name": "Ophiognomonia leptostyla",
        "pathogen_type": "Fungal",
        "common_name": {"en": "Walnut Anthracnose / Leaf Blotch", "ur": "اخروٹ کا اینتھراکنوز", "ps": "د اخروټ انترکنوز"},
        "identity": "Walnut_Anthracnose is caused by Ophiognomonia leptostyla. Common names: Walnut Anthracnose (en), اخروٹ کا اینتھراکنوز (ur), د اخروټ انترکنوز (ps).",
        "symptoms": "Circular brown lesions on leaves and green husk of nuts. Early defoliation and poor nut filling.",
        "epidemiology": "Favored by rainy spring weather.",
        "cultural_control": "Rake fallen leaves in winter. Prune lower limbs.",
        "biological_control": "Bio-copper spray.",
        "chemical_control": "Copper Hydroxide spray at leaf expansion.",
        "prevention": "Autumn leaf sanitation.",
        "safety": "Standard safety equipment.",
        "pakistan": "Major issue in walnut valleys of KP (Swat, Dir) and GB.",
        "sources": "UC IPM / CABI.",
    },
    "Walnut_Blotch": {
        "crop": "Walnut",
        "scientific_name": "Xanthomonas arboricola pv. juglandis",
        "pathogen_type": "Bacterial",
        "common_name": {"en": "Walnut Bacterial Blight", "ur": "اخروٹ کا بیکٹیریل بلائٹ", "ps": "د اخروټ باکتریایي ټاپي"},
        "identity": "Walnut_Blotch is caused by Xanthomonas arboricola pv. juglandis. Common names: Walnut Blight (en), اخروٹ کا بیکٹیریل بلائٹ (ur), د اخروټ باکتریایي ټاپي (ps). Bacterial disease.",
        "symptoms": "Black water-soaked spots on leaves, catkins, and young nuts. Nuts drop prematurely.",
        "epidemiology": "Favored by spring rain during catkin emergence and bloom.",
        "cultural_control": "Prune out dead shoots. Avoid sprinkler irrigation.",
        "biological_control": "Bacillus subtilis spray.",
        "chemical_control": "Copper Hydroxide + Mancozeb spray during bloom.",
        "prevention": "Bloom bactericide sprays.",
        "safety": "Standard protective gear.",
        "pakistan": "Widespread across walnut growing districts of KP and GB.",
        "sources": "UC IPM / CABI.",
    },
    "Walnut_Gall_Mite": {
        "crop": "Walnut",
        "scientific_name": "Aceria erinea",
        "pathogen_type": "Pest",
        "common_name": {"en": "Walnut Blister Gall Mite", "ur": "اخروٹ کی گال مائٹ", "ps": "د اخروټ چنجی يا مایټ"},
        "identity": "Walnut_Gall_Mite is caused by Aceria erinea. Common names: Walnut Gall Mite (en), اخروٹ کی گال مائٹ (ur), د اخروټ مایټ (ps). PEST: Do NOT use fungicides.",
        "symptoms": "Large yellow-green blister-like galls on upper leaf surface. Dense felt-like yellow hair mats on underside.",
        "epidemiology": "Mites overwinter in bud scales.",
        "cultural_control": "Usually non-fatal; prune out heavily galled leaves.",
        "biological_control": "Predatory mites.",
        "chemical_control": "Wettable Sulfur spray at spring bud burst.",
        "prevention": "Dormant sulfur application.",
        "safety": "Eye protection.",
        "pakistan": "Common in northern hilly walnut groves.",
        "sources": "UC IPM.",
    },
    "Walnut_Shot_Hole": {
        "crop": "Walnut",
        "scientific_name": "Wilsonomyces / Gnomonia spp.",
        "pathogen_type": "Fungal",
        "common_name": {"en": "Walnut Shot Hole", "ur": "اخروٹ کے پتے کے سوراخ", "ps": "د اخروټ شاټ هول"},
        "identity": "Walnut_Shot_Hole is caused by Wilsonomyces or Gnomonia species. Common names: Walnut Shot Hole (en), اخروٹ کے پتے کے سوراخ (ur), د اخروټ شاټ هول (ps).",
        "symptoms": "Small brown leaf spots that drop out leaving clean holes.",
        "epidemiology": "Favored by rain splash.",
        "cultural_control": "Winter leaf sanitation, pruning.",
        "biological_control": "Bio-copper spray.",
        "chemical_control": "Fixed Copper spray.",
        "prevention": "Sanitation.",
        "safety": "Standard protective gear.",
        "pakistan": "Occurs in northern high-altitude orchards.",
        "sources": "UC IPM / CABI.",
    },

    # -------------------------------------------------------------------------
    # PERSIMMON & LOKAT (3 Diseases/Healthy)
    # -------------------------------------------------------------------------
    "Persimmons_Brown_Spot": {
        "crop": "Persimmon",
        "scientific_name": "Cercospora kakivora / Mycosphaerella nawae",
        "pathogen_type": "Fungal",
        "common_name": {"en": "Persimmon Angular Leaf Spot", "ur": "املوک کے بھورے دھبے", "ps": "د املوک نسواري ټاپي"},
        "identity": "Persimmons_Brown_Spot is caused by Cercospora kakivora / Mycosphaerella nawae. Common names: Persimmon Angular Leaf Spot (en), املوک کے بھورے دھبے (ur), د املوک نسواري ټاپي (ps).",
        "symptoms": "Angular dark brown spots on leaves bounded by leaf veinlets. Early leaf defoliation causing fruit drop.",
        "epidemiology": "Favored by summer rainfall and high relative humidity.",
        "cultural_control": "Collect and destroy fallen leaf litter in autumn. Prune tree canopy.",
        "biological_control": "Trichoderma bio-fungicide.",
        "chemical_control": "Mancozeb or Difenoconazole spray during summer.",
        "prevention": "Autumn sanitation, summer foliage protection.",
        "safety": "Standard protective gear.",
        "pakistan": "Common in persimmon orchards of KP (Peshawar, Charsadda, Swat).",
        "sources": "CABI Plantwise.",
    },
    "Lokat_Leaf_Spot": {
        "crop": "Lokat",
        "scientific_name": "Entomosporium mespili",
        "pathogen_type": "Fungal",
        "common_name": {"en": "Loquat Entomosporium Leaf Spot", "ur": "لوکاٹ کے پتے کے دھبے", "ps": "د لوکاټ د پاڼو ټاپي"},
        "identity": "Lokat_Leaf_Spot is caused by Entomosporium mespili. Common names: Loquat Leaf Spot (en), لوکاٹ کے پتے کے دھبے (ur), د لوکاټ د پاڼو ټاپي (ps).",
        "symptoms": "Small reddish-brown circular spots with dark halos on loquat leaves. Lesions coalesce causing leaf desiccation.",
        "epidemiology": "Favored by cool wet spring conditions.",
        "cultural_control": "Prune inner branches for ventilation. Destroy fallen leaves.",
        "biological_control": "Bacillus subtilis bio-fungicide.",
        "chemical_control": "Copper Hydroxide or Mancozeb spray.",
        "prevention": "Pruning and fall leaf sanitation.",
        "safety": "Standard safety procedures.",
        "pakistan": "Prevalent in loquat orchards of KP (Mardan, Haripur) and Punjab (Kallar Kahar).",
        "sources": "CABI Plantwise.",
    },
    "Lokat_Healthy": {
        "crop": "Lokat",
        "scientific_name": "Eriobotrya japonica",
        "pathogen_type": "Healthy",
        "common_name": {"en": "Healthy Loquat Leaf", "ur": "لوکاٹ کا صحت مند پتا", "ps": "د لوکاټ روغه پاڼه"},
        "identity": "Lokat_Healthy represents a healthy, disease-free Eriobotrya japonica loquat tree.",
        "symptoms": "Leaves are dark glossy green, leathery, and free of fungal leaf spots or insect damage.",
        "epidemiology": "Optimal growth under good orchard management.",
        "cultural_control": "Maintain good orchard management: balanced NPK fertilization, timely irrigation, annual pruning.",
        "biological_control": "Promote natural beneficial insects.",
        "chemical_control": "No chemical control required for healthy crops. Maintain regular field scouting.",
        "prevention": "Routine orchard hygiene.",
        "safety": "Maintain clean equipment.",
        "pakistan": "Standard healthy loquat crop target in Pakistan.",
        "sources": "CABI / FAO GAP Guidelines.",
    },
}


def build_unknown_chunk(cname: str) -> dict:
    """Build single mandatory chunk for Unknown classes."""
    crop = cname.split("_")[0]
    cid = cname.upper()

    return {
        "chunk_id": f"{cid}_UNCERTAIN",
        "disease_id": cid,
        "disease_class": cname,
        "crop": crop,
        "country": "Pakistan",
        "province": "All",
        "section": "uncertain_classification",
        "evidence_level": "A2",
        "verified": True,
        "parent_id": cid,
        "text": f"This classification ({cname}) is uncertain. Request clearer image.",
        "source_organization": "ZARI.ai Classification Verification System",
        "url": "https://zari.ai/unknown-policy",
    }


def main() -> None:
    print("=" * 75)
    print("  ZARI.ai — REMAINING CROPS EVIDENCE RESEARCH & CHUNKING ENGINE")
    print("=" * 75)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    chunks_list: list[dict] = []
    class_chunk_counts: dict[str, int] = {}

    sections = [
        "identity", "symptoms", "epidemiology", "cultural_control",
        "biological_control", "chemical_control", "prevention", "safety",
        "pakistan", "sources"
    ]

    # 1. Process Unknown Classes (9 classes)
    for cname in sorted(UNKNOWN_CLASSES):
        unknown_chunk = build_unknown_chunk(cname)
        chunks_list.append(unknown_chunk)
        class_chunk_counts[cname] = 1

    # 2. Process Disease & Healthy Classes (32 classes)
    for cname, data in REMAINING_EVIDENCE_DATA.items():
        cid = cname.upper()
        count_for_class = 0

        if data.get("pathogen_type") == "Healthy":
            # Healthy maintenance chunks
            chunk_entry = {
                "chunk_id": f"{cid}_HEALTHY_MAINTENANCE",
                "disease_id": cid,
                "disease_class": cname,
                "crop": data["crop"],
                "country": "Pakistan",
                "province": "All",
                "section": "healthy_maintenance",
                "evidence_level": "A2",
                "verified": True,
                "parent_id": cid,
                "text": f"This crop ({cname}) appears healthy. Maintain with good agricultural practices, balanced fertilization, proper irrigation, and routine IPM field scouting.",
                "source_organization": "CABI / FAO GAP Guidelines",
                "url": "https://www.cabi.org/plantwiseplus",
            }
            chunks_list.append(chunk_entry)
            count_for_class = 1
        else:
            # Full structured chunks
            for sec in sections:
                if sec not in data:
                    continue

                content_text = data[sec]
                if not content_text:
                    continue

                e_level = "A1" if sec == "sources" else ("B1" if sec == "biological_control" else "A2")

                chunk_entry = {
                    "chunk_id": f"{cid}_{sec.upper()}",
                    "disease_id": cid,
                    "disease_class": cname,
                    "crop": data["crop"],
                    "country": "Pakistan",
                    "province": "All",
                    "section": sec,
                    "evidence_level": e_level,
                    "verified": True,
                    "parent_id": cid,
                    "text": content_text,
                    "source_organization": data.get("sources", "CABI / APS / UC IPM").split("/")[0].strip(),
                    "url": "https://www.cabi.org/plantwiseplus",
                }
                chunks_list.append(chunk_entry)
                count_for_class += 1

        class_chunk_counts[cname] = count_for_class

    # Save to JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(chunks_list, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Created total {len(chunks_list)} structured RAG chunks across {len(class_chunk_counts)} remaining classes.")
    print(f"✓ Saved remaining chunks JSON to: {OUTPUT_JSON}\n")

    # Print Summary Table
    print("=" * 75)
    print(f"{'Class Name':<28} | {'Type':<12} | {'Chunks Created':<15} | {'Section/Policy':<15}")
    print("-" * 75)

    for cname in sorted(class_chunk_counts.keys()):
        count = class_chunk_counts[cname]
        if cname in UNKNOWN_CLASSES:
            ptype = "Unknown"
            sec = "Uncertain policy"
        elif REMAINING_EVIDENCE_DATA.get(cname, {}).get("pathogen_type") == "Healthy":
            ptype = "Healthy"
            sec = "Maintenance"
        else:
            ptype = REMAINING_EVIDENCE_DATA[cname]["pathogen_type"]
            sec = "Full IPM (10)"

        print(f"{cname:<28} | {ptype:<12} | {count:<15} | {sec:<15}")

    print("-" * 75)
    print("✅ REMAINING CLASSES RAG CHUNKING COMPLETE!")


if __name__ == "__main__":
    main()
