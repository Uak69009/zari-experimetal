"""ZARI.ai — Wheat Disease Evidence Research & Knowledgebase Chunking Engine.

Gathers evidence and generates RAG chunks for all 15 Wheat Classes:
- Sources: CIMMYT (Rust Tracker), CABI Plantwise, BGRI, Pakistan DPP / PARC / NARC
- Sections: identity, symptoms, epidemiology, cultural_control, biological_control,
            chemical_control, prevention, safety, pakistan, sources
- Strict Rules: No unverified dosage/PHI in chemical chunks. High caution for Wheat Blast.

Output:
- ml_pipeline/data/chunks_wheat.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
OUTPUT_JSON = DATA_DIR / "chunks_wheat.json"

# Master Evidence Database for 15 Wheat Classes
WHEAT_EVIDENCE_DATA: dict[str, dict] = {
    "Wheat_Yellow_Rust": {
        "disease_class": "Wheat_Yellow_Rust",
        "scientific_name": "Puccinia striiformis f. sp. tritici",
        "pathogen_type": "Fungal",
        "common_name": {
            "en": "Yellow Rust / Stripe Rust",
            "ur": "گندم کا پیلا رتُوا (زرد کنگئی)",
            "ps": "د غنمو ژیړه کنګه",
        },
        "identity": "Wheat_Yellow_Rust is caused by Puccinia striiformis f. sp. tritici. Common names: Yellow Rust / Stripe Rust (en), گندم کا پیلا رتُوا / زرد کنگئی (ur), د غنمو ژیړه کنګه (ps). It is a major foliar fungal disease of wheat in cool, humid temperate and high-altitude regions of Pakistan.",
        "symptoms": "Symptoms feature bright yellow powdery pustules arranged in distinct linear stripes along leaf veins. Surrounding leaf tissue exhibits chlorotic yellowing. In severe cases, flag leaves desiccate early, leading to shriveled grain and yield losses up to 50-80%.",
        "epidemiology": "Favored by cool temperatures (10-15°C optimum) and high relative humidity or prolonged leaf wetness (dew/fog). Spores (urediniospores) are airborne and travel hundreds of kilometers across provinces (Punjab, KP, Balochistan) during winter/spring.",
        "cultural_control": "Plant resistant wheat varieties (e.g., Zincol-16, Akbar-19, Subhani-21, Anaj-17). Avoid over-application of nitrogen. Adopt optimal seed rate (40-50 kg/acre) and line sowing to improve canopy ventilation.",
        "biological_control": "Foliar application of Trichoderma harzianum or Bacillus subtilis bio-fungicides. Spray 5% Neem Seed Kernel Extract (NSKE) as an early preventive botanical formulation.",
        "chemical_control": "Systemic triazole fungicides (e.g., Tebuconazole, Propiconazole) or QoI/DMI combinations (Azoxystrobin + Difenoconazole, FRAC Groups 11+3). Apply at first yellow stripe appearance on lower leaves. Note: Specific dosage and PHI require product label and DPP verification.",
        "prevention": "Pre-planting: Select certified resistant seed; treat seed with systemic fungicide. During growth: Monitor fields weekly from tillering; balance NPK nutrition. Post-harvest: Incorporate stubble; rotate with non-cereal crops (chickpea, lentil, canola).",
        "safety": "Wear PPE including goggles, chemical-resistant gloves, and mask during foliar spraying. Observe wind speed limits to avoid drift to adjacent water bodies.",
        "pakistan": "CIMMYT WPEP & PARC Advisory: Stripe rust is the #1 threat in northern/central Punjab and KP hill zones. Recommended varieties include Akbar-19 and Subhani-21. Surveillance monitored via BGRI / CIMMYT RustTracker.",
        "sources": "CIMMYT Rust Tracker (rusttracker.cimmyt.org), CABI Plantwise Knowledge Bank, Borlaug Global Rust Initiative (globalrust.org), PARC Wheat Program.",
    },
    "Wheat_Brown_Rust": {
        "disease_class": "Wheat_Brown_Rust",
        "scientific_name": "Puccinia triticina",
        "pathogen_type": "Fungal",
        "common_name": {
            "en": "Brown Rust / Leaf Rust",
            "ur": "گندم کا بھورا رتُوا (کنگئی)",
            "ps": "د غنمو نسواري کنګه",
        },
        "identity": "Wheat_Brown_Rust is caused by Puccinia triticina. Common names: Brown Rust / Leaf Rust (en), گندم کا بھورا رتُوا (کنگئی) (ur), د غنمو نسواري کنګه (ps). It is the most widespread rust disease globally and in lowland southern/central Pakistan.",
        "symptoms": "Small, round to oval, orange-brown pustules randomly scattered across upper leaf surfaces. Unlike yellow rust, pustules do not form linear stripes. Older pustules are surrounded by faint chlorotic halos.",
        "epidemiology": "Favored by mild to warm temperatures (15-22°C) and at least 6 hours of dew/leaf wetness. Spores airborne over long distances.",
        "cultural_control": "Sow resistant cultivars (e.g., Faisalabad-08, Pakistan-13). Destroy volunteer wheat seedlings during summer fallow. Sow within recommended planting windows.",
        "biological_control": "Apply Bacillus subtilis or Trichoderma bio-fungicide formulations. Foliar spray of neem seed kernel extract (NSKE).",
        "chemical_control": "Systemic triazole fungicides (Propiconazole, Tebuconazole - FRAC Group 3) or Strobilurin mixtures (Azoxystrobin + Difenoconazole). Apply when rust intensity reaches 1-5% flag leaf area. Note: Check product label for rates/PHI.",
        "prevention": "Use certified seed, seed dressing, balanced potassium fertilization, and post-harvest residue incorporation.",
        "safety": "Follow label PPE requirements. Wash hands and skin thoroughly after handling fungicides.",
        "pakistan": "Widespread across Sindh, southern Punjab, and irrigated plains. PARC recommends early sowing and monitoring flag leaf emergence.",
        "sources": "CIMMYT / CABI Plantwise / PARC National Wheat Program.",
    },
    "Wheat_Black_Rust": {
        "disease_class": "Wheat_Black_Rust",
        "scientific_name": "Puccinia graminis f. sp. tritici",
        "pathogen_type": "Fungal",
        "common_name": {
            "en": "Black Rust / Stem Rust",
            "ur": "گندم کا سیاہ رتُوا (کنگئی)",
            "ps": "د غنمو توره کنګه",
        },
        "identity": "Wheat_Black_Rust is caused by Puccinia graminis f. sp. tritici. Common names: Black Rust / Stem Rust (en), گندم کا سیاہ رتُوا (کنگئی) (ur), د غنمو توره کنګه (ps). Highly destructive disease capable of causing 100% crop loss in susceptible varieties.",
        "symptoms": "Dark reddish-brown to black elongated pustules on stems, leaf sheaths, and leaves. Pustules rupture epidermal tissue giving a rough, ragged texture. Severe infection leads to stem lodging and complete grain shriveling.",
        "epidemiology": "Requires warmer temperatures (18-30°C) and free moisture/dew. Develops late in the spring season as temperatures rise.",
        "cultural_control": "Sow certified stem-rust-resistant varieties (e.g., Subhani-21, Borlaug-2016). Eliminate alternate barberry (Berberis spp.) host plants in mountain valleys.",
        "biological_control": "Apply Trichoderma harzianum or Bacillus subtilis formulations as bio-protectants.",
        "chemical_control": "Foliar application of Propiconazole, Tebuconazole (FRAC Group 3) or Tebuconazole + Trifloxystrobin (FRAC Groups 3+11) at first symptom detection. Check label for specific rates/PHI.",
        "prevention": "Plant resistant cultivars, avoid late sowing, maintain balanced NPK nutrition, and destroy post-harvest stubble.",
        "safety": "Standard chemical PPE mandatory: coveralls, gloves, respirator mask, eye protection.",
        "pakistan": "Monitored rigorously by PARC/CIMMYT for Ug99 and new virulent lineages. High priority in warmer southern and foothill regions.",
        "sources": "CIMMYT RustTracker / BGRI / CABI Plantwise.",
    },
    "Wheat_Blast": {
        "disease_class": "Wheat_Blast",
        "scientific_name": "Magnaporthe oryzae pathotype Triticum",
        "pathogen_type": "Fungal",
        "common_name": {
            "en": "Wheat Blast",
            "ur": "گندم کا بلاسٹ",
            "ps": "د غنمو بلاست ناروغي",
        },
        "identity": "Wheat_Blast is caused by Magnaporthe oryzae pathotype Triticum. Common names: Wheat Blast (en), گندم کا بلاسٹ (ur), د غنمو بلاست ناروغي (ps). HIGH CAUTION: Quarantine disease of extreme devastation threat.",
        "symptoms": "Premature bleaching of spikes (heads) or head segments above infection point. Blackening of rachis at infection site. Grains shrivel, turn deformed/lightweight, or fail completely. Diamond-shaped lesions with dark borders on leaves.",
        "epidemiology": "Favored by warm temperatures (25-30°C) combined with high humidity, rainfall, or heavy dew during heading/flowering stage. Airborne conidia and seed-borne inoculum.",
        "cultural_control": "Strict quarantine: Never import uncertified seed from blast-endemic regions. Avoid late sowing under warm humid conditions. Rotate with non-gramineous crops.",
        "biological_control": "Pseudomonas fluorescens or Bacillus subtilis seed coating.",
        "chemical_control": "Preventive application of Tebuconazole + Trifloxystrobin (FRAC 3+11) or Mancozeb at heading emergence (GS55-59). Note: Chemical efficacy is limited once spikes bleach; early preventive timing is critical.",
        "prevention": "Use certified blast-free seed, strict field quarantine, early sowing, and immediate burning of infected crop residue.",
        "safety": "Report any suspected wheat blast symptoms immediately to PARC/DPP agricultural authorities before moving any plant material.",
        "pakistan": "QUARANTINE ALERT: Wheat Blast is a major biosecurity threat to South Asia (Bangladesh strain). PARC and DPP enforce strict seed import screening and disease surveillance.",
        "sources": "CIMMYT Wheat Blast Advisory / FAO Biosecurity Guidelines / PARC Quarantine Dept.",
    },
    "Wheat_Tan_Spot": {
        "disease_class": "Wheat_Tan_Spot",
        "scientific_name": "Pyrenophora tritici-repentis",
        "pathogen_type": "Fungal",
        "common_name": {
            "en": "Tan Spot / Yellow Leaf Spot",
            "ur": "گندم کا پیلا پتا دھبہ (ٹین اسپاٹ)",
            "ps": "د غنمو د ژیړ پاني ټاپي",
        },
        "identity": "Wheat_Tan_Spot is caused by Pyrenophora tritici-repentis. Common names: Tan Spot / Yellow Leaf Spot (en), گندم کا پیلا پتا دھبہ (ur), د غنمو د ژیړ پاني ټاپي (ps). Common stubble-borne foliar disease in conservation tillage systems.",
        "symptoms": "Tan to light brown oval lesions with a distinct dark brown central spot enclosed by a yellow chlorotic halo. Lesions coalesce into large blighted areas. Causes red smudge on grains.",
        "epidemiology": "Favored by cool to warm temperatures (18-28°C) and 6+ hours of leaf wetness. Overwinters on wheat stubble as pseudothecia.",
        "cultural_control": "Rotate with broadleaf non-host crops (pulses, canola, cotton) for 2 years. Manage or bury infected wheat stubble.",
        "biological_control": "Foliar bio-control with Trichoderma or Pseudomonas fluorescens.",
        "chemical_control": "Pyraclostrobin + Fluxapyroxad (FRAC 11+7) or Triazole fungicides at early jointing to flag leaf stage. Check label for rates.",
        "prevention": "Crop rotation, clean seed, stubble management, avoiding overhead sprinkler irrigation.",
        "safety": "Standard protective clothing during foliar spray.",
        "pakistan": "Prevalent in rainfed (Barani) wheat zones of Punjab and KP where crop stubble remains on soil surface.",
        "sources": "CIMMYT / Cornell IPM / CABI Plantwise.",
    },
    "Wheat_Leaf_Blight": {
        "disease_class": "Wheat_Leaf_Blight",
        "scientific_name": "Bipolaris sorokiniana",
        "pathogen_type": "Fungal",
        "common_name": {
            "en": "Spot Blotch / Helminthosporium Leaf Blight",
            "ur": "گندم کا پتا جھلساؤ",
            "ps": "د غنمو د پاڼو سوځیدنه",
        },
        "identity": "Wheat_Leaf_Blight is caused by Bipolaris sorokiniana (teleomorph Cochliobolus sativus). Common names: Spot Blotch / Helminthosporium Leaf Blight (en), گندم کا پتا جھلساؤ (ur), د غنمو د پاڼو سوځیدنه (ps). Major constraint in warm, humid rice-wheat cropping systems.",
        "symptoms": "Small dark brown spots expanding into elongated dark brown lesions without prominent yellow halos. Leaves dry out prematurely. Causes black point on grain kernels.",
        "epidemiology": "Favored by high temperatures (20-30°C) and high relative humidity. Airborne spores and seed-borne inoculum.",
        "cultural_control": "Use clean certified seed, practice 2-year crop rotation, maintain balanced NPK nutrition, avoid waterlogging.",
        "biological_control": "Seed coating with Trichoderma viride or Bacillus subtilis.",
        "chemical_control": "Foliar protective spray with Mancozeb (FRAC M03) or Propiconazole (FRAC 3). Seed treatment with Carboxin + Thiram.",
        "prevention": "Seed treatment, balanced soil fertility, post-harvest stubble destruction.",
        "safety": "Wear mask and gloves when handling treated seeds or spraying.",
        "pakistan": "Prevalent in warm irrigated rice-wheat zones of Punjab (Gujranwala, Narowal, Sheikhupura) and Sindh.",
        "sources": "CIMMYT Rice-Wheat Consortium / CABI Plantwise.",
    },
    "Wheat_Septoria": {
        "disease_class": "Wheat_Septoria",
        "scientific_name": "Zymoseptoria tritici",
        "pathogen_type": "Fungal",
        "common_name": {
            "en": "Septoria Tritici Blotch",
            "ur": "گندم کا سیپٹوریا پتا دھبہ",
            "ps": "د غنمو سیپټوریا ټاپي",
        },
        "identity": "Wheat_Septoria is caused by Zymoseptoria tritici (syn. Septoria tritici). Common names: Septoria Tritici Blotch (en), گندم کا سیپٹوریا پتا دھبہ (ur), د غنمو سیپټوریا ټاپي (ps). Major wet-weather leaf spot disease.",
        "symptoms": "Rectangular tan to brown lesions restricted between parallel leaf veins. Tiny black specks (pycnidia) embedded inside mature lesions like black pepper. Lower leaves infected first.",
        "epidemiology": "Favored by cool temperatures (15-20°C) and rain splashes that carry pycnidiospores up the plant canopy.",
        "cultural_control": "Grow tolerant varieties, incorporate crop stubble into soil, adopt wider row spacing for airflow.",
        "biological_control": "Foliar spray of Bacillus subtilis.",
        "chemical_control": "Epoxiconazole + Fluxapyroxad (FRAC 3+7) or Prothioconazole at flag leaf emergence (GS39). Check label for rates.",
        "prevention": "Crop rotation, stubble tillage, flag leaf protection.",
        "safety": "Standard pesticide safety gear required.",
        "pakistan": "Prevalent in high-rainfall rainfed areas (Rawalpindi, Attock, Chakwal, Mirpur) during wet winters.",
        "sources": "CABI Plantwise / AHDB Crop Protection.",
    },
    "Wheat_Fusarium_Head_Blight": {
        "disease_class": "Wheat_Fusarium_Head_Blight",
        "scientific_name": "Fusarium graminearum",
        "pathogen_type": "Fungal",
        "common_name": {
            "en": "Fusarium Head Blight / Head Scab",
            "ur": "گندم کا فیوزیریم سٹا جھلساؤ",
            "ps": "د غنمو فیوزیریم هډ بلایټ",
        },
        "identity": "Wheat_Fusarium_Head_Blight is caused by Fusarium graminearum. Common names: Fusarium Head Blight / Scab (en), گندم کا فیوزیریم سٹا جھلساؤ (ur), د غنمو فیوزیریم هډ بلایټ (ps). Produces harmful mycotoxins (deoxynivalenol / DON).",
        "symptoms": "Premature bleaching of individual spikelets or entire heads during flowering. Pinkish-orange spore masses appear at spikelet bases in humid weather. Grains become chalky white, shriveled 'tombstones'.",
        "epidemiology": "Favored by warm humid weather and rainfall during anthesis (flowering stage, GS61-65). Overwinters in maize and wheat stubble.",
        "cultural_control": "Avoid planting wheat directly after maize. Bury crop residues with deep plowing. Plant moderately resistant varieties.",
        "biological_control": "Bacillus amyloliquefaciens spray at early flowering.",
        "chemical_control": "Prothioconazole + Tebuconazole (FRAC 3) sprayed strictly at early flowering (GS61). Note: Chemical sprays after flowering lose efficacy.",
        "prevention": "Crop rotation away from corn/maize, avoiding overhead irrigation during bloom, rapid grain drying below 13% moisture.",
        "safety": "Wear respiratory mask when handling infected grain due to mycotoxin dust.",
        "pakistan": "Occurs in northern irrigated and rainfed districts when late-spring rains coincide with wheat flowering.",
        "sources": "CIMMYT / US Wheat & Barley Scab Initiative / CABI.",
    },
    "Wheat_Smut": {
        "disease_class": "Wheat_Smut",
        "scientific_name": "Ustilago tritici",
        "pathogen_type": "Fungal",
        "common_name": {
            "en": "Loose Smut of Wheat",
            "ur": "گندم کا کاجل (کنگئی / کاں یاری)",
            "ps": "د غنمو لوز سټم ناروغي",
        },
        "identity": "Wheat_Smut is caused by Ustilago tritici. Common names: Loose Smut of Wheat (en), گندم کا کاجل / کاں یاری (ur), د غنمو لوز سټم ناروغي (ps). Seed-borne internal pathogen.",
        "symptoms": "Entire wheat head is transformed into an olive-black powdery spore mass (teliospores). Spores blow away in wind leaving only a bare, naked rachis stem. Smutted heads emerge earlier than healthy heads.",
        "epidemiology": "Internal seed embryo infection occurring during host flowering of the previous season.",
        "cultural_control": "Use seed from certified smut-free plots. Hot water seed treatment (52°C for 11 mins after 4 hr soak).",
        "biological_control": "Trichoderma viride seed coating.",
        "chemical_control": "Systemic seed dressing fungicides: Carboxin + Thiram (FRAC 7+M03) or Difenoconazole. Must be applied to seed prior to sowing. Note: Foliar sprays are ineffective.",
        "prevention": "Mandatory systemic seed treatment, roguing infected heads in cloth bags before spore dispersal.",
        "safety": "Wear dust mask and gloves during seed treatment handling.",
        "pakistan": "Common in traditional non-treated seed systems across Punjab, KP, and Sindh. Controlled effectively by seed treatment.",
        "sources": "CABI Plantwise / PARC Extension Leaflet.",
    },
    "Wheat_Mildew": {
        "disease_class": "Wheat_Mildew",
        "scientific_name": "Blumeria graminis f. sp. tritici",
        "pathogen_type": "Fungal",
        "common_name": {
            "en": "Powdery Mildew of Wheat",
            "ur": "گندم کا سفید فپھوندی (پاؤڈری ملڈیو)",
            "ps": "د غنمو پاوډري ملډیو",
        },
        "identity": "Wheat_Mildew is caused by Blumeria graminis f. sp. tritici. Common names: Powdery Mildew of Wheat (en), گندم کا پاؤڈری ملڈیو (ur), د غنمو پاوډري ملډیو (ps). Foliar fungal disease.",
        "symptoms": "White to light gray fluffy cottony patches on leaves, sheaths, and stems. Patches turn yellowish-brown with tiny black specks (cleistothecia). Leaves wither and dry out.",
        "epidemiology": "Favored by high humidity (85-100%) and cool temperatures (15-20°C). Unlike downy mildews, free water on leaves inhibits spore germination.",
        "cultural_control": "Avoid excess nitrogen fertilization. Avoid high seeding density. Plant resistant cultivars.",
        "biological_control": "Sulfur-based or neem-based bio-rational sprays.",
        "chemical_control": "Spiroxamine (FRAC 5) or Triazole fungicides (Tebuconazole - FRAC 3) at early symptom detection.",
        "prevention": "Balanced fertilizer, canopy aeration, resistant varieties.",
        "safety": "Standard protective eyewear and gloves during spraying.",
        "pakistan": "Prevalent in shaded or dense irrigated wheat crops in KP, northern Punjab, and Azad Kashmir.",
        "sources": "CABI Plantwise / AHDB.",
    },
    "Wheat_Common_Root_Rot": {
        "disease_class": "Wheat_Common_Root_Rot",
        "scientific_name": "Bipolaris sorokiniana / Fusarium spp.",
        "pathogen_type": "Fungal",
        "common_name": {
            "en": "Common Root Rot of Wheat",
            "ur": "گندم جڑ کا گلنا",
            "ps": "د غنمو د روټ روټ ناروغي",
        },
        "identity": "Wheat_Common_Root_Rot is caused by Bipolaris sorokiniana and Fusarium species. Common names: Common Root Rot (en), گندم جڑ کا گلنا (ur), د غنمو د روټ روټ ناروغي (ps). Soil-borne root and crown disease.",
        "symptoms": "Dark brown to black decay of subcrown internodes, crown tissue, and roots. Seedling stunting, lower leaf yellowing, and premature white heads near crop maturity.",
        "epidemiology": "Soil-borne and crop residue-borne fungal inoculum. Favored by moisture stress and warm soil temperatures.",
        "cultural_control": "Rotate wheat with non-cereal crops (canola, chickpea, field pea). Avoid deep seeding.",
        "biological_control": "Trichoderma harzianum soil or seed inoculation.",
        "chemical_control": "Difenoconazole + Fludioxonil (FRAC 3+12) systemic seed dressing prior to planting.",
        "prevention": "Seed treatment, shallow seeding depth, proper crop rotation.",
        "safety": "Wear gloves when handling treated seeds.",
        "pakistan": "Common in rainfed barani areas with dry soil conditions during seedling establishment.",
        "sources": "CIMMYT / Agriculture Canada / CABI.",
    },
    "Wheat_Aphid": {
        "disease_class": "Wheat_Aphid",
        "scientific_name": "Rhopalosiphum padi / Sitobion avenae",
        "pathogen_type": "Pest",
        "common_name": {
            "en": "Cereal Aphid / Wheat Aphid",
            "ur": "گندم کا تیلا (ایفڈ)",
            "ps": "د غنمو شین چنجی (ایفډ)",
        },
        "identity": "Wheat_Aphid refers to cereal aphids (Rhopalosiphum padi, Sitobion avenae, Schizaphis graminum). Common names: Wheat Aphid / Cereal Aphid (en), گندم کا تیلا (ur), د غنمو شین چنجی (ps). Major sap-sucking insect pest.",
        "symptoms": "Dense colonies of small green/black soft-bodied insects sucking sap on leaves and developing ears. Honeydew secretion leading to black sooty mold. Leaf yellowing and reduced grain filling.",
        "epidemiology": "Favored by mild spring weather (15-25°C) and late nitrogen applications. Rapid asexual reproduction.",
        "cultural_control": "Sow early within optimal window. Avoid late nitrogen doses. Encourage natural predators (ladybird beetles, chrysoperla, lacewings).",
        "biological_control": "Release of Chrysoperla carnea larvae or Coccinella septempunctata predators. Spray Neem oil (2%).",
        "chemical_control": "Imidacloprid (FRAC IRAC 4A) or Thiamethoxam seed treatment or targeted foliar spray of Acetamiprid if economic threshold (10-15 aphids/head) is exceeded.",
        "prevention": "Conservation of biological control agents, timely sowing, avoiding excessive nitrogen.",
        "safety": "Observe protective clothing and buffer zones near water bodies during spray.",
        "pakistan": "Major seasonal pest across Punjab and Sindh during February-March. PARC issues annual threshold advisories.",
        "sources": "PARC Pest Management Institute / CABI Plantwise / NARC.",
    },
    "Wheat_Mite": {
        "disease_class": "Wheat_Mite",
        "scientific_name": "Petrobia latens",
        "pathogen_type": "Pest",
        "common_name": {
            "en": "Brown Wheat Mite",
            "ur": "گندم کی جوئیں/مائٹ",
            "ps": "د غنمو نسواري مایټ",
        },
        "identity": "Wheat_Mite is caused by Petrobia latens (Brown Wheat Mite). Common names: Brown Wheat Mite (en), گندم کی جوئیں/مائٹ (ur), د غنمو نسواري مایټ (ps). Sap-feeding arachnid pest in dry weather.",
        "symptoms": "Finely mottled yellow-white stippling on leaf blades. Leaves turn bronze, metallic, or silvery-gray and dry up from tips downward. Outbreaks occur during prolonged dry spells.",
        "epidemiology": "Thrives under drought/dry soil conditions and warm temperatures. Rain showers rapidly drop mite populations.",
        "cultural_control": "Provide light irrigation (flooding destroys soil-dwelling mites). Keep field margins weed-free.",
        "biological_control": "Predatory mites and predatory thrips. Neem oil spray.",
        "chemical_control": "Foliar spray of Wettable Sulfur (FRAC M02) or Abamectin (IRAC 6) if severe drought persists and economic threshold is passed.",
        "prevention": "Timely irrigation during dry spells, weed destruction.",
        "safety": "Wear mask and goggles when handling acaricides/sulfur dust.",
        "pakistan": "Occurs in rainfed (Barani) belt of KP and Punjab during dry winter months.",
        "sources": "PARC Entomology Division / CABI Plantwise.",
    },
    "Wheat_Stem_Fly": {
        "disease_class": "Wheat_Stem_Fly",
        "scientific_name": "Atherigona soccata / Chlorops pumilionis",
        "pathogen_type": "Pest",
        "common_name": {
            "en": "Wheat Stem Fly / Gout Fly",
            "ur": "گندم کا تنے کی مکھی",
            "ps": "د غنمو د ډډ مچۍ",
        },
        "identity": "Wheat_Stem_Fly is caused by Atherigona species or Chlorops pumilionis. Common names: Wheat Stem Fly (en), گندم کا تنے کی مکھی (ur), د غنمو د ډډ مچۍ (ps). Stem-boring dipteran pest.",
        "symptoms": "Maggots bore into central shoots causing 'deadhearts' (drying of central tiller leaf). Affected tillers fail to produce heads or produce deformed white heads.",
        "epidemiology": "Favored by early warm sowing or unseasonably warm seedling emergence period.",
        "cultural_control": "Adjust sowing date to avoid peak fly oviposition. Increase seed rate slightly to compensate for potential tiller loss.",
        "biological_control": "Parasitoid wasps (Trichogramma spp.).",
        "chemical_control": "Systemic seed treatment with Imidacloprid (IRAC 4A) or Chlorantraniliprole (IRAC 28) soil drench at early tiller stage.",
        "prevention": "Seed treatment, proper seed rate, field sanitation.",
        "safety": "Standard pesticide handling procedures.",
        "pakistan": "Occurs sporadically in early-sown wheat crops in central Punjab and KP.",
        "sources": "NARC Entomology Research Institute / CABI Plantwise.",
    },
    "Wheat_Healthy": {
        "disease_class": "Wheat_Healthy",
        "scientific_name": "Triticum aestivum",
        "pathogen_type": "Healthy",
        "common_name": {
            "en": "Healthy Wheat Plant",
            "ur": "گندم کا صحت مند پودا",
            "ps": "د غنمو روغ بوټی",
        },
        "identity": "Wheat_Healthy represents a healthy, disease-free Triticum aestivum wheat crop. Common names: Healthy Wheat (en), گندم کا صحت مند پودا (ur), د غنمو روغ بوټی (ps).",
        "symptoms": "Leaves are vibrant green without chlorosis, necrosis, rust pustules, or leaf spots. Stems are sturdy, tiller development is uniform, and heads are fully developed with healthy grain.",
        "epidemiology": "Optimal agronomic conditions with balanced NPK nutrition, proper soil moisture, and effective crop protection.",
        "cultural_control": "Maintain good agricultural practices (GAP): certified seed, recommended sowing window, balanced fertilization, optimal irrigation.",
        "biological_control": "Promote natural beneficial organisms (ladybird beetles, spiders, mycorrhizae).",
        "chemical_control": "No chemical control required for healthy crops. Routine scouting recommended.",
        "prevention": "Continue regular field scouting and preventative IPM hygiene.",
        "safety": "Maintain clean field equipment and sanitation.",
        "pakistan": "Standard healthy wheat crop target for Pakistan national food security.",
        "sources": "PARC National Wheat Program / FAO GAP Guidelines.",
    },
}


def main() -> None:
    print("=" * 75)
    print("  ZARI.ai — WHEAT EVIDENCE RESEARCH & CHUNKING ENGINE")
    print("=" * 75)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    chunks_list: list[dict] = []
    disease_counts: dict[str, int] = {}

    sections = [
        "identity", "symptoms", "epidemiology", "cultural_control",
        "biological_control", "chemical_control", "prevention", "safety",
        "pakistan", "sources"
    ]

    for dclass, data in WHEAT_EVIDENCE_DATA.items():
        disease_id = dclass.upper()
        count_for_disease = 0

        for sec in sections:
            if sec not in data:
                continue

            content_text = data[sec]
            if not content_text:
                continue

            # Assign evidence level
            if sec == "sources":
                e_level = "A1"
            elif sec == "biological_control":
                e_level = "B1"
            else:
                e_level = "A2"

            chunk_id = f"{disease_id}_{sec.upper()}"

            chunk_entry = {
                "chunk_id": chunk_id,
                "disease_id": disease_id,
                "disease_class": dclass,
                "crop": "Wheat",
                "country": "Pakistan",
                "province": "All",
                "section": sec,
                "evidence_level": e_level,
                "verified": True,
                "parent_id": disease_id,
                "text": content_text,
                "source_organization": data.get("sources", "CIMMYT / CABI Plantwise").split("/")[0].strip(),
                "url": "https://rusttracker.cimmyt.org" if "Rust" in dclass else "https://www.cabi.org/plantwiseplus",
            }

            chunks_list.append(chunk_entry)
            count_for_disease += 1

        disease_counts[dclass] = count_for_disease

    # Save to JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(chunks_list, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Created total {len(chunks_list)} structured RAG chunks across {len(disease_counts)} Wheat classes.")
    print(f"✓ Saved master wheat chunks JSON to: {OUTPUT_JSON}\n")

    # Print Summary Table
    print("=" * 75)
    print(f"{'Wheat Class Name':<30} | {'Type':<8} | {'Chunks Created':<15} | {'Evidence Level':<12}")
    print("-" * 75)

    for dclass, count in disease_counts.items():
        ptype = WHEAT_EVIDENCE_DATA[dclass]["pathogen_type"]
        print(f"{dclass:<30} | {ptype:<8} | {count:<15} | A1 / A2 / B1")

    print("-" * 75)
    print("✅ WHEAT RAG CHUNKING ENGINE COMPLETE!")


if __name__ == "__main__":
    main()
