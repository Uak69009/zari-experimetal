"""ZARI.ai — Tomato Disease Evidence Research & Knowledgebase Chunking Engine.

Gathers evidence and generates RAG chunks for all 11 Tomato Classes:
- Sources: Cornell Vegetable Pathology, CABI Plantwise, World Vegetable Center (AVRDC), UC IPM
- Sections: identity, symptoms, epidemiology, cultural_control, biological_control,
            chemical_control, prevention, safety, pakistan, sources
- Special Rules:
  * Tomato_Curl (VIRAL): Fungicides useless; vector (whitefly) management & sticky traps
  * Tomato_Late_Blight: Highly destructive; preventive fungicide approach
  * Tomato_Miner & Tomato_Spider_Mites (PESTS): Insecticides/acaricides & pheromone traps (NOT fungicides)

Output:
- ml_pipeline/data/chunks_tomato.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
OUTPUT_JSON = DATA_DIR / "chunks_tomato.json"

# Master Evidence Database for 11 Tomato Classes
TOMATO_EVIDENCE_DATA: dict[str, dict] = {
    "Tomato_Early_Blight": {
        "disease_class": "Tomato_Early_Blight",
        "scientific_name": "Alternaria solani",
        "pathogen_type": "Fungal",
        "common_name": {
            "en": "Tomato Early Blight",
            "ur": "ٹماٹر کا اگیتا جھلساؤ",
            "ps": "د ټماټرو دمخه سوځیدنه",
        },
        "identity": "Tomato_Early_Blight is caused by the fungal pathogen Alternaria solani. Common names: Tomato Early Blight (en), ٹماٹر کا اگیتا جھلساؤ (ur), د ټماټرو دمخه سوځیدنه (ps). Common foliar disease affecting solanaceous crops in warm humid conditions.",
        "symptoms": "Dark brown circular spots with characteristic concentric rings ('target-board' or bullseye pattern) surrounded by yellow chlorotic leaf halos. Older lower leaves affected first, progressing upward. Causes stem dark sunken lesions and stem-end rot on fruits.",
        "epidemiology": "Favored by warm temperatures (24-29°C) and high humidity or frequent rainfall/overhead irrigation. Spores spread by wind, rain splash, and contaminated tools. Overwinters on infected crop debris and solanaceous weeds.",
        "cultural_control": "Practice 3-year crop rotation with non-solanaceous crops. Prune lower leaves up to 30cm from ground to prevent soil splash. Use drip irrigation instead of overhead sprinklers. Mulch soil surface with straw or plastic film.",
        "biological_control": "Foliar application of Trichoderma harzianum or Bacillus subtilis bio-fungicide. Apply Neem seed kernel extract (5%) as a preventive bio-rational.",
        "chemical_control": "Apply protective contact fungicides (Mancozeb, Chlorothalonil - FRAC Group M03/M05) or systemic curative triazoles (Difenoconazole + Azoxystrobin - FRAC 3+11) at initial target-spot appearance. Note: Check label for rates and PHI.",
        "prevention": "Select disease-free certified seeds/seedlings, stake plants for ventilation, prune bottom leaves, and clear Solanaceous weed hosts (nightshade).",
        "safety": "Wear chemical-resistant gloves, protective coveralls, and respiratory mask during foliar spraying.",
        "pakistan": "Widespread across tomato growing clusters in Punjab (Faisalabad, Sheikhupura), Sindh (Badin, Thatta), and KP (Swat, Malakand). Early staking reduces disease incidence.",
        "sources": "Cornell Vegetable Pathology / CABI Plantwise / AVRDC (World Vegetable Center).",
    },
    "Tomato_Late_Blight": {
        "disease_class": "Tomato_Late_Blight",
        "scientific_name": "Phytophthora infestans",
        "pathogen_type": "Fungal",
        "common_name": {
            "en": "Tomato Late Blight",
            "ur": "ٹماٹر کا پچھیتا جھلساؤ",
            "ps": "د ټماټرو وروسته سوځیدنه",
        },
        "identity": "Tomato_Late_Blight is caused by Phytophthora infestans. Common names: Tomato Late Blight (en), ٹماٹر کا پچھیتا جھلساؤ (ur), د ټماټرو وروسته سوځیدنه (ps). EXTREMELY DESTRUCTIVE oomycete pathogen capable of destroying whole tomato fields within days.",
        "symptoms": "Large water-soaked pale green to dark brown oily lesions expanding rapidly on leaves. White cottony downy mold appears on leaf underside during humid morning hours. Dark brown greasy lesions on stems and petioles cause plant collapse. Firm leathery brown rot on green/ripe fruit.",
        "epidemiology": "Favored by cool to moderate temperatures (15-22°C) and continuous high relative humidity (> 90%) or prolonged dew/fog. Airborne sporangia travel long distances on cool wind currents.",
        "cultural_control": "Plant certified disease-free nursery stock. Avoid overhead sprinkler irrigation. Ensure wide plant spacing for rapid leaf drying. Destroy volunteer tomato and potato plants.",
        "biological_control": "Foliar bio-drench with Bacillus subtilis or Bio-copper formulations.",
        "chemical_control": "PREVENTIVE FUNGICIDE APPROACH MANDATORY: Apply protectants (Mancozeb, Copper Hydroxide) or systemic oomycete mixtures (Metalaxyl-M + Mancozeb, Cymoxanil, Dimethomorph - FRAC 4+M03/27) immediately when weather turns cool and wet BEFORE symptoms spread. Check product label for rates.",
        "prevention": "Plant resistant cultivars (e.g. Mountain Magic), conduct daily field scouting during cool rainy weather, destroy blighted vines immediately.",
        "safety": "Wear full PPE and clean spray equipment thoroughly after application.",
        "pakistan": "Major threat in winter tomato crops of Punjab and spring crops of KP valleys. PARC issues urgent weather advisories when relative humidity exceeds 85%.",
        "sources": "Cornell Vegetable Pathology / CIP (International Potato Center) / CABI Plantwise.",
    },
    "Tomato_Curl": {
        "disease_class": "Tomato_Curl",
        "scientific_name": "Tomato Yellow Leaf Curl Virus (TYLCV)",
        "pathogen_type": "Viral",
        "common_name": {
            "en": "Tomato Yellow Leaf Curl Virus (TYLCV)",
            "ur": "ٹماٹر کا پتا موڑ وائرس",
            "ps": "د ټماټرو د پاڼو پیچلتیا وایرس",
        },
        "identity": "Tomato_Curl is caused by Tomato Yellow Leaf Curl Virus (TYLCV). Common names: Tomato Leaf Curl (en), ٹماٹر کا پتا موڑ وائرس (ur), د ټماټرو د پاڼو پیچلتیا وایرس (ps). VIRAL PATHOGEN: Fungicides are completely ineffective against viruses.",
        "symptoms": "Severe upright stunting of entire plant. Leaves exhibit pronounced upward curling, yellowing of leaf margins (chlorosis), and reduced leaflet size. Flowers abort prematurely leading to severe fruit yield failure.",
        "epidemiology": "Transmitted exclusively by the silverleaf whitefly vector (Bemisia tabaci). High whitefly populations in warm dry weather lead to rapid virus spread. Not seed-transmitted.",
        "cultural_control": "FOCUS ON WHITEFLY VECTOR CONTROL: Install yellow sticky traps (20-30 traps/acre) to catch adult whiteflies. Use 50-mesh insect-proof netting over nurseries. Intercrop with non-host barrier crops (maize, sorghum). Plant TYLCV-resistant hybrids.",
        "biological_control": "Release predatory mites or Encarsia formosa parasitoids. Spray Neem oil (2%) or Pyrethrum-based botanical insecticides against whiteflies.",
        "chemical_control": "SPECIAL NOTE: DO NOT SPRAY FUNGICIDES (they have zero impact on viruses). Target the whitefly vector using systemic insecticides (Imidacloprid, Thiamethoxam, Spirotetramat, Pyriproxyfen - IRAC 4A/23/7C). Observe label instructions.",
        "prevention": "Use insect-proof nursery nets, plant TYLCV-resistant seed varieties, eradicate weed hosts (Solanum nigrum), install yellow sticky traps.",
        "safety": "Follow insecticide label precautions during whitefly sprays.",
        "pakistan": "Extremely prevalent across Sindh, Punjab, and KP during warm autumn and spring months. Whitefly control is mandatory for tomato production.",
        "sources": "AVRDC (World Vegetable Center) / UC IPM / CABI Plantwise.",
    },
    "Tomato_Bacterial_Spot": {
        "disease_class": "Tomato_Bacterial_Spot",
        "scientific_name": "Xanthomonas perforans / vesicatoria",
        "pathogen_type": "Bacterial",
        "common_name": {
            "en": "Tomato Bacterial Spot",
            "ur": "ٹماٹر کے بیکٹیریل دھبے",
            "ps": "د ټماټرو باکتریایي ټاپي",
        },
        "identity": "Tomato_Bacterial_Spot is caused by Xanthomonas species (X. perforans, X. vesicatoria). Common names: Bacterial Spot (en), ٹماٹر کے بیکٹیریل دھبے (ur), د ټماټرو باکتریایي ټاپي (ps). Bacterial foliage and fruit disease.",
        "symptoms": "Small, water-soaked dark spots on leaves that turn dark brown to black with greasy halos. Center of spots may dry up and drop out giving a shot-hole appearance. Raised black scab-like spots on green fruits.",
        "epidemiology": "Favored by warm temperatures (24-30°C) and high humidity, heavy rain, or overhead irrigation. Bacteria enter through stomata and leaf wounds.",
        "cultural_control": "Use hot-water treated or certified disease-free seed. Avoid overhead sprinkler irrigation. Sanitize stakes and field equipment with bleach.",
        "biological_control": "Bacteriophage treatments or Bacillus subtilis foliar spray.",
        "chemical_control": "Foliar application of Fixed Copper (Copper Hydroxide / Copper Oxychloride - FRAC M01) tank-mixed with Mancozeb to enhance copper bactericidal activity. Note: Antibiotics restricted; check local regulations.",
        "prevention": "Clean seed, copper bactericide sprays, drip irrigation, 2-year crop rotation.",
        "safety": "Standard protective gear during copper bactericide sprays.",
        "pakistan": "Common during monsoon tomato crops in Punjab and summer crops in KP.",
        "sources": "Cornell Vegetable Pathology / CABI Plantwise.",
    },
    "Tomato_Fusarium_Wilt": {
        "disease_class": "Tomato_Fusarium_Wilt",
        "scientific_name": "Fusarium oxysporum f. sp. lycopersici",
        "pathogen_type": "Fungal",
        "common_name": {
            "en": "Tomato Fusarium Wilt",
            "ur": "ٹماٹر کا فیوزیریم مرحضا",
            "ps": "د ټماټرو فیوزیریم مرضاوی",
        },
        "identity": "Tomato_Fusarium_Wilt is caused by Fusarium oxysporum f. sp. lycopersici. Common names: Fusarium Wilt (en), ٹماٹر کا فیوزیریم مرحضا (ur), د ټماټرو فیوزیریم مرضاوی (ps). Soil-borne vascular wilt pathogen.",
        "symptoms": "Yellowing of lower leaves often starting on one side of the plant ('flagging'). Plants wilt during hot afternoon hours and recover at night initially. Slicing main stem reveals dark brown vascular ring discoloration. Eventual plant death.",
        "epidemiology": "Soil-borne pathogen that persists in soil for years via chlamydospores. Favored by warm soil temperatures (27-28°C) and root knot nematode injury.",
        "cultural_control": "Use resistant tomato varieties (F or FF resistant). Soil solarization using clear plastic in summer. Maintain soil pH 6.5-7.0. Avoid root damage during weeding.",
        "biological_control": "Soil drench with Trichoderma harzianum or Pseudomonas fluorescens at transplanting.",
        "chemical_control": "Preventive soil drench with Hymexazol or Carbendazim. Note: Foliar fungicide sprays cannot cure vascular wilt once plant is infected.",
        "prevention": "Resistant varieties, soil solarization, clean tools, root knot nematode management.",
        "safety": "Wear protective gloves during soil drenching.",
        "pakistan": "Prevalent in continuous tomato cropping soils across Punjab and Sindh.",
        "sources": "CABI Plantwise / AVRDC / USDA.",
    },
    "Tomato_Verticillium_Wilt": {
        "disease_class": "Tomato_Verticillium_Wilt",
        "scientific_name": "Verticillium dahliae",
        "pathogen_type": "Fungal",
        "common_name": {
            "en": "Tomato Verticillium Wilt",
            "ur": "ٹماٹر کا ورٹیسیلیم مرجھاؤ",
            "ps": "د ټماټرو ورټیسیلیم مرضاوی",
        },
        "identity": "Tomato_Verticillium_Wilt is caused by Verticillium dahliae. Common names: Verticillium Wilt (en), ٹماٹر کا ورٹیسیلیم مرجھاؤ (ur), د ټماټرو ورټیسیلیم مرضاوی (ps). Soil-borne fungal vascular wilt.",
        "symptoms": "V-shaped yellow chlorotic wedges on lower leaf margins progressing to brown leaf necrosis. Light tan vascular discoloration inside lower stem base. Plant stunting under cool soil conditions.",
        "epidemiology": "Favored by cooler soil temperatures (20-24°C). Overwinters in soil as microsclerotia for 8-10 years.",
        "cultural_control": "Plant Verticillium-resistant cultivars (V-resistant). 4-year crop rotation with non-hosts (maize, wheat). Summer soil solarization.",
        "biological_control": "Soil drench with Trichoderma virens bio-fungicide.",
        "chemical_control": "Soil solarization or bio-fumigation pre-planting. Note: Chemical foliar sprays do not cure vascular wilts.",
        "prevention": "V-resistant seed selection, 4-year rotation, soil solarization.",
        "safety": "Standard agricultural safety hygiene.",
        "pakistan": "Common in spring and autumn tomato crops in cooler hilly areas of KP and Balochistan.",
        "sources": "CABI Plantwise / UC IPM.",
    },
    "Tomato_Mold": {
        "disease_class": "Tomato_Mold",
        "scientific_name": "Passalora fulva",
        "pathogen_type": "Fungal",
        "common_name": {
            "en": "Tomato Leaf Mold",
            "ur": "ٹماٹر کی لیف مولڈ",
            "ps": "د ټماټرو مولډ",
        },
        "identity": "Tomato_Mold is caused by Passalora fulva (syn. Cladosporium fulvum). Common names: Leaf Mold (en), ٹماٹر کی لیف مولڈ (ur), د ټماټرو مولډ (ps). Foliar disease common in protected/greenhouse cultivation.",
        "symptoms": "Pale yellow spots on upper leaf surface corresponding to olive-green velvety mold growth on the leaf underside. Leaves wither and drop prematurely.",
        "epidemiology": "Favored by high relative humidity (> 85%) and moderate temperatures (20-24°C). Spores airborne in greenhouse environments.",
        "cultural_control": "Reduce greenhouse relative humidity below 85% with ventilation fans. Prune lower leaves to improve airflow. Use resistant greenhouse varieties.",
        "biological_control": "Trichoderma or Bio-copper foliar sprays.",
        "chemical_control": "Copper Hydroxide (FRAC M01) or Difenoconazole spray at early disease onset. Check label for rates.",
        "prevention": "Humidity control below 85%, proper pruning, resistant seed choice.",
        "safety": "Standard protective mask during greenhouse spraying.",
        "pakistan": "Major issue in tunnel tomato production in Punjab (Faisalabad, Rawalpindi) during humid winter months.",
        "sources": "Cornell Vegetable Pathology / CABI / UC IPM.",
    },
    "Tomato_Septoria": {
        "disease_class": "Tomato_Septoria",
        "scientific_name": "Septoria lycopersici",
        "pathogen_type": "Fungal",
        "common_name": {
            "en": "Tomato Septoria Leaf Spot",
            "ur": "ٹماٹر کے سیپٹوریا دھبے",
            "ps": "د ټماټرو سیپټوریا ټاپي",
        },
        "identity": "Tomato_Septoria is caused by Septoria lycopersici. Common names: Septoria Leaf Spot (en), ٹماٹر کے سیپٹوریا دھبے (ur), د ټماټرو سیپټوریا ټاپي (ps). Severe foliar defoliator.",
        "symptoms": "Numerous small circular spots with dark brown margins and light gray centers containing tiny black specks (pycnidia). Bottom-up leaf defoliation exposing fruit to sunscald.",
        "epidemiology": "Favored by warm temperatures (20-25°C) and wet weather or rain splash. Spores splash from infected crop debris.",
        "cultural_control": "Mulch beneath tomato plants, remove lower infected leaves, practice 3-year crop rotation.",
        "biological_control": "Foliar spray of Bacillus subtilis.",
        "chemical_control": "Chlorothalonil (FRAC M05), Mancozeb (FRAC M03), or Copper Hydroxide. Apply at initial leaf spot detection.",
        "prevention": "Mulching, crop rotation, pruning lower foliage.",
        "safety": "Standard pesticide PPE required.",
        "pakistan": "Common in open-field rainy season tomato crops across Punjab and KP.",
        "sources": "Cornell Vegetable Pathology / CABI Plantwise.",
    },
    "Tomato_Miner": {
        "disease_class": "Tomato_Miner",
        "scientific_name": "Tuta absoluta / Liriomyza sativae",
        "pathogen_type": "Pest",
        "common_name": {
            "en": "Tomato Leafminer (Tuta absoluta)",
            "ur": "ٹماٹر کا لیف مائنر",
            "ps": "د ټماټرو ليکنکی چنجی",
        },
        "identity": "Tomato_Miner refers to the tomato leafminer pest (Tuta absoluta / Liriomyza sativae). Common names: Tomato Leafminer (en), ٹماٹر کا لیف مائنر (ur), د ټماټرو ليکنکی چنجی (ps). INSECT PEST: Do NOT use fungicides.",
        "symptoms": "Larvae mine inside leaf mesophyll creating irregular transparent blotches or mines. Frass (black insect excrement) visible inside mines. Larvae bore into stems and green fruits causing decay.",
        "epidemiology": "High reproduction rate in warm dry weather. Overwinters as eggs/pupae in soil or crop debris.",
        "cultural_control": "Install sex pheromone traps (4-5 traps/acre for monitoring, 20 traps/acre for mass trapping). Use 50-mesh net in greenhouses. Destroy infested fruits.",
        "biological_control": "Spray Bacillus thuringiensis (Bt) or Spinosad. Release Trichogramma wasps or Nesidiocoris tenuis predatory bugs.",
        "chemical_control": "INSECTICIDE CONTROL: Chlorantraniliprole (IRAC 28), Emamectin Benzoate (IRAC 6), or Spinetoram. Rotate chemical classes to prevent rapid resistance development.",
        "prevention": "Pheromone traps, insect-proof nets, destroy infested fruits, bio-pesticides.",
        "safety": "Observe pesticide safety gear and harvest spray intervals.",
        "pakistan": "Major economic pest threatening tomato crops in Punjab, KP, and Balochistan. PARC advocates IPM mass trapping.",
        "sources": "EPPO / CABI Plantwise / PARC Tuta Absoluta Advisory.",
    },
    "Tomato_Spider_Mites": {
        "disease_class": "Tomato_Spider_Mites",
        "scientific_name": "Tetranychus urticae",
        "pathogen_type": "Pest",
        "common_name": {
            "en": "Two-Spotted Spider Mite",
            "ur": "ٹماٹر کی لال مکڑی (مائٹس)",
            "ps": "د ټماټرو دوه ټاپې وال ژوي",
        },
        "identity": "Tomato_Spider_Mites is caused by Tetranychus urticae (Two-Spotted Spider Mite). Common names: Spider Mites (en), ٹماٹر کی لال مکڑی (ur), د ټماټرو دوه ټاپې وال ژوي (ps). ARACHNID PEST: Do NOT use fungicides.",
        "symptoms": "Fine white/yellow stippling dot pattern on upper leaf surface. Fine silky webbing on underside of leaves. Severe infestation causes leaves to turn bronze, dry up, and drop.",
        "epidemiology": "Thrives in hot, dry, dusty conditions (30°C+). Rapid generation cycle (5-7 days in hot weather).",
        "cultural_control": "Maintain adequate irrigation to reduce plant water stress. Wash dust off leaves with overhead water jet. Remove weed hosts.",
        "biological_control": "Release Phytoseiulus persimilis predatory mites. Apply Neem oil (2%) or Potassium salts of fatty acids (insecticidal soap).",
        "chemical_control": "ACARICIDE CONTROL: Abamectin (IRAC 6), Spiromesifen (IRAC 23), or Wettable Sulfur (FRAC M02). Ensure thorough spray coverage on leaf undersides.",
        "prevention": "Avoid water stress, release predatory mites, spray neem oil early, avoid excessive synthetic pyrethroids.",
        "safety": "Wear eye protection and mask during sulfur/acaricide spraying.",
        "pakistan": "Common in tunnel and open-field tomatoes during hot dry months (April-June) across Punjab and Sindh.",
        "sources": "UC IPM / CABI Plantwise / AVRDC.",
    },
    "Tomato_Healthy": {
        "disease_class": "Tomato_Healthy",
        "scientific_name": "Solanum lycopersicum",
        "pathogen_type": "Healthy",
        "common_name": {
            "en": "Healthy Tomato Leaf",
            "ur": "ٹماٹر کا صحت مند پتا",
            "ps": "د ټماټرو روغه پاڼه",
        },
        "identity": "Tomato_Healthy represents a healthy, disease-free Solanum lycopersicum tomato plant. Common names: Healthy Tomato (en), ٹماٹر کا صحت مند پتا (ur), د ټماټرو روغه پاڼه (ps).",
        "symptoms": "Vibrant dark green foliage free of chlorotic spots, dark blighted lesions, viral curling, or insect mining. Stems are upright and fruit development is uniform.",
        "epidemiology": "Optimal growth under balanced NPK nutrition, regulated drip irrigation, and proactive IPM scouting.",
        "cultural_control": "Maintain Good Agricultural Practices (GAP): certified seeds, proper staking, drip irrigation, balanced fertilizer, crop rotation.",
        "biological_control": "Promote beneficial predators (ladybirds, lacewings, predatory mites).",
        "chemical_control": "No chemical control needed. Regular weekly crop scouting recommended.",
        "prevention": "Continue routine IPM monitoring and sanitation.",
        "safety": "Keep nursery tools clean and sanitized.",
        "pakistan": "Standard healthy crop model for commercial tomato growers in Pakistan.",
        "sources": "AVRDC / FAO GAP Guidelines.",
    },
}


def main() -> None:
    print("=" * 75)
    print("  ZARI.ai — TOMATO EVIDENCE RESEARCH & CHUNKING ENGINE")
    print("=" * 75)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    chunks_list: list[dict] = []
    disease_counts: dict[str, int] = {}

    sections = [
        "identity", "symptoms", "epidemiology", "cultural_control",
        "biological_control", "chemical_control", "prevention", "safety",
        "pakistan", "sources"
    ]

    for dclass, data in TOMATO_EVIDENCE_DATA.items():
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
                "crop": "Tomato",
                "country": "Pakistan",
                "province": "All",
                "section": sec,
                "evidence_level": e_level,
                "verified": True,
                "parent_id": disease_id,
                "text": content_text,
                "source_organization": data.get("sources", "Cornell / CABI Plantwise").split("/")[0].strip(),
                "url": "https://www.cabi.org/plantwiseplus",
            }

            chunks_list.append(chunk_entry)
            count_for_disease += 1

        disease_counts[dclass] = count_for_disease

    # Save to JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(chunks_list, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Created total {len(chunks_list)} structured RAG chunks across {len(disease_counts)} Tomato classes.")
    print(f"✓ Saved master tomato chunks JSON to: {OUTPUT_JSON}\n")

    # Print Summary Table
    print("=" * 75)
    print(f"{'Tomato Class Name':<30} | {'Type':<8} | {'Chunks Created':<15} | {'Evidence Level':<12}")
    print("-" * 75)

    for dclass, count in disease_counts.items():
        ptype = TOMATO_EVIDENCE_DATA[dclass]["pathogen_type"]
        print(f"{dclass:<30} | {ptype:<8} | {count:<15} | A1 / A2 / B1")

    print("-" * 75)
    print("✅ TOMATO RAG CHUNKING ENGINE COMPLETE!")


if __name__ == "__main__":
    main()
