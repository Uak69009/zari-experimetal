"""ZARI.ai — Disease Identity Mapping & Scientific Nomenclature Registry.

This script maps all 67 Head Classes to:
1. Crop Name
2. Scientific Taxonomy / Pathogen Name (Latin)
3. Pathogen Type (Fungal, Bacterial, Viral, Pest, Healthy, Unknown)
4. Common Names (English, Urdu, Pashto)
5. Similar Diseases (Differential Diagnosis candidates)

Outputs:
- ml_pipeline/data/disease_identity.json
- ml_pipeline/ANALYSIS_COMPLETE/reports/disease_mapping_table.txt
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# Directory Paths
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
REPORTS_DIR = SCRIPT_DIR / "ANALYSIS_COMPLETE" / "reports"
INPUT_CLASS_MAP = DATA_DIR / "class_map_final.json"
OUTPUT_JSON = DATA_DIR / "disease_identity.json"
OUTPUT_TABLE = REPORTS_DIR / "disease_mapping_table.txt"

# Master Taxonomy Knowledge Base for 67 Head Classes
DISEASE_KNOWLEDGE_BASE: dict[str, dict] = {
    "Apple_Black_Spot": {
        "crop": "Apple",
        "scientific_name": "Venturia inaequalis",
        "pathogen_type": "Fungal",
        "common_name": {
            "english": "Apple Black Spot / Scab",
            "urdu": "سیب کے سیاہ دھبے / اسکایب",
            "pashto": "د مڼو تور ټاپي",
        },
        "similar_diseases": ["Apple_Brown_Spot", "Pear_Black_Spot"],
    },
    "Apple_Brown_Spot": {
        "crop": "Apple",
        "scientific_name": "Marssonina coronaria",
        "pathogen_type": "Fungal",
        "common_name": {
            "english": "Marssonina Leaf Blotch / Brown Spot",
            "urdu": "سیب کے بھورے دھبے",
            "pashto": "د مڼو نسواري ټاپي",
        },
        "similar_diseases": ["Apple_Black_Spot"],
    },
    "Apple_Unknown": {
        "crop": "Apple",
        "scientific_name": "Malus domestica (Unspecified Disorder)",
        "pathogen_type": "Unknown",
        "common_name": {
            "english": "Unspecified Apple Leaf Damage",
            "urdu": "سیب کا نامعلوم عارضہ",
            "pashto": "د مڼو نامعلومه ناروغي",
        },
        "similar_diseases": ["Apple_Black_Spot", "Apple_Brown_Spot"],
    },
    "Apricot_Blight": {
        "crop": "Apricot",
        "scientific_name": "Monilinia laxa",
        "pathogen_type": "Fungal",
        "common_name": {
            "english": "Apricot Brown Rot / Blight",
            "urdu": "خوبانی کا جلاؤ",
            "pashto": "د زردالو وسوځیدنه",
        },
        "similar_diseases": ["Apricot_Shot_Hole"],
    },
    "Apricot_Shot_Hole": {
        "crop": "Apricot",
        "scientific_name": "Wilsonomyces carpophilus",
        "pathogen_type": "Fungal",
        "common_name": {
            "english": "Shot Hole Disease",
            "urdu": "خوبانی کے پتے کے سوراخ",
            "pashto": "د زردالو شاټ هول",
        },
        "similar_diseases": ["Apricot_Blight", "Walnut_Shot_Hole"],
    },
    "Apricot_Unknown": {
        "crop": "Apricot",
        "scientific_name": "Prunus armeniaca (Unspecified Disorder)",
        "pathogen_type": "Unknown",
        "common_name": {
            "english": "Unspecified Apricot Leaf Damage",
            "urdu": "خوبانی کا نامعلوم عارضہ",
            "pashto": "د زردالو نامعلومه ناروغي",
        },
        "similar_diseases": ["Apricot_Blight", "Apricot_Shot_Hole"],
    },
    "Bean_Fungal": {
        "crop": "Bean",
        "scientific_name": "Colletotrichum lindemuthianum / Cercospora spp.",
        "pathogen_type": "Fungal",
        "common_name": {
            "english": "Bean Fungal Leaf Spot",
            "urdu": "پھلی کا فنگس عارضہ",
            "pashto": "د لوبیا فنګسي ناروغي",
        },
        "similar_diseases": ["Bean_Rust", "Bean_Shot_Hole"],
    },
    "Bean_Rust": {
        "crop": "Bean",
        "scientific_name": "Uromyces appendiculatus",
        "pathogen_type": "Fungal",
        "common_name": {
            "english": "Bean Rust",
            "urdu": "پھلی کا زنگ",
            "pashto": "د لوبیا زنګ",
        },
        "similar_diseases": ["Bean_Fungal", "Fig_Rust"],
    },
    "Bean_Shot_Hole": {
        "crop": "Bean",
        "scientific_name": "Pseudocercospora / Stigmina spp.",
        "pathogen_type": "Fungal",
        "common_name": {
            "english": "Bean Shot Hole",
            "urdu": "پھلی کے پتے کے سوراخ",
            "pashto": "د لوبیا سوري لرونکي ټاپي",
        },
        "similar_diseases": ["Bean_Fungal", "Apricot_Shot_Hole"],
    },
    "Bean_Unknown": {
        "crop": "Bean",
        "scientific_name": "Phaseolus vulgaris (Unspecified Disorder)",
        "pathogen_type": "Unknown",
        "common_name": {
            "english": "Unspecified Bean Leaf Damage",
            "urdu": "پھلی کا نامعلوم عارضہ",
            "pashto": "د لوبیا نامعلومه ناروغي",
        },
        "similar_diseases": ["Bean_Fungal", "Bean_Rust"],
    },
    "Cherry_Brown_Spot": {
        "crop": "Cherry",
        "scientific_name": "Blumeriella jaapii",
        "pathogen_type": "Fungal",
        "common_name": {
            "english": "Cherry Leaf Spot / Brown Spot",
            "urdu": "چیری کے بھورے دھبے",
            "pashto": "د چیری نسواري ټاپي",
        },
        "similar_diseases": ["Cherry_Purple_Spot", "Cherry_Shot_Hole"],
    },
    "Cherry_Purple_Spot": {
        "crop": "Cherry",
        "scientific_name": "Cercospora circumscissa",
        "pathogen_type": "Fungal",
        "common_name": {
            "english": "Cherry Purple Spot",
            "urdu": "چیری کے جامنی دھبے",
            "pashto": "د چیری ارغواني ټاپي",
        },
        "similar_diseases": ["Cherry_Brown_Spot", "Cherry_Scorch"],
    },
    "Cherry_Scorch": {
        "crop": "Cherry",
        "scientific_name": "Gnomonia erythrostoma / Xylella fastidiosa",
        "pathogen_type": "Fungal / Bacterial",
        "common_name": {
            "english": "Cherry Leaf Scorch",
            "urdu": "چیری کے پتے کا جلساؤ",
            "pashto": "د چیری د پاڼو سوځیدنه",
        },
        "similar_diseases": ["Cherry_Brown_Spot", "Cherry_Purple_Spot"],
    },
    "Cherry_Shot_Hole": {
        "crop": "Cherry",
        "scientific_name": "Wilsonomyces carpophilus",
        "pathogen_type": "Fungal",
        "common_name": {
            "english": "Cherry Shot Hole",
            "urdu": "چیری کے پتے کے سوراخ",
            "pashto": "د چیری شاټ هول",
        },
        "similar_diseases": ["Cherry_Brown_Spot", "Apricot_Shot_Hole"],
    },
    "Cherry_Unknown": {
        "crop": "Cherry",
        "scientific_name": "Prunus avium (Unspecified Disorder)",
        "pathogen_type": "Unknown",
        "common_name": {
            "english": "Unspecified Cherry Leaf Damage",
            "urdu": "چیری کا نامعلوم عارضہ",
            "pashto": "د چیری نامعلومه ناروغي",
        },
        "similar_diseases": ["Cherry_Brown_Spot", "Cherry_Purple_Spot"],
    },
    "Corn_Fungal": {
        "crop": "Corn",
        "scientific_name": "Bipolaris / Exserohilum spp.",
        "pathogen_type": "Fungal",
        "common_name": {
            "english": "Corn Fungal Leaf Blight",
            "urdu": "مکئی کا فنگس عارضہ",
            "pashto": "د جوارو فنګسي ناروغي",
        },
        "similar_diseases": ["Corn_Gray_Spot", "Corn_Holcus_Spot"],
    },
    "Corn_Gray_Spot": {
        "crop": "Corn",
        "scientific_name": "Cercospora zeae-maydis",
        "pathogen_type": "Fungal",
        "common_name": {
            "english": "Gray Leaf Spot of Corn",
            "urdu": "مکئی کے خاکستری دھبے",
            "pashto": "د جوارو خړ ټاپي",
        },
        "similar_diseases": ["Corn_Fungal", "Corn_Holcus_Spot"],
    },
    "Corn_Holcus_Spot": {
        "crop": "Corn",
        "scientific_name": "Pseudomonas syringae pv. lapsa",
        "pathogen_type": "Bacterial",
        "common_name": {
            "english": "Holcus Bacterial Spot",
            "urdu": "ہولکس بیکٹیریل دھبے",
            "pashto": "د جوارو ہولکس باکتریایي ټاپي",
        },
        "similar_diseases": ["Corn_Gray_Spot", "Corn_Fungal"],
    },
    "Corn_Unknown": {
        "crop": "Corn",
        "scientific_name": "Zea mays (Unspecified Disorder)",
        "pathogen_type": "Unknown",
        "common_name": {
            "english": "Unspecified Corn Leaf Damage",
            "urdu": "مکئی کا نامعلوم عارضہ",
            "pashto": "د جوارو نامعلومه ناروغي",
        },
        "similar_diseases": ["Corn_Gray_Spot", "Corn_Fungal"],
    },
    "Fig_Blight": {
        "crop": "Fig",
        "scientific_name": "Pellicularia koleroga",
        "pathogen_type": "Fungal",
        "common_name": {
            "english": "Fig Thread Blight",
            "urdu": "انجیر کا بلاسٹ یا جلاؤ",
            "pashto": "د انځر سوځیدنه",
        },
        "similar_diseases": ["Fig_Brown_Spot", "Fig_Rust"],
    },
    "Fig_Brown_Spot": {
        "crop": "Fig",
        "scientific_name": "Cercospora fici",
        "pathogen_type": "Fungal",
        "common_name": {
            "english": "Fig Leaf Spot / Brown Spot",
            "urdu": "انجیر کے بھورے دھبے",
            "pashto": "د انځر نسواري ټاپي",
        },
        "similar_diseases": ["Fig_Blight", "Fig_Rust"],
    },
    "Fig_Rust": {
        "crop": "Fig",
        "scientific_name": "Cerotelium fici",
        "pathogen_type": "Fungal",
        "common_name": {
            "english": "Fig Rust",
            "urdu": "انجیر کا زنگ",
            "pashto": "د انځر زنګ",
        },
        "similar_diseases": ["Fig_Brown_Spot", "Fig_Blight"],
    },
    "Fig_Unknown": {
        "crop": "Fig",
        "scientific_name": "Ficus carica (Unspecified Disorder)",
        "pathogen_type": "Unknown",
        "common_name": {
            "english": "Unspecified Fig Leaf Damage",
            "urdu": "انجیر کا نامعلوم عارضہ",
            "pashto": "د انځر نامعلومه ناروغي",
        },
        "similar_diseases": ["Fig_Brown_Spot", "Fig_Blight"],
    },
    "Grape_Anthracnose": {
        "crop": "Grape",
        "scientific_name": "Elsinoë ampelina",
        "pathogen_type": "Fungal",
        "common_name": {
            "english": "Grape Anthracnose / Bird's Eye Rot",
            "urdu": "انگور کا اینتھراکنوز",
            "pashto": "د انګورو انترکنوز",
        },
        "similar_diseases": ["Grape_Brown_Spot", "Grape_Shot_Hole"],
    },
    "Grape_Brown_Spot": {
        "crop": "Grape",
        "scientific_name": "Pseudocercospora vitis",
        "pathogen_type": "Fungal",
        "common_name": {
            "english": "Grape Leaf Spot / Brown Spot",
            "urdu": "انگور کے بھورے دھبے",
            "pashto": "د انګورو نسواري ټاپي",
        },
        "similar_diseases": ["Grape_Anthracnose", "Grape_Downy_Mildew"],
    },
    "Grape_Downy_Mildew": {
        "crop": "Grape",
        "scientific_name": "Plasmopara viticola",
        "pathogen_type": "Fungal",
        "common_name": {
            "english": "Grape Downy Mildew",
            "urdu": "انگور کا ڈاؤن ہی ملڈیو",
            "pashto": "د انګورو ډاوني ملډیو",
        },
        "similar_diseases": ["Grape_Powdery_Mildew", "Grape_Brown_Spot"],
    },
    "Grape_Mites": {
        "crop": "Grape",
        "scientific_name": "Colomerus vitis",
        "pathogen_type": "Pest",
        "common_name": {
            "english": "Grape Erinose Mite / Blister Mite",
            "urdu": "انگور کی مائٹس",
            "pashto": "د انګورو ژوي یا مایټس",
        },
        "similar_diseases": ["Grape_Powdery_Mildew", "Tomato_Spider_Mites"],
    },
    "Grape_Powdery_Mildew": {
        "crop": "Grape",
        "scientific_name": "Erysiphe necator",
        "pathogen_type": "Fungal",
        "common_name": {
            "english": "Grape Powdery Mildew",
            "urdu": "انگور کا پاؤڈری ملڈیو",
            "pashto": "د انګورو پاوډري ملډیو",
        },
        "similar_diseases": ["Grape_Downy_Mildew", "Wheat_Mildew"],
    },
    "Grape_Shot_Hole": {
        "crop": "Grape",
        "scientific_name": "Phyllosticta ampelicida",
        "pathogen_type": "Fungal",
        "common_name": {
            "english": "Grape Shot Hole / Black Rot Leaf Spot",
            "urdu": "انگور کے پتے کے سوراخ",
            "pashto": "د انګورو سوري لرونکي ټاپي",
        },
        "similar_diseases": ["Grape_Anthracnose", "Grape_Brown_Spot"],
    },
    "Grape_Unknown": {
        "crop": "Grape",
        "scientific_name": "Vitis vinifera (Unspecified Disorder)",
        "pathogen_type": "Unknown",
        "common_name": {
            "english": "Unspecified Grape Leaf Damage",
            "urdu": "انگور کا نامعلوم عارضہ",
            "pashto": "د انګورو نامعلومه ناروغي",
        },
        "similar_diseases": ["Grape_Brown_Spot", "Grape_Downy_Mildew"],
    },
    "Lokat_Healthy": {
        "crop": "Lokat",
        "scientific_name": "Eriobotrya japonica",
        "pathogen_type": "Healthy",
        "common_name": {
            "english": "Healthy Loquat Leaf",
            "urdu": "لوکاٹ کا صحت مند پتا",
            "pashto": "د لوکاټ روغه پاڼه",
        },
        "similar_diseases": [],
    },
    "Lokat_Leaf_Spot": {
        "crop": "Lokat",
        "scientific_name": "Entomosporium mespili",
        "pathogen_type": "Fungal",
        "common_name": {
            "english": "Loquat Entomosporium Leaf Spot",
            "urdu": "لوکاٹ کے پتے کے دھبے",
            "pashto": "د لوکاټ د پاڼو ټاپي",
        },
        "similar_diseases": ["Pear_Black_Spot", "Apple_Black_Spot"],
    },
    "Pear_Black_Spot": {
        "crop": "Pear",
        "scientific_name": "Alternaria gaisen / Venturia pirina",
        "pathogen_type": "Fungal",
        "common_name": {
            "english": "Pear Black Spot / Scab",
            "urdu": "ناشپاتی کے سیاہ دھبے",
            "pashto": "د ناک تور ټاپي",
        },
        "similar_diseases": ["Apple_Black_Spot", "Lokat_Leaf_Spot"],
    },
    "Pear_Fire_Blight": {
        "crop": "Pear",
        "scientific_name": "Erwinia amylovora",
        "pathogen_type": "Bacterial",
        "common_name": {
            "english": "Fire Blight of Pear",
            "urdu": "ناشپاتی کا فائر بلائٹ",
            "pashto": "د ناک د اور سوځیدنه",
        },
        "similar_diseases": ["Apple_Black_Spot", "Pear_Black_Spot"],
    },
    "Pear_Unknown": {
        "crop": "Pear",
        "scientific_name": "Pyrus communis (Unspecified Disorder)",
        "pathogen_type": "Unknown",
        "common_name": {
            "english": "Unspecified Pear Leaf Damage",
            "urdu": "ناشپاتی کا نامعلوم عارضہ",
            "pashto": "د ناک نامعلومه ناروغي",
        },
        "similar_diseases": ["Pear_Black_Spot", "Pear_Fire_Blight"],
    },
    "Persimmons_Brown_Spot": {
        "crop": "Persimmon",
        "scientific_name": "Cercospora kakivora / Mycosphaerella nawae",
        "pathogen_type": "Fungal",
        "common_name": {
            "english": "Persimmon Angular Leaf Spot",
            "urdu": "املوک کے بھورے دھبے",
            "pashto": "د املوک نسواري ټاپي",
        },
        "similar_diseases": ["Fig_Brown_Spot", "Apple_Brown_Spot"],
    },
    "Tomato_Bacterial_Spot": {
        "crop": "Tomato",
        "scientific_name": "Xanthomonas perforans / vesicatoria",
        "pathogen_type": "Bacterial",
        "common_name": {
            "english": "Tomato Bacterial Spot",
            "urdu": "ٹماٹر کے بیکٹیریل دھبے",
            "pashto": "د ټماټرو باکتریایي ټاپي",
        },
        "similar_diseases": ["Tomato_Septoria", "Tomato_Early_Blight"],
    },
    "Tomato_Curl": {
        "crop": "Tomato",
        "scientific_name": "Tomato Yellow Leaf Curl Virus (TYLCV)",
        "pathogen_type": "Viral",
        "common_name": {
            "english": "Tomato Yellow Leaf Curl",
            "urdu": "ٹماٹر کا پتا موڑ وائرس",
            "pashto": "د ټماټرو د پاڼو پیچلتیا وایرس",
        },
        "similar_diseases": ["Tomato_Healthy"],
    },
    "Tomato_Early_Blight": {
        "crop": "Tomato",
        "scientific_name": "Alternaria solani",
        "pathogen_type": "Fungal",
        "common_name": {
            "english": "Tomato Early Blight",
            "urdu": "ٹماٹر کا اگیتا جھلساؤ",
            "pashto": "د ټماټرو دمخه سوځیدنه",
        },
        "similar_diseases": ["Tomato_Late_Blight", "Tomato_Septoria"],
    },
    "Tomato_Fusarium_Wilt": {
        "crop": "Tomato",
        "scientific_name": "Fusarium oxysporum f. sp. lycopersici",
        "pathogen_type": "Fungal",
        "common_name": {
            "english": "Tomato Fusarium Wilt",
            "urdu": "ٹماٹر کا فیوزیریم مرحضا",
            "pashto": "د ټماټرو فیوزیریم مرضاوی",
        },
        "similar_diseases": ["Tomato_Verticillium_Wilt", "Tomato_Late_Blight"],
    },
    "Tomato_Healthy": {
        "crop": "Tomato",
        "scientific_name": "Solanum lycopersicum",
        "pathogen_type": "Healthy",
        "common_name": {
            "english": "Healthy Tomato Leaf",
            "urdu": "ٹماٹر کا صحت مند پتا",
            "pashto": "د ټماټرو روغه پاڼه",
        },
        "similar_diseases": [],
    },
    "Tomato_Late_Blight": {
        "crop": "Tomato",
        "scientific_name": "Phytophthora infestans",
        "pathogen_type": "Fungal",
        "common_name": {
            "english": "Tomato Late Blight",
            "urdu": "ٹماٹر کا پچھیتا جھلساؤ",
            "pashto": "د ټماټرو وروسته سوځیدنه",
        },
        "similar_diseases": ["Tomato_Early_Blight", "Tomato_Fusarium_Wilt"],
    },
    "Tomato_Miner": {
        "crop": "Tomato",
        "scientific_name": "Tuta absoluta / Liriomyza sativae",
        "pathogen_type": "Pest",
        "common_name": {
            "english": "Tomato Leafminer",
            "urdu": "ٹماٹر کا لیف مائنر",
            "pashto": "د ټماټرو ليکنکی چنجی",
        },
        "similar_diseases": ["Tomato_Septoria"],
    },
    "Tomato_Mold": {
        "crop": "Tomato",
        "scientific_name": "Passalora fulva (Cladosporium fulvum)",
        "pathogen_type": "Fungal",
        "common_name": {
            "english": "Tomato Leaf Mold",
            "urdu": "ٹماٹر کی لیف مولڈ",
            "pashto": "د ټماټرو مولډ",
        },
        "similar_diseases": ["Tomato_Early_Blight", "Tomato_Septoria"],
    },
    "Tomato_Septoria": {
        "crop": "Tomato",
        "scientific_name": "Septoria lycopersici",
        "pathogen_type": "Fungal",
        "common_name": {
            "english": "Tomato Septoria Leaf Spot",
            "urdu": "ٹماٹر کے سیپٹوریا دھبے",
            "pashto": "د ټماټرو سیپټوریا ټاپي",
        },
        "similar_diseases": ["Tomato_Bacterial_Spot", "Tomato_Early_Blight"],
    },
    "Tomato_Spider_Mites": {
        "crop": "Tomato",
        "scientific_name": "Tetranychus urticae",
        "pathogen_type": "Pest",
        "common_name": {
            "english": "Two-Spotted Spider Mite",
            "urdu": "ٹماٹر کی لال مکڑی (مائٹس)",
            "pashto": "د ټماټرو دوه ټاپې وال ژوي",
        },
        "similar_diseases": ["Grape_Mites", "Wheat_Mite"],
    },
    "Tomato_Verticillium_Wilt": {
        "crop": "Tomato",
        "scientific_name": "Verticillium dahliae",
        "pathogen_type": "Fungal",
        "common_name": {
            "english": "Tomato Verticillium Wilt",
            "urdu": "ٹماٹر کا ورٹیسیلیم مرجھاؤ",
            "pashto": "د ټماټرو ورټیسیلیم مرضاوی",
        },
        "similar_diseases": ["Tomato_Fusarium_Wilt", "Tomato_Early_Blight"],
    },
    "Walnut_Anthracnose": {
        "crop": "Walnut",
        "scientific_name": "Ophiognomonia leptostyla",
        "pathogen_type": "Fungal",
        "common_name": {
            "english": "Walnut Anthracnose / Leaf Blotch",
            "urdu": "اخروٹ کا اینتھراکنوز",
            "pashto": "د اخروټ انترکنوز",
        },
        "similar_diseases": ["Walnut_Blotch", "Walnut_Shot_Hole"],
    },
    "Walnut_Blotch": {
        "crop": "Walnut",
        "scientific_name": "Xanthomonas arboricola pv. juglandis",
        "pathogen_type": "Bacterial",
        "common_name": {
            "english": "Walnut Bacterial Blight / Blotch",
            "urdu": "اخروٹ کا بیکٹیریل بلائٹ",
            "pashto": "د اخروټ باکتریایي ټاپي",
        },
        "similar_diseases": ["Walnut_Anthracnose", "Walnut_Shot_Hole"],
    },
    "Walnut_Gall_Mite": {
        "crop": "Walnut",
        "scientific_name": "Aceria erinea",
        "pathogen_type": "Pest",
        "common_name": {
            "english": "Walnut Blister Gall Mite",
            "urdu": "اخروٹ کی گال مائٹ",
            "pashto": "د اخروټ چنجی يا مایټ",
        },
        "similar_diseases": ["Grape_Mites"],
    },
    "Walnut_Shot_Hole": {
        "crop": "Walnut",
        "scientific_name": "Wilsonomyces / Gnomonia spp.",
        "pathogen_type": "Fungal",
        "common_name": {
            "english": "Walnut Shot Hole",
            "urdu": "اخروٹ کے پتے کے سوراخ",
            "pashto": "د اخروټ شاټ هول",
        },
        "similar_diseases": ["Walnut_Anthracnose", "Apricot_Shot_Hole"],
    },
    "Walnut_Unknown": {
        "crop": "Walnut",
        "scientific_name": "Juglans regia (Unspecified Disorder)",
        "pathogen_type": "Unknown",
        "common_name": {
            "english": "Unspecified Walnut Leaf Damage",
            "urdu": "اخروٹ کا نامعلوم عارضہ",
            "pashto": "د اخروټ نامعلومه ناروغي",
        },
        "similar_diseases": ["Walnut_Anthracnose", "Walnut_Blotch"],
    },
    "Wheat_Aphid": {
        "crop": "Wheat",
        "scientific_name": "Rhopalosiphum padi / Sitobion avenae",
        "pathogen_type": "Pest",
        "common_name": {
            "english": "Cereal Aphid",
            "urdu": "گندم کا تیلا (ایفڈ)",
            "pashto": "د غنمو شین چنجی (ایفډ)",
        },
        "similar_diseases": ["Wheat_Mite"],
    },
    "Wheat_Black_Rust": {
        "crop": "Wheat",
        "scientific_name": "Puccinia graminis f. sp. tritici",
        "pathogen_type": "Fungal",
        "common_name": {
            "english": "Black Rust / Stem Rust",
            "urdu": "گندم کا سیاہ رتُوا (کنگئی)",
            "pashto": "د غنمو توره کنګه",
        },
        "similar_diseases": ["Wheat_Brown_Rust", "Wheat_Yellow_Rust"],
    },
    "Wheat_Blast": {
        "crop": "Wheat",
        "scientific_name": "Magnaporthe oryzae pathotype Triticum",
        "pathogen_type": "Fungal",
        "common_name": {
            "english": "Wheat Blast",
            "urdu": "گندم کا بلاسٹ",
            "pashto": "د غنمو بلاست ناروغي",
        },
        "similar_diseases": ["Wheat_Fusarium_Head_Blight", "Wheat_Tan_Spot"],
    },
    "Wheat_Brown_Rust": {
        "crop": "Wheat",
        "scientific_name": "Puccinia triticina",
        "pathogen_type": "Fungal",
        "common_name": {
            "english": "Brown Rust / Leaf Rust",
            "urdu": "گندم کا بھورا رتُوا (کنگئی)",
            "pashto": "د غنمو نسواري کنګه",
        },
        "similar_diseases": ["Wheat_Black_Rust", "Wheat_Yellow_Rust"],
    },
    "Wheat_Common_Root_Rot": {
        "crop": "Wheat",
        "scientific_name": "Bipolaris sorokiniana / Fusarium spp.",
        "pathogen_type": "Fungal",
        "common_name": {
            "english": "Common Root Rot of Wheat",
            "urdu": "گندم جڑ کا گلنا",
            "pashto": "د غنمو د روټ روټ ناروغي",
        },
        "similar_diseases": ["Wheat_Fusarium_Head_Blight", "Wheat_Leaf_Blight"],
    },
    "Wheat_Fusarium_Head_Blight": {
        "crop": "Wheat",
        "scientific_name": "Fusarium graminearum",
        "pathogen_type": "Fungal",
        "common_name": {
            "english": "Fusarium Head Blight / Scab",
            "urdu": "گندم کا فیوزیریم سٹا جھلساؤ",
            "pashto": "د غنمو فیوزیریم هډ بلایټ",
        },
        "similar_diseases": ["Wheat_Blast", "Wheat_Common_Root_Rot"],
    },
    "Wheat_Healthy": {
        "crop": "Wheat",
        "scientific_name": "Triticum aestivum",
        "pathogen_type": "Healthy",
        "common_name": {
            "english": "Healthy Wheat Plant",
            "urdu": "گندم کا صحت مند پودا",
            "pashto": "د غنمو روغ بوټی",
        },
        "similar_diseases": [],
    },
    "Wheat_Leaf_Blight": {
        "crop": "Wheat",
        "scientific_name": "Bipolaris sorokiniana",
        "pathogen_type": "Fungal",
        "common_name": {
            "english": "Helminthosporium Leaf Blight / Spot Blotch",
            "urdu": "گندم کا پتا جھلساؤ",
            "pashto": "د غنمو د پاڼو سوځیدنه",
        },
        "similar_diseases": ["Wheat_Tan_Spot", "Wheat_Septoria"],
    },
    "Wheat_Mildew": {
        "crop": "Wheat",
        "scientific_name": "Blumeria graminis f. sp. tritici",
        "pathogen_type": "Fungal",
        "common_name": {
            "english": "Powdery Mildew of Wheat",
            "urdu": "گندم کا سفید فپھوندی (پاؤڈری ملڈیو)",
            "pashto": "د غنمو پاوډري ملډیو",
        },
        "similar_diseases": ["Grape_Powdery_Mildew"],
    },
    "Wheat_Mite": {
        "crop": "Wheat",
        "scientific_name": "Petrobia latens",
        "pathogen_type": "Pest",
        "common_name": {
            "english": "Brown Wheat Mite",
            "urdu": "گندم کی جوئیں/مائٹ",
            "pashto": "د غنمو نسواري مایټ",
        },
        "similar_diseases": ["Wheat_Aphid", "Tomato_Spider_Mites"],
    },
    "Wheat_Septoria": {
        "crop": "Wheat",
        "scientific_name": "Zymoseptoria tritici",
        "pathogen_type": "Fungal",
        "common_name": {
            "english": "Septoria Tritici Blotch",
            "urdu": "گندم کا سیپٹوریا پتا دھبہ",
            "pashto": "د غنمو سیپټوریا ټاپي",
        },
        "similar_diseases": ["Wheat_Leaf_Blight", "Wheat_Tan_Spot"],
    },
    "Wheat_Smut": {
        "crop": "Wheat",
        "scientific_name": "Ustilago tritici",
        "pathogen_type": "Fungal",
        "common_name": {
            "english": "Loose Smut of Wheat",
            "urdu": "گندم کا کاجل (کنگئی / کاں یاری)",
            "pashto": "د غنمو لوز سټم ناروغي",
        },
        "similar_diseases": ["Wheat_Black_Rust", "Wheat_Fusarium_Head_Blight"],
    },
    "Wheat_Stem_Fly": {
        "crop": "Wheat",
        "scientific_name": "Atherigona soccata / Chlorops pumilionis",
        "pathogen_type": "Pest",
        "common_name": {
            "english": "Wheat Stem Fly / Gout Fly",
            "urdu": "گندم کا تنے کی مکھی",
            "pashto": "د غنمو د ډډ مچۍ",
        },
        "similar_diseases": ["Wheat_Aphid"],
    },
    "Wheat_Tan_Spot": {
        "crop": "Wheat",
        "scientific_name": "Pyrenophora tritici-repentis",
        "pathogen_type": "Fungal",
        "common_name": {
            "english": "Tan Spot / Yellow Leaf Spot",
            "urdu": "گندم کا پیلا پتا دھبہ (ٹین اسپاٹ)",
            "pashto": "د غنمو د ژیړ پاني ټاپي",
        },
        "similar_diseases": ["Wheat_Septoria", "Wheat_Leaf_Blight"],
    },
    "Wheat_Yellow_Rust": {
        "crop": "Wheat",
        "scientific_name": "Puccinia striiformis f. sp. tritici",
        "pathogen_type": "Fungal",
        "common_name": {
            "english": "Yellow Rust / Stripe Rust",
            "urdu": "گندم کا پیلا رتُوا (زرد کنگئی)",
            "pashto": "د غنمو ژیړه کنګه",
        },
        "similar_diseases": ["Wheat_Brown_Rust", "Wheat_Black_Rust"],
    },
}


def main() -> None:
    print("=" * 75)
    print("  ZARI.ai — SCIENTIFIC NOMENCLATURE & DISEASE IDENTITY MAPPING")
    print("=" * 75)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_CLASS_MAP.exists():
        raise FileNotFoundError(f"Missing master class map JSON at {INPUT_CLASS_MAP}")

    with open(INPUT_CLASS_MAP, "r", encoding="utf-8") as f:
        class_map_data = json.load(f)

    head_classes_dict = class_map_data.get("head_classes", {})
    sorted_head_classes = sorted(head_classes_dict.items(), key=lambda x: x[1])

    num_classes = len(sorted_head_classes)
    print(f"\n✓ Found {num_classes} Head Field Classes in master mapping.")
    assert num_classes == 67, f"Expected 67 head classes, got {num_classes}"

    # Build final identity mapping payload
    identity_mapping: dict[str, dict] = {}

    for cname, cid in sorted_head_classes:
        if cname in DISEASE_KNOWLEDGE_BASE:
            info = DISEASE_KNOWLEDGE_BASE[cname]
        else:
            # Fallback for unexpected naming
            crop = cname.split("_")[0]
            info = {
                "crop": crop,
                "scientific_name": f"{crop} disorder",
                "pathogen_type": "Unknown",
                "common_name": {
                    "english": cname.replace("_", " "),
                    "urdu": f"{crop} عارضہ",
                    "pashto": f"{crop} ناروغي",
                },
                "similar_diseases": [],
            }

        identity_mapping[cname] = {
            "class_id": cid,
            "crop": info["crop"],
            "scientific_name": info["scientific_name"],
            "pathogen_type": info["pathogen_type"],
            "common_name": info["common_name"],
            "similar_diseases": info["similar_diseases"],
        }

    # Save JSON Payload
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(identity_mapping, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved master disease identity JSON to: {OUTPUT_JSON}")

    # Build ASCII Summary Table
    table_header = f"{'ID':<3} | {'Class Name':<28} | {'Crop':<10} | {'Scientific Name':<38} | {'Type':<8}"
    separator = "-" * len(table_header)

    table_rows = [
        "==========================================================================================================",
        "ZARI.ai — MASTER DISEASE NOMENCLATURE & PATHOGEN REGISTRY (67 HEAD CLASSES)",
        "==========================================================================================================",
        table_header,
        separator,
    ]

    terminal_rows = [table_header, separator]

    for cname, data in identity_mapping.items():
        cid = data["class_id"]
        crop = data["crop"]
        sname = data["scientific_name"]
        ptype = data["pathogen_type"]

        row_str = f"{cid:<3} | {cname:<28} | {crop:<10} | {sname:<38} | {ptype:<8}"
        table_rows.append(row_str)
        terminal_rows.append(row_str)

    table_rows.append(separator)
    terminal_rows.append(separator)

    # Save ASCII Table Report
    OUTPUT_TABLE.write_text("\n".join(table_rows), encoding="utf-8")
    print(f"✓ Saved ASCII disease mapping table to: {OUTPUT_TABLE}")

    # Print summary table to terminal
    print("\n" + "\n".join(terminal_rows))
    print("\n✅ MASTER DISEASE IDENTITY MAPPING COMPLETE FOR ALL 67 CLASSES!")


if __name__ == "__main__":
    main()
