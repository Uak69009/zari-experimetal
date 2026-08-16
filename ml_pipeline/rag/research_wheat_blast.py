"""ZARI.ai — Wheat Blast High Caution Evidence Research & Specialized RAG Chunking.

Specialized handling for Wheat Blast (Magnaporthe oryzae pathotype Triticum):
- Urgency       : "CRITICAL — confirm diagnosis immediately"
- Protocol      : 7-Step Action Plan (Confirmation -> Quarantine -> Seed -> Residue -> Resistance -> Preventive Chem -> Extension Alert)
- Strict Rules  : Fungicides provide PARTIAL protection only and must be applied PREVENTIVELY.
                  Never suggest 'spray and it will be fine'.
                  Do NOT open-burn infected fields without extension guidance (spore uplift hazard).

Output:
- ml_pipeline/data/chunks_wheat_blast.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
OUTPUT_JSON = DATA_DIR / "chunks_wheat_blast.json"

# Specialized High Caution Chunks for Wheat Blast
WHEAT_BLAST_CHUNKS: list[dict] = [
    {
        "chunk_id": "WHEAT_BLAST_IDENTITY",
        "disease_id": "WHEAT_BLAST",
        "disease_class": "Wheat_Blast",
        "crop": "Wheat",
        "country": "Pakistan",
        "province": "All",
        "section": "identity",
        "urgency": "CRITICAL — confirm diagnosis immediately",
        "evidence_level": "A1",
        "verified": True,
        "parent_id": "WHEAT_BLAST",
        "text": (
            "URGENCY: CRITICAL — confirm diagnosis immediately. Wheat Blast is caused by the fungal pathogen "
            "Magnaporthe oryzae pathotype Triticum (MoT). Common names: Wheat Blast (en), گندم کا بلاسٹ (ur), "
            "د غنمو بلاست ناروغي (ps). It is a devastating quarantine threat capable of causing up to 100% crop loss. "
            "Visual identification alone is inconclusive; molecular or lab culture diagnosis is mandatory."
        ),
        "source_organization": "CIMMYT Wheat Blast Advisory / FAO Biosecurity Guidelines",
        "url": "https://www.cimmyt.org/wheat-blast",
    },
    {
        "chunk_id": "WHEAT_BLAST_SYMPTOMS",
        "disease_id": "WHEAT_BLAST",
        "disease_class": "Wheat_Blast",
        "crop": "Wheat",
        "country": "Pakistan",
        "province": "All",
        "section": "symptoms",
        "urgency": "CRITICAL — confirm diagnosis immediately",
        "evidence_level": "A2",
        "verified": True,
        "parent_id": "WHEAT_BLAST",
        "text": (
            "Symptoms of Wheat Blast: Bleached, straw-colored spikelets or entire heads above the point of infection "
            "while lower leaves remain green. Gray-white lesions on spikes with a characteristic dark blackened "
            "rachis junction. Shriveled, deformed, light-weight grains or complete grain failure. Diamond-shaped "
            "tan lesions with dark reddish-brown borders appear on lower leaves under high relative humidity."
        ),
        "source_organization": "CIMMYT / CABI Plantwise / USDA ARS",
        "url": "https://www.cimmyt.org/wheat-blast",
    },
    {
        "chunk_id": "WHEAT_BLAST_EPIDEMIOLOGY",
        "disease_id": "WHEAT_BLAST",
        "disease_class": "Wheat_Blast",
        "crop": "Wheat",
        "country": "Pakistan",
        "province": "All",
        "section": "epidemiology",
        "urgency": "CRITICAL — confirm diagnosis immediately",
        "evidence_level": "A2",
        "verified": True,
        "parent_id": "WHEAT_BLAST",
        "text": (
            "Epidemiology and Spread: Magnaporthe oryzae pathotype Triticum spreads through infected seed, crop "
            "residue, and airborne conidia spores. Spores can travel tens of kilometers on wind currents. The pathogen "
            "can survive on alternate grass hosts (e.g., Digitaria, Eleusine, Cenchrus species) during off-seasons."
        ),
        "source_organization": "CIMMYT Wheat Blast Surveillance Center",
        "url": "https://www.cimmyt.org/wheat-blast",
    },
    {
        "chunk_id": "WHEAT_BLAST_RISK",
        "disease_id": "WHEAT_BLAST",
        "disease_class": "Wheat_Blast",
        "crop": "Wheat",
        "country": "Pakistan",
        "province": "All",
        "section": "risk",
        "urgency": "CRITICAL — confirm diagnosis immediately",
        "evidence_level": "A2",
        "verified": True,
        "parent_id": "WHEAT_BLAST",
        "text": (
            "Environmental Risk Conditions: High humidity (> 90%), prolonged rainy spells or heavy dew, and warm "
            "temperatures (25-30°C) during the heading and flowering stages (GS55-65) create maximum blast outbreak risk. "
            "Late-sown wheat crops exposed to warm spring rains face elevated vulnerability."
        ),
        "source_organization": "CIMMYT / FAO Advisory",
        "url": "https://www.cimmyt.org/wheat-blast",
    },
    {
        "chunk_id": "WHEAT_BLAST_MANAGEMENT",
        "disease_id": "WHEAT_BLAST",
        "disease_class": "Wheat_Blast",
        "crop": "Wheat",
        "country": "Pakistan",
        "province": "All",
        "section": "management",
        "urgency": "CRITICAL — confirm diagnosis immediately",
        "evidence_level": "A2",
        "verified": True,
        "parent_id": "WHEAT_BLAST",
        "text": (
            "Recommended 7-Step Action Plan for Wheat Blast:\n"
            "Step 1: Confirm diagnosis immediately via official agricultural extension / lab sample (do NOT rely on visual only).\n"
            "Step 2: Check local outbreak status with PARC / Department of Plant Protection (DPP).\n"
            "Step 3: Seed management: Never sow uncertified seed from blast-endemic regions.\n"
            "Step 4: Residue management: Deeply incorporate or manage infected straw to break spore inoculum.\n"
            "Step 5: Plant resistant cultivars (e.g., BARI Gom 33 or locally approved resistant lines).\n"
            "Step 6: Preventive fungicide application strictly prior to symptom development if risk is high.\n"
            "Step 7: Contact local agricultural extension officers immediately to isolate affected plots."
        ),
        "source_organization": "CIMMYT / PARC / FAO Joint Protocol",
        "url": "https://www.cimmyt.org/wheat-blast",
    },
    {
        "chunk_id": "WHEAT_BLAST_CHEMICAL",
        "disease_id": "WHEAT_BLAST",
        "disease_class": "Wheat_Blast",
        "crop": "Wheat",
        "country": "Pakistan",
        "province": "All",
        "section": "chemical_control",
        "urgency": "CRITICAL — confirm diagnosis immediately",
        "evidence_level": "A2",
        "verified": True,
        "parent_id": "WHEAT_BLAST",
        "text": (
            "Chemical Control Limitations: Fungicides provide PARTIAL protection only and CANNOT cure bleached "
            "heads once symptoms appear. Never promise 'spray and it will be fine'. If locally recommended by DPP/extension, "
            "preventive mixtures of Triazole + Strobilurin (e.g., Tebuconazole + Trifloxystrobin, FRAC Groups 3+11) or "
            "Mancozeb must be applied strictly at head emergence (GS55-59) BEFORE flowering. Check product label for rates."
        ),
        "source_organization": "CIMMYT / CABI Plantwise / PARC Advisory",
        "url": "https://www.cimmyt.org/wheat-blast",
    },
    {
        "chunk_id": "WHEAT_BLAST_PAKISTAN",
        "disease_id": "WHEAT_BLAST",
        "disease_class": "Wheat_Blast",
        "crop": "Wheat",
        "country": "Pakistan",
        "province": "All",
        "section": "pakistan",
        "urgency": "CRITICAL — confirm diagnosis immediately",
        "evidence_level": "A1",
        "verified": True,
        "parent_id": "WHEAT_BLAST",
        "text": (
            "Pakistan Biosecurity & Extension Protocol: Wheat Blast is a regulated quarantine threat in Pakistan. "
            "Farmers and agronomists must immediately contact the Department of Plant Protection (DPP), PARC National "
            "Wheat Program, or local District Agriculture Extension officers upon noticing bleached heads with dark rachis. "
            "Do NOT move seeds or plant material out of suspected outbreak fields."
        ),
        "source_organization": "PARC National Wheat Program / Pakistan DPP",
        "url": "http://www.plantprotection.gov.pk",
    },
    {
        "chunk_id": "WHEAT_BLAST_SAFETY",
        "disease_id": "WHEAT_BLAST",
        "disease_class": "Wheat_Blast",
        "crop": "Wheat",
        "country": "Pakistan",
        "province": "All",
        "section": "safety",
        "urgency": "CRITICAL — confirm diagnosis immediately",
        "evidence_level": "A2",
        "verified": True,
        "parent_id": "WHEAT_BLAST",
        "text": (
            "Safety and Containment Hazards: Do NOT open-burn infected fields without official agricultural guidance, "
            "as thermal updrafts can transport viable fungal spores into high-altitude air currents, spreading the disease "
            "to neighboring districts. Disinfect machinery, harvesting equipment, boots, and tools with 70% ethanol or "
            "bleach solution before moving out of suspected fields."
        ),
        "source_organization": "FAO Biosecurity / CIMMYT Sanitation Protocol",
        "url": "https://www.cimmyt.org/wheat-blast",
    },
]


def main() -> None:
    print("=" * 75)
    print("  ZARI.ai — WHEAT BLAST HIGH CAUTION RAG CHUNKING ENGINE")
    print("=" * 75)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(WHEAT_BLAST_CHUNKS, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Generated {len(WHEAT_BLAST_CHUNKS)} specialized High-Caution chunks for Wheat Blast.")
    print(f"✓ Saved Wheat Blast chunks to: {OUTPUT_JSON}\n")

    print("=" * 75)
    print(f"{'Chunk ID':<28} | {'Section':<18} | {'Urgency Level':<25}")
    print("-" * 75)

    for chunk in WHEAT_BLAST_CHUNKS:
        cid = chunk["chunk_id"]
        sec = chunk["section"]
        urg = chunk["urgency"]
        print(f"{cid:<28} | {sec:<18} | {urg:<25}")

    print("-" * 75)
    print("✅ WHEAT BLAST HIGH CAUTION RAG CHUNKING COMPLETE!")


if __name__ == "__main__":
    main()
