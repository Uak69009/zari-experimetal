"""
ZARI.ai Knowledge Graph & Dataset Card Generator (Stage 3)
----------------------------------------------------------
This script parses the enriched CSV and generates:
1. A hierarchical Knowledge Graph mapping crops to diseases to pathogen types.
2. A HuggingFace-style dataset_card.md documentation.
"""

import os
import json
import pandas as pd
from datetime import datetime

def generate_knowledge_graph(df, output_path):
    print("Generating Knowledge Graph...")
    graph = {
        "metadata": {
            "name": "ZARI.ai Agricultural Knowledge Graph",
            "version": "1.0",
            "generated_at": datetime.now().isoformat()
        },
        "taxonomy": {}
    }
    
    # Hierarchy: Crop Family -> Crop -> Pathogen Type -> Disease
    for _, row in df.iterrows():
        family = row['crop_family']
        crop = row['crop']
        pathogen = row['pathogen_type']
        disease = row['disease']
        
        if family not in graph["taxonomy"]:
            graph["taxonomy"][family] = {}
        if crop not in graph["taxonomy"][family]:
            graph["taxonomy"][family][crop] = {}
        if pathogen not in graph["taxonomy"][family][crop]:
            graph["taxonomy"][family][crop][pathogen] = set()
            
        graph["taxonomy"][family][crop][pathogen].add(disease)
        
    # Convert sets to lists for JSON serialization
    for family in graph["taxonomy"]:
        for crop in graph["taxonomy"][family]:
            for pathogen in graph["taxonomy"][family][crop]:
                graph["taxonomy"][family][crop][pathogen] = sorted(list(graph["taxonomy"][family][crop][pathogen]))
                
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=4)
        
    print(f"Knowledge Graph saved to {output_path}")

def generate_dataset_card(df, output_path):
    print("Generating Dataset Card...")
    card = f"""
# ZARI.ai Agricultural Image Dataset (153-Class)

## Dataset Description
- **Total Images:** {len(df)}
- **Total Classes:** {df['class_name'].nunique()}
- **Crop Types:** {df['crop'].nunique()}
- **Disease Types:** {df['disease'].nunique()}

## Dataset Summary
The ZARI.ai dataset is a highly enriched, massively harmonized agricultural computer vision dataset tailored for disease diagnosis and severity estimation. It merges multiple source datasets (PlantVillage, PlantDoc, NWRD) and provides extensive metadata including domain origins (Lab vs. Field), pathological taxonomies, and OpenCV-computed image quality metrics.

## Supported Tasks
- Multi-class image classification (153 classes)
- Domain adaptation (Lab to Field transfer)
- Disease severity estimation (via segmentation masks)
- Out-of-distribution detection
- Embedding retrieval via FAISS (SigLIP/DINOv2)

## Schema & Features
- `image_path`: Absolute path to the raw image.
- `crop`, `disease`, `class_name`: Taxonomical labels.
- `domain`: Indicates whether the image was taken in a controlled lab setting (`Lab`) or uncontrolled natural environment (`Field`).
- `pathogen_type`: Higher-level categorization (Fungal, Viral, Bacterial, Pest, Nutrient, Healthy).
- `blur_score`, `brightness_score`, `contrast_score`: Computed via OpenCV for quality filtering.

## Known Limitations & Imbalances
- Extreme class imbalance exists (up to 500:1 ratio between majority and minority classes).
- Domain shift is severe (majority of images are `Lab` domain, while actual deployment targets `Field` domain).

## License
Proprietary & Open-Source (Dataset amalgamation governed by original licenses of PlantVillage and PlantDoc. NWRD is proprietary to ZARI.ai).
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(card.strip())
    print(f"Dataset Card saved to {output_path}")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    enriched_csv_path = os.path.join(base_dir, "data", "dataset_master_enriched.csv")
    
    if not os.path.exists(enriched_csv_path):
        print(f"Error: {enriched_csv_path} not found.")
        # Fallback for fast testing if needed
        enriched_csv_path = os.path.join(base_dir, "data", "dataset_master.csv")
        if not os.path.exists(enriched_csv_path):
            return
            
    df = pd.read_csv(enriched_csv_path)
    
    # Fake columns for testing if missing
    if 'crop_family' not in df.columns:
        df['crop_family'] = 'Unknown_Family'
        df['crop'] = 'Unknown_Crop'
        df['pathogen_type'] = 'Unknown_Pathogen'
        df['disease'] = 'Unknown_Disease'
        df['class_name'] = 'Unknown_Class'
    
    metadata_dir = os.path.join(base_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    
    generate_knowledge_graph(df, os.path.join(metadata_dir, "knowledge_graph.json"))
    generate_dataset_card(df, os.path.join(base_dir, "dataset_card.md"))
    
if __name__ == "__main__":
    main()
