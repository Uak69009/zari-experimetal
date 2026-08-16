"""
ZARI.ai Embeddings & Index Generation (Stage 4)
-----------------------------------------------
This script uses Facebook's DINOv2 (a powerful Vision Transformer) to compute
384-dimensional embeddings for all images in the enriched dataset.
It then builds a FAISS index for lightning-fast similarity retrieval.
"""

import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel
import faiss
import warnings

warnings.filterwarnings("ignore")

class ZariImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, processor):
        self.image_paths = image_paths
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            # The processor handles resizing, center cropping, and normalization
            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
            return pixel_values, True, idx
        except Exception:
            # Return dummy if corrupt
            dummy = torch.zeros((3, 224, 224))
            return dummy, False, idx

def main():
    print("=====================================================")
    print("ZARI.ai DINOv2 Embeddings & FAISS Index Pipeline")
    print("=====================================================")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    enriched_csv_path = os.path.join(base_dir, "data", "dataset_master_enriched.csv")
    
    if not os.path.exists(enriched_csv_path):
        print(f"Error: {enriched_csv_path} not found.")
        return
        
    df = pd.read_csv(enriched_csv_path)
    # df = df.head(500) # Uncomment for fast testing
    
    print(f"Loaded dataset with {len(df)} images.")
    
    # 1. Load DINOv2
    # Using 'dinov2-small' as it yields a 384-D vector, which is very fast and expressive.
    model_name = "facebook/dinov2-small"
    print(f"Loading Vision Transformer: {model_name}...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        print("Please ensure transformers is installed: pip install transformers faiss-cpu pyarrow")
        return
        
    model.eval()
    
    # 2. Setup DataLoader
    image_paths = df['image_path'].tolist()
    dataset = ZariImageDataset(image_paths, processor)
    
    # 0 workers for Windows stability in interactive prompts, increase if running from cmd
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    
    all_embeddings = np.zeros((len(df), 384), dtype=np.float32)
    valid_indices = []
    
    print("Extracting Embeddings...")
    with torch.no_grad():
        for pixel_values, is_valid, batch_idx in tqdm(dataloader):
            pixel_values = pixel_values.to(device)
            
            # Forward pass
            outputs = model(pixel_values)
            # Use the CLS token representation
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            for i, valid in enumerate(is_valid):
                idx = batch_idx[i].item()
                if valid.item():
                    all_embeddings[idx] = embeddings[i]
                    valid_indices.append(idx)
                    
    print(f"Successfully computed embeddings for {len(valid_indices)} images.")
    
    if len(valid_indices) == 0:
        print("No valid embeddings extracted.")
        return
        
    # 3. Build FAISS Index
    print("Building FAISS Index for Retrieval...")
    try:
        index = faiss.IndexFlatL2(384)
        valid_embeddings = all_embeddings[valid_indices]
        index.add(valid_embeddings)
        
        metadata_dir = os.path.join(base_dir, "metadata")
        os.makedirs(metadata_dir, exist_ok=True)
        faiss_path = os.path.join(metadata_dir, "faiss_index.bin")
        faiss.write_index(index, faiss_path)
        print(f"Saved FAISS Index to {faiss_path}")
    except Exception as e:
        print(f"FAISS error: {e}")
    
    # 4. Save Embeddings to Parquet
    print("Saving Embeddings to Parquet format...")
    try:
        df['embedding'] = [emb.tolist() if i in valid_indices else None for i, emb in enumerate(all_embeddings)]
        parquet_path = os.path.join(base_dir, "data", "embeddings_master.parquet")
        df.to_parquet(parquet_path, engine='pyarrow', index=False)
        print(f"Saved Embeddings Parquet to {parquet_path}")
    except Exception as e:
        print(f"Parquet save error: {e}")
    
if __name__ == "__main__":
    main()
