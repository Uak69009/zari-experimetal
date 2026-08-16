import os
import pandas as pd

def append_nwrd():
    """
    Appends the NWRD dataset to the existing dataset_master.csv.
    Maps predefined splits and normalizes labels with a 'Wheat_' prefix.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "data", "dataset_master.csv")
    nwrd_dir = os.path.join(base_dir, "data", "raw", "nwrd", "data")
    
    print(f"Loading existing dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    initial_len = len(df)
    
    # Drop nwrd if it exists to allow re-running cleanly
    if 'nwrd' in df['dataset_source'].unique():
        print("Cleaning up previously appended NWRD records...")
        df = df[df['dataset_source'] != 'nwrd']
        
    if not os.path.isdir(nwrd_dir):
        print(f"ERROR: NWRD data directory not found at {nwrd_dir}")
        return
        
    records = []
    splits = ['train', 'valid', 'test']
    
    print("Scanning NWRD directory...")
    for split in splits:
        split_dir = os.path.join(nwrd_dir, split)
        if not os.path.isdir(split_dir):
            print(f"Warning: Split directory {split_dir} not found.")
            continue
            
        # Map NWRD's 'valid' to our standard 'val'
        mapped_split = 'val' if split == 'valid' else split
        
        for raw_label in os.listdir(split_dir):
            label_dir = os.path.join(split_dir, raw_label)
            if not os.path.isdir(label_dir):
                continue
                
            # Harmonize label: e.g., "black_rust_test" -> "Black_Rust"
            # Remove _test or _valid suffixes
            clean_label = raw_label.lower().replace("_test", "").replace(" test", "")
            clean_label = clean_label.replace("_valid", "").replace(" valid", "")
            # Title case and underscores
            clean_label = clean_label.replace(" ", "_").title()
            
            unified_label = f"Wheat_{clean_label}"
            
            # Special case for healthy to ensure consistency
            if clean_label == "Healthy":
                unified_label = "Wheat_Healthy"
                
            for file in os.listdir(label_dir):
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    image_path = os.path.join(label_dir, file)
                    records.append({
                        "image_path": image_path,
                        "dataset_source": "nwrd",
                        "raw_label": raw_label,
                        "unified_label": unified_label,
                        "split": mapped_split
                    })
                    
    if not records:
        print("No images found in the NWRD directory.")
        return
        
    print(f"Found {len(records)} NWRD images.")
    new_df = pd.DataFrame(records)
    
    # Show a sample of what we are adding
    print("\n--- NWRD Class Distribution Sample ---")
    print(new_df['unified_label'].value_counts())
    
    print("\nAppending to master ledger...")
    combined_df = pd.concat([df, new_df], ignore_index=True)
    print(f"Original length: {initial_len}")
    print(f"New total length: {len(combined_df)}")
    
    # Save back to disk
    combined_df.to_csv(csv_path, index=False)
    print(f"Successfully updated {csv_path}!")

if __name__ == "__main__":
    append_nwrd()
