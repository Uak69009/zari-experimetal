import json
import pandas as pd

with open("ml_pipeline/data/class_map_final.json") as f:
    cmap = json.load(f)

head_classes = cmap["head_classes"]
pretrain_classes = cmap["pretrain_classes"]

def crop_of(class_name: str) -> str:
    return class_name.split("_")[0]

id_to_name_head = {v: k for k, v in head_classes.items()}
id_to_name_pretrain = {v: k for k, v in pretrain_classes.items()}

def resolve_class_name(row):
    if row.get("class_id") in id_to_name_head:
        return id_to_name_head[row["class_id"]]
    if row.get("pretrain_id") in id_to_name_pretrain:
        return id_to_name_pretrain[row["pretrain_id"]]
    return None

def run_analysis(csv_path, title):
    print(f"=== {title} ({csv_path}) ===")
    df = pd.read_csv(csv_path, low_memory=False)
    df["resolved_class_name"] = df.apply(resolve_class_name, axis=1)
    df = df[df["resolved_class_name"].notna()].copy()
    df["crop"] = df["resolved_class_name"].apply(crop_of)

    crop_totals = df["crop"].value_counts()
    crop_class_counts = df.groupby("crop")["resolved_class_name"].nunique()
    field_totals = df[df["split"].isin(["val", "test"])].groupby("crop").size()

    header_crop = "Crop"
    header_total = "Total Images"
    header_classes = "Classes"
    header_field = "Field(val+test)"

    print(f"{header_crop:<15} {header_total:>15} {header_classes:>10} {header_field:>18}")
    print("-" * 60)
    for crop in crop_totals.index:
        tot = crop_totals[crop]
        cls_cnt = crop_class_counts.get(crop, 0)
        fld_cnt = field_totals.get(crop, 0)
        print(f"{crop:<15} {tot:>15,} {cls_cnt:>10} {fld_cnt:>18,}")

    print(f"\nTop 5 by total image volume: {list(crop_totals.head(5).index)}\n")

run_analysis("ml_pipeline/data/dataset_final_training.csv", "BASELINE DATASET")
run_analysis("ml_pipeline/data/dataset_final_training_v2.csv", "EXPANDED DATASET V2")
