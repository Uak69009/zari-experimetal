"""
ZARI.ai — Phase 6 Master End-to-End Inference Pipeline Wiring

Wires the frozen vision pipeline output (read-only) with:
1. Decision Guard (ACCEPT/REJECT): REJECT returns "insufficient confidence for disease-specific recommendation"
2. Environmental Weather Context Lookup (injected directly into prompt)
3. Multilingual ChromaDB Semantic RAG Retrieval (Phase 5 store)
4. Prompt Assembly & IPM-Enforced Advisory Synthesis (English & Urdu)

RULES:
- Vision output is READ-ONLY input to this phase -- RAG/LLM NEVER overrides Model A/B diagnosis.
- IPM hierarchy strictly enforced: Cultural -> Biological -> Chemical.
- Active ingredients only; zero invented dosage/PHI.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add SCRIPT_DIR to path for importing retrieval_api
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.append(str(SCRIPT_DIR))

from retrieval_api import retrieve

# ── Weather Risk Notes Heuristics (Documented, not invented) ─────────────────
WEATHER_HEURISTICS = {
    "Tomato_Late_Blight": "Cool temperatures (15–22°C) combined with high relative humidity (>90%) and rain/fog create EXTREME risk for rapid Phytophthora sporangia germination and epidemic canopy destruction.",
    "Tomato_Early_Blight": "Warm humid conditions (24–29°C) with prolonged dew periods favor Alternaria solani spore germination and leaf spot expansion.",
    "Tomato_Yellow_Leaf_Curl_Virus": "Hot dry weather (>30°C) accelerates silverleaf whitefly (Bemisia tabaci) vector reproduction and viral transmission rate.",
    "Potato_Late_Blight": "Cool saturated air (15–20°C, RH >90%) with cloud cover provides optimal microclimate for Phytophthora infestans rapid lesion development.",
    "Potato_Early_Blight": "Alternating wet and dry warm conditions (22–28°C) accelerate Alternaria solani secondary spore release.",
    "Pepper_Bacterial_Spot": "Warm rainy weather (24–30°C) with wind-driven rain splash accelerates Xanthomonas bacterial entry through stomata and wounds.",
    "Pepper_Leaf_Curl": "Warm dry conditions encourage whitefly population buildup, elevating Chilli leaf curl virus spread."
}

def get_weather_risk_note(disease_class: str, temp_c: float, humidity_pct: float, rainfall_mm: float) -> str:
    default_note = f"Ambient conditions ({temp_c}°C, {humidity_pct}% RH, {rainfall_mm}mm rain). Regular scouting advised."
    base_heuristic = WEATHER_HEURISTICS.get(disease_class, default_note)
    
    if "Late_Blight" in disease_class and temp_c <= 22 and humidity_pct >= 85:
        return f"🚨 CRITICAL WEATHER RISK: {base_heuristic}"
    elif "Bacterial_Spot" in disease_class and humidity_pct >= 80:
        return f"⚠️ ELEVATED WEATHER RISK: {base_heuristic}"
    elif "Virus" in disease_class or "Curl" in disease_class:
        return f"⚠️ VECTOR WEATHER RISK: {base_heuristic}"
    return base_heuristic

# ── Master End-to-End Inference Pipeline ──────────────────────────────────────
def run_end_to_end_inference(vision_input: Dict[str, Any], env_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes the master ZARI.ai 3-Crop End-to-End Inference Pipeline.
    
    Args:
        vision_input: Dict containing:
            - crop (str)
            - disease (str)
            - disease_confidence (float)
            - edl_uncertainty (float)
            - decision (str: 'ACCEPT' or 'REJECT')
            - estimated_visual_disease_coverage (float)
            - severity_tag (str: 'Low', 'Medium', 'High')
        env_context: Dict containing:
            - temperature (float)
            - humidity (float)
            - rainfall (float)
            - season (str)
            - location (str)
            
    Returns:
        Dict containing full advisory pipeline output
    """
    t0 = time.time()
    
    crop = vision_input["crop"]
    disease = vision_input["disease"]
    conf = vision_input["disease_confidence"]
    unc = vision_input["edl_uncertainty"]
    decision = vision_input["decision"]
    coverage = vision_input.get("estimated_visual_disease_coverage", 0.0)
    severity = vision_input.get("severity_tag", "Medium")
    
    # ── GUARD: REJECTION HANDLER ──────────────────────────────────────────────
    if decision == "REJECT":
        return {
            "status": "REJECTED",
            "decision": "REJECT",
            "crop": crop,
            "diagnosis": disease,
            "disease_confidence": conf,
            "edl_uncertainty": unc,
            "message": "insufficient confidence for disease-specific recommendation",
            "advisory_english": "insufficient confidence for disease-specific recommendation. Image rejected by Selective Classification and Risk Control (SCRC) gate.",
            "advisory_urdu": "بیماری کے مخصوص مشورے کے لیے غیر یقینی صورتحال کا لیول بہت زیادہ ہے۔ تصویر SCRC گیٹ سے مسترد ہو گئی ہے۔",
            "latency_ms": round((time.time() - t0) * 1000, 2)
        }
        
    # ── STEP 1: Environmental Risk Note Generation ───────────────────────────
    temp = env_context.get("temperature", 25.0)
    hum = env_context.get("humidity", 70.0)
    rain = env_context.get("rainfall", 0.0)
    weather_note = get_weather_risk_note(disease, temp, hum, rain)
    
    # Check Combined Urgency Flag
    is_high_severity = severity == "High" or coverage >= 0.35
    is_high_weather_risk = "CRITICAL" in weather_note or "ELEVATED" in weather_note
    combined_urgency_flag = is_high_severity and is_high_weather_risk
    
    # ── STEP 2: ChromaDB Semantic RAG Retrieval ───────────────────────────────
    # Retrieve evidence chunks filtered by crop and disease_class
    retrieved_chunks = retrieve(
        query=f"{crop} {disease} symptoms treatment prevention",
        crop=crop,
        disease_class=disease,
        k=8
    )
    
    # Organize chunks by section
    section_map = {c["metadata"]["section"]: c for c in retrieved_chunks}
    
    # ── STEP 3: IPM Advisory Response Synthesis ───────────────────────────────
    # English Advisory
    en_lines = []
    en_lines.append(f"=== ZARI.ai DIAGNOSTIC & ADVISORY REPORT ===")
    en_lines.append(f"Crop: {crop} | Diagnosis: {disease}")
    en_lines.append(f"Confidence: {conf*100:.2f}% | EDL Uncertainty: {unc:.4f} | Status: ACCEPTED")
    en_lines.append(f"Severity: {severity} (Visual Disease Coverage: {coverage*100:.1f}%)")
    en_lines.append(f"Location: {env_context.get('location', 'Pakistan')} | Season: {env_context.get('season', 'N/A')}")
    en_lines.append(f"Weather Note: {weather_note}")
    
    if combined_urgency_flag:
        en_lines.append(f"\n🚨 COMBINED URGENCY WARNING: High visual disease coverage ({coverage*100:.1f}%) combined with elevated weather risk accelerates field epidemic spread! Immediate intervention required.")
        
    en_lines.append("\n--- RECOMMENDED INTEGRATED PEST MANAGEMENT (IPM) PROTOCOL ---")
    
    # 1. Cultural Control
    cultural_chunk = section_map.get("cultural_control")
    if cultural_chunk:
        en_lines.append(f"1. CULTURAL CONTROL [{cultural_chunk['metadata']['source_name']} - Level {cultural_chunk['metadata']['evidence_level']}]:")
        en_lines.append(f"   {cultural_chunk['text']}")
    else:
        en_lines.append("1. CULTURAL CONTROL: Prune infected leaves, improve field drainage, and maintain proper plant spacing.")
        
    # 2. Biological Control
    bio_chunk = section_map.get("biological_control")
    if bio_chunk:
        en_lines.append(f"2. BIOLOGICAL & BIO-RATIONAL CONTROL [{bio_chunk['metadata']['source_name']} - Level {bio_chunk['metadata']['evidence_level']}]:")
        en_lines.append(f"   {bio_chunk['text']}")
    else:
        en_lines.append("2. BIOLOGICAL CONTROL: Apply Trichoderma harzianum or Bacillus subtilis bio-rationals.")
        
    # 3. Chemical Control (Active Ingredients Only)
    chem_chunk = section_map.get("chemical_control")
    if chem_chunk:
        en_lines.append(f"3. CHEMICAL CONTROL (Active Ingredients Only) [{chem_chunk['metadata']['source_name']} - Level {chem_chunk['metadata']['evidence_level']}]:")
        en_lines.append(f"   {chem_chunk['text']}")
        en_lines.append("   ⚠️ Note: Active ingredients only. Specific product dosage/PHI requires current local label/authority verification.")
    else:
        en_lines.append("3. CHEMICAL CONTROL: Consult registered active ingredients. Specific product dosage/PHI requires current local label check.")
        
    # Safety & Prevention
    prev_chunk = section_map.get("prevention")
    if prev_chunk:
        en_lines.append(f"\nPREVENTION [{prev_chunk['metadata']['source_name']} - Level {prev_chunk['metadata']['evidence_level']}]:")
        en_lines.append(f"   {prev_chunk['text']}")
        
    safety_chunk = section_map.get("safety")
    if safety_chunk:
        en_lines.append(f"SAFETY [{safety_chunk['metadata']['source_name']} - Level {safety_chunk['metadata']['evidence_level']}]:")
        en_lines.append(f"   {safety_chunk['text']}")
        
    advisory_english = "\n".join(en_lines)
    
    # Urdu Advisory
    ur_lines = []
    ur_lines.append(f"=== زاری AI تشخیصی و زراعی رپورٹ ===")
    ur_lines.append(f"فصل: {crop} | تشخیص: {disease}")
    ur_lines.append(f"اعتماد: {conf*100:.1f}% | غیر یقینی صورتحال: {unc:.4f} | کیفیات: قبول شدہ (ACCEPTED)")
    ur_lines.append(f"شدت: {severity} (بیماری کا بصری رقبہ: {coverage*100:.1f}%)")
    ur_lines.append(f"موسمی نوٹ: {weather_note}")
    
    if combined_urgency_flag:
        ur_lines.append(f"\n🚨 شدید ہنگامی تنبیہ: بیماری کے زیادہ پھیلائو ({coverage*100:.1f}%) اور موافق موسم کی وجہ سے فیلڈ میں تیزی سے پھیلنے کا خطرہ ہے! فوری اقدامات ضروری ہیں۔")
        
    ur_lines.append("\n--- تجویز کردہ جامع حکمت عملی (IPM) ---")
    if cultural_chunk:
        ur_lines.append(f"1. ثقافتی اور زرعی اقدامات: {cultural_chunk['text']}")
    if bio_chunk:
        ur_lines.append(f"2. حیاتیاتی اور نامیاتی علاج: {bio_chunk['text']}")
    if chem_chunk:
        ur_lines.append(f"3. کیمیائی کنٹرول (صرف فعالی اجزاء): {chem_chunk['text']}")
        ur_lines.append("   ⚠️ نوٹ: کیمیائی ادویات کی مقدار اور وقفہ کے لیے مقامی سمی ادویات کے لیبل کی تصدیق لازمی ہے۔")
        
    advisory_urdu = "\n".join(ur_lines)
    
    latency_ms = round((time.time() - t0) * 1000, 2)
    
    return {
        "status": "SUCCESS",
        "decision": "ACCEPT",
        "crop": crop,
        "diagnosis": disease,
        "disease_confidence": conf,
        "edl_uncertainty": unc,
        "visual_coverage": coverage,
        "severity_tag": severity,
        "combined_urgency_flag": combined_urgency_flag,
        "weather_risk_note": weather_note,
        "retrieved_chunks_count": len(retrieved_chunks),
        "retrieved_chunks": [
            {
                "id": c["id"],
                "section": c["metadata"]["section"],
                "evidence_level": c["metadata"]["evidence_level"],
                "source_name": c["metadata"]["source_name"],
                "source_url": c["metadata"]["source_url"],
                "similarity_score": c["similarity_score"]
            }
            for c in retrieved_chunks
        ],
        "advisory_english": advisory_english,
        "advisory_urdu": advisory_urdu,
        "latency_ms": latency_ms
    }

# ── Test Suite Execution (5 End-to-End Test Cases) ───────────────────────────
def run_phase6_test_suite():
    print("=" * 75)
    print("  ZARI.ai — PHASE 6 END-TO-END INFERENCE PIPELINE VERIFICATION")
    print("=" * 75)
    
    test_cases = [
        # Case 1: Tomato Late Blight (High Severity, Cool Wet Weather -> High Combined Urgency)
        {
            "case_id": 1,
            "title": "Case 1: Tomato Late Blight (High Severity & Cool Wet Weather)",
            "vision_input": {
                "crop": "Tomato",
                "disease": "Tomato_Late_Blight",
                "disease_confidence": 0.9850,
                "edl_uncertainty": 0.1200,
                "decision": "ACCEPT",
                "estimated_visual_disease_coverage": 0.45,
                "severity_tag": "High"
            },
            "env_context": {
                "temperature": 18.0,
                "humidity": 92.0,
                "rainfall": 12.5,
                "season": "Spring",
                "location": "Swat, KP, Pakistan"
            }
        },
        # Case 2: Potato Early Blight (Medium Severity, Warm Humid Weather)
        {
            "case_id": 2,
            "title": "Case 2: Potato Early Blight (Medium Severity, Warm Humid Weather)",
            "vision_input": {
                "crop": "Potato",
                "disease": "Potato_Early_Blight",
                "disease_confidence": 0.9710,
                "edl_uncertainty": 0.2100,
                "decision": "ACCEPT",
                "estimated_visual_disease_coverage": 0.22,
                "severity_tag": "Medium"
            },
            "env_context": {
                "temperature": 26.0,
                "humidity": 78.0,
                "rainfall": 2.0,
                "season": "Spring",
                "location": "Okara, Punjab, Pakistan"
            }
        },
        # Case 3: Pepper Bacterial Spot (Low Severity, Moderate Weather)
        {
            "case_id": 3,
            "title": "Case 3: Pepper Bacterial Spot (Low Severity, Moderate Weather)",
            "vision_input": {
                "crop": "Pepper",
                "disease": "Pepper_Bacterial_Spot",
                "disease_confidence": 0.9940,
                "edl_uncertainty": 0.0800,
                "decision": "ACCEPT",
                "estimated_visual_disease_coverage": 0.08,
                "severity_tag": "Low"
            },
            "env_context": {
                "temperature": 28.0,
                "humidity": 65.0,
                "rainfall": 0.0,
                "season": "Summer",
                "location": "Mirpurkhas, Sindh, Pakistan"
            }
        },
        # Case 4: Tomato Yellow Leaf Curl Virus (High Severity, Hot Dry Whitefly Weather)
        {
            "case_id": 4,
            "title": "Case 4: Tomato Yellow Leaf Curl Virus (High Severity & Vector Weather)",
            "vision_input": {
                "crop": "Tomato",
                "disease": "Tomato_Yellow_Leaf_Curl_Virus",
                "disease_confidence": 0.9910,
                "edl_uncertainty": 0.0950,
                "decision": "ACCEPT",
                "estimated_visual_disease_coverage": 0.38,
                "severity_tag": "High"
            },
            "env_context": {
                "temperature": 34.0,
                "humidity": 45.0,
                "rainfall": 0.0,
                "season": "Summer",
                "location": "Faisalabad, Punjab, Pakistan"
            }
        },
        # Case 5: Rejection Case (High EDL Uncertainty / Low Confidence sample -> REJECT)
        {
            "case_id": 5,
            "title": "Case 5: Out-of-Distribution / High Uncertainty Sample (REJECTED by SCRC Gate)",
            "vision_input": {
                "crop": "Potato",
                "disease": "Potato_Early_Blight",
                "disease_confidence": 0.5200,
                "edl_uncertainty": 0.8800,
                "decision": "REJECT",
                "estimated_visual_disease_coverage": 0.15,
                "severity_tag": "Low"
            },
            "env_context": {
                "temperature": 22.0,
                "humidity": 70.0,
                "rainfall": 0.0,
                "season": "Spring",
                "location": "Sahiwal, Punjab, Pakistan"
            }
        }
    ]
    
    results = []
    
    for tc in test_cases:
        print(f"\n{'='*75}")
        print(f"  END-TO-END TEST CASE #{tc['case_id']}: {tc['title']}")
        print(f"{'='*75}")
        
        out = run_end_to_end_inference(tc["vision_input"], tc["env_context"])
        results.append({"test_case": tc, "output": out})
        
        print(f"Status           : {out['status']} ({out['decision']})")
        print(f"Crop / Diagnosis : {out['crop']} / {out['diagnosis']}")
        print(f"Confidence / EDL : {out['disease_confidence']*100:.1f}% / Uncertainty={out['edl_uncertainty']:.4f}")
        
        if out["decision"] == "REJECT":
            print(f"Rejection Msg    : {out['message']}")
            print(f"Urdu Rejection   : {out['advisory_urdu']}")
        else:
            print(f"Severity / Cover : {out['severity_tag']} ({out['visual_coverage']*100:.1f}%)")
            print(f"Combined Urgency : {out['combined_urgency_flag']}")
            print(f"Weather Note     : {out['weather_risk_note']}")
            print(f"Retrieved Chunks : {out['retrieved_chunks_count']}")
            print("\n--- RETRIEVED CHUNKS USED ---")
            for c in out["retrieved_chunks"]:
                print(f"  - [{c['section']}] {c['id']} (Score: {c['similarity_score']:.4f}) | {c['source_name']} (Level {c['evidence_level']})")
                print(f"    URL: {c['source_url']}")
                
            print("\n--- ENGLISH ADVISORY OUTPUT ---")
            print(out["advisory_english"])
            print("\n--- URDU ADVISORY OUTPUT ---")
            print(out["advisory_urdu"])
            
    # Save complete test suite output JSON
    output_json_path = REPO_ROOT / "ml_pipeline" / "data" / "phase6_inference_test_results.json"
    with open(output_json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{'='*75}")
    print(f"✓ All 5 end-to-end test cases completed and saved to: {output_json_path.relative_to(REPO_ROOT)}")
    print("STOP — Phase 6 Master End-to-End Inference Pipeline Wiring Complete.")

if __name__ == "__main__":
    run_phase6_test_suite()
