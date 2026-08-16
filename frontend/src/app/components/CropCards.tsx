"use client";

import React from "react";
import { Sprout, ShieldAlert, ChevronRight } from "lucide-react";

interface CropInfo {
  id: string;
  name: string;
  urduName: string;
  category: string;
  description: string;
  diseases: { name: string; urdu: string; severity: "High" | "Medium" }[];
  imageUrl: string;
}

const cropData: CropInfo[] = [
  {
    id: "wheat",
    name: "Wheat (گندم)",
    urduName: "گندم",
    category: "Staple Food Crop",
    description: "Pakistan's primary staple crop, cultivated extensively across Punjab and Sindh during the Rabi season.",
    diseases: [
      { name: "Leaf Rust (Stripe/Yellow)", urdu: "پتوں کی زنگ", severity: "High" },
      { name: "Powdery Mildew", urdu: "سفوفی پھپھوندی", severity: "Medium" },
      { name: "Loose Smut", urdu: "سرمئی کُنگیا", severity: "Medium" }
    ],
    imageUrl: "https://images.unsplash.com/photo-1574323347407-f5e1ad6d020b?auto=format&fit=crop&w=800&q=80"
  },
  {
    id: "cotton",
    name: "Cotton (کپاس)",
    urduName: "کپاس",
    category: "Cash Crop (White Gold)",
    description: "The backbone of Pakistan's textile industry, highly susceptible to viral and sucking pest complexes.",
    diseases: [
      { name: "Cotton Leaf Curl Virus (CLCuV)", urdu: "کپاس کے پتوں کا مڑاؤ وائرس", severity: "High" },
      { name: "Bacterial Blight", urdu: "بیکٹیریل بلائٹ", severity: "High" },
      { name: "Fusarium Wilt", urdu: "مرجھاؤ", severity: "Medium" }
    ],
    imageUrl: "https://cdn.britannica.com/72/270772-159-7C6263D6/Cotton-plants-in-a-field.jpg"
  },
  {
    id: "rice",
    name: "Rice (چاول)",
    urduName: "چاول",
    category: "Major Export Grain",
    description: "Basmati and IRRI varieties grown in monsoon flooded paddies across Kalar tract and lower Sindh.",
    diseases: [
      { name: "Bacterial Leaf Blight (BLB)", urdu: "پتوں کا جھلساؤ", severity: "High" },
      { name: "Rice Blast", urdu: "چاول کا بلاسٹ", severity: "High" },
      { name: "Sheath Blight", urdu: "غلاف کا جھلساؤ", severity: "Medium" }
    ],
    imageUrl: "https://images.unsplash.com/photo-1586201375761-83865001e31c?auto=format&fit=crop&w=800&q=80"
  },
  {
    id: "potato",
    name: "Potato (آلو)",
    urduName: "آلو",
    category: "High-Yield Horticulture",
    description: "Vital tuber crop in Okara, Sahiwal, and northern valleys, vulnerable to sudden fungal epidemics.",
    diseases: [
      { name: "Late Blight (Phytophthora)", urdu: "پچھلا جھلساؤ", severity: "High" },
      { name: "Early Blight (Alternaria)", urdu: "اگلا جھلساؤ", severity: "Medium" },
      { name: "Blackleg & Soft Rot", urdu: "کالی ٹانگ اور نرم گلنا", severity: "High" }
    ],
    imageUrl: "https://images.unsplash.com/photo-1518977676601-b53f82aba655?auto=format&fit=crop&w=800&q=80"
  },
  {
    id: "tomato",
    name: "Tomato (ٹماٹر)",
    urduName: "ٹماٹر",
    category: "Vegetable Cash Crop",
    description: "Cultivated nationwide; highly prone to whitefly-transmitted geminiviruses and fungal leaf spots.",
    diseases: [
      { name: "Tomato Yellow Leaf Curl Virus", urdu: "پیلا مڑاؤ وائرس", severity: "High" },
      { name: "Septoria Leaf Spot", urdu: "سیپٹوریا دھبے", severity: "Medium" },
      { name: "Target Spot", urdu: "ٹارگٹ دھبے", severity: "Medium" }
    ],
    imageUrl: "https://images.unsplash.com/photo-1592924357228-91a4daadcfea?auto=format&fit=crop&w=800&q=80"
  },
  {
    id: "sugarcane",
    name: "Sugarcane (گنا)",
    urduName: "گنا",
    category: "Perennial Industrial Crop",
    description: "Long-duration crop feeding Pakistan's sugar mills, heavily impacted by soil-borne fungal rots.",
    diseases: [
      { name: "Red Rot (Sugarcane Cancer)", urdu: "سرخ سڑاند", severity: "High" },
      { name: "Sugarcane Mosaic Virus", urdu: "موزیک وائرس", severity: "Medium" },
      { name: "Whip Smut", urdu: "کوڑا کُنگیا", severity: "Medium" }
    ],
    imageUrl: "https://plantix.net/en/library/assets/custom/crop-images/sugarcane.jpeg"
  },
  {
    id: "tobacco",
    name: "Tobacco (تمباکو)",
    urduName: "تمباکو",
    category: "Cash Crop (KPK Region)",
    description: "A highly profitable cash crop cultivated primarily in KPK, heavily affected by viral and soil-borne diseases.",
    diseases: [
      { name: "Tobacco Mosaic Virus (TMV)", urdu: "تمباکو کا موزیک وائرس", severity: "High" },
      { name: "Black Shank", urdu: "کالا تنا", severity: "High" },
      { name: "Frog Eye Leaf Spot", urdu: "پتوں کے دھبے", severity: "Medium" }
    ],
    imageUrl: "https://matixgroup.com/wp-content/uploads/2025/03/Tobacco-1.png"
  }
];

export default function CropCards() {
  return (
    <section className="w-full py-20 bg-gray-50 dark:bg-[#07130a] text-gray-900 dark:text-gray-100 border-t border-gray-200 dark:border-gray-800 transition-colors duration-300">
      <div className="max-w-7xl mx-auto px-6">
        
        {/* Header Title */}
        <div className="mb-14">
          <div className="flex items-center gap-2 text-emerald-700 dark:text-emerald-400 text-base font-semibold tracking-wider uppercase mb-2">
            <Sprout className="w-5 h-5 text-emerald-600 dark:text-emerald-400" />
            <span>Pakistani Agricultural Profile</span>
          </div>
          <h2 className="text-3xl md:text-5xl font-extrabold text-gray-900 dark:text-white tracking-tight">
            Major Crops & Frequent Diseases
          </h2>
          <p className="text-gray-600 dark:text-gray-400 text-base md:text-lg mt-2">
            Common pathological threats affecting Pakistani agriculture detected by ZARI.ai.
          </p>
        </div>

        {/* Cards Grid - Clean White & Green Theme */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {cropData.map((crop) => (
            <div
              key={crop.id}
              className="group bg-white dark:bg-[#0c1a11] border border-gray-200 dark:border-gray-800 hover:border-emerald-500 dark:hover:border-emerald-500 rounded-2xl p-7 transition-all duration-300 flex flex-col justify-between shadow-sm hover:shadow-xl relative overflow-hidden"
            >
              <div>
                {/* Card Top: Title Left, Realistic Crop Image Right */}
                <div className="flex items-start justify-between gap-5 mb-5">
                  <div className="flex-1">
                    <h3 className="text-2xl font-bold text-gray-900 dark:text-white group-hover:text-emerald-700 dark:group-hover:text-emerald-400 transition-colors leading-snug">
                      {crop.name}
                    </h3>
                    <span className="inline-block mt-2 text-xs md:text-sm font-semibold text-emerald-800 dark:text-emerald-300 bg-emerald-50 dark:bg-emerald-900/30 border border-emerald-200 dark:border-emerald-800/50 px-3 py-1 rounded-full">
                      {crop.category}
                    </span>
                  </div>
                  
                  {/* Crop Image Thumbnail */}
                  <div className="w-24 h-24 md:w-28 md:h-28 rounded-2xl overflow-hidden border border-gray-200 dark:border-gray-700 flex-shrink-0 bg-gray-100 dark:bg-gray-800 shadow-sm group-hover:border-emerald-400 dark:group-hover:border-emerald-500 transition-colors">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={crop.imageUrl}
                      alt={crop.name}
                      className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500"
                    />
                  </div>
                </div>

                {/* Info Description */}
                <p className="text-sm md:text-base text-gray-600 dark:text-gray-400 leading-relaxed mb-6 border-b border-gray-100 dark:border-gray-800/60 pb-5">
                  {crop.description}
                </p>

                {/* Frequent Diseases List */}
                <div className="space-y-2.5 mb-6">
                  <div className="flex items-center gap-2 text-sm font-bold text-gray-800 dark:text-gray-200 mb-3">
                    <ShieldAlert className="w-4 h-4 text-amber-600 dark:text-amber-500" />
                    <span>Frequent Pathogens in Pakistan:</span>
                  </div>
                  {crop.diseases.map((dis, idx) => (
                    <div
                      key={idx}
                      className="flex items-center justify-between text-xs md:text-sm bg-gray-50 dark:bg-[#112417] px-3.5 py-2.5 rounded-xl border border-gray-200/70 dark:border-gray-800"
                    >
                      <div className="flex items-center gap-2.5">
                        <span className={`w-2 h-2 rounded-full ${dis.severity === "High" ? "bg-red-500" : "bg-amber-500"}`} />
                        <span className="text-gray-900 dark:text-gray-200 font-medium">{dis.name}</span>
                      </div>
                      <span className="text-gray-600 dark:text-gray-400 text-xs md:text-sm font-serif" dir="rtl">
                        {dis.urdu}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Card Footer Action */}
              <div className="pt-4 border-t border-gray-100 dark:border-gray-800/60 flex items-center justify-between text-sm text-emerald-700 dark:text-emerald-500 group-hover:text-emerald-800 dark:group-hover:text-emerald-400 font-bold transition-colors">
                <span>AI Diagnostic Ready</span>
                <ChevronRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
