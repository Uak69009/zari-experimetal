"use client";

import React from "react";
import { ShoppingBag, Star, ShieldCheck, ArrowRight, Sprout } from "lucide-react";
import dynamic from "next/dynamic";

const MotionDiv = dynamic(() => import("framer-motion").then((mod) => mod.motion.div), { ssr: false }) as any;

const products = [
  {
    id: 1,
    name: "Engro Urea",
    urduName: "اینگرو یوریا",
    category: "Nitrogen Fertilizer",
    description: "Pakistan's most trusted nitrogen fertilizer (46% N). Ensures rapid vegetative growth and lush green crops.",
    price: "₨ 3,590",
    rating: 4.9,
    reviews: 428,
    badge: "Best Seller",
    imageUrl: "/images/engro_urea.png"
  },
  {
    id: 2,
    name: "Engro DAP",
    urduName: "اینگرو ڈی اے پی",
    category: "Phosphate Fertilizer",
    description: "Premium Di-Ammonium Phosphate (18:46 N-P ratio) for strong root development and early plant vigor.",
    price: "₨ 12,850",
    rating: 4.8,
    reviews: 312,
    badge: "Top Rated",
    imageUrl: "/images/engro_dap.png"
  },
  {
    id: 3,
    name: "Engro Zarkhez Plus",
    urduName: "اینگرو زرخیز پلس",
    category: "Compound NPK Fertilizer",
    description: "A specialized blend (NPK 8:23:18) with organic fillers and bio-stimulants for superior yield quality.",
    price: "₨ 7,150",
    rating: 4.8,
    reviews: 185,
    badge: "Essential",
    imageUrl: "/images/engro_zarkhez.png"
  },
  {
    id: 4,
    name: "WeedMaster Herbicide",
    urduName: "ویڈ ماسٹر جڑی بوٹی مار",
    category: "Selective Herbicide",
    description: "Effectively controls broadleaf weeds without damaging your primary crop.",
    price: "₨ 1,200",
    rating: 4.6,
    reviews: 95,
    badge: "New Arrival",
    imageUrl: "https://grassplusinc.com/wp-content/uploads/2023/07/gpi-man-spraying-herbicide.webp"
  },
  {
    id: 5,
    name: "BioBoost Organic Compost",
    urduName: "بائیو بوسٹ نامیاتی کھاد",
    category: "Organic Soil Amender",
    description: "100% natural compost enriched with beneficial microbes to restore soil health.",
    price: "₨ 800",
    rating: 4.9,
    reviews: 310,
    badge: "Eco-Friendly",
    imageUrl: "https://images.unsplash.com/photo-1518977676601-b53f82aba655?auto=format&fit=crop&w=600&q=80"
  },
  {
    id: 6,
    name: "ZARI Root Guard",
    urduName: "زاری روٹ گارڈ",
    category: "Seed Treatment",
    description: "Protects germinating seeds from soil-borne pathogens and root rot diseases.",
    price: "₨ 1,500",
    rating: 4.8,
    reviews: 142,
    badge: "Prevention",
    imageUrl: "https://5.imimg.com/data5/SELLER/Default/2022/5/BU/NC/JM/152087513/1kg-root-power-pouch.jpg"
  }
];

export default function AgriStoreAds() {
  return (
    <section className="w-full py-24 bg-gray-50 dark:bg-[#07130a] text-gray-900 dark:text-gray-100 border-t border-gray-200 dark:border-gray-800 transition-colors duration-300 relative overflow-hidden">
      
      {/* Background Accent */}
      <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-emerald-100/40 dark:bg-emerald-900/10 rounded-full blur-[150px] pointer-events-none" />

      <div className="max-w-7xl mx-auto px-6 relative z-10">
        
        {/* Header Section */}
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-6 mb-12">
          <div>
            <div className="flex items-center gap-2 text-emerald-700 dark:text-emerald-400 text-base font-semibold tracking-wider uppercase mb-2">
              <ShoppingBag className="w-5 h-5 text-emerald-600 dark:text-emerald-400" />
              <span>ZARI Agri-Store Partners</span>
            </div>
            <h2 className="text-3xl md:text-4xl font-extrabold text-gray-900 dark:text-white tracking-tight">
              Recommended Treatments
            </h2>
            <p className="text-gray-600 dark:text-gray-400 mt-3 max-w-2xl text-lg">
              Authentic, high-grade fertilizers and pesticides verified by ZARI agronomists to treat detected diseases.
            </p>
          </div>
          <button className="bg-emerald-600 hover:bg-emerald-700 dark:bg-zari-accent dark:hover:bg-emerald-400 dark:text-zari-bg text-white px-6 py-3 rounded-xl font-bold flex items-center gap-2 transition-all shadow-md hover:shadow-lg">
            Visit Partner Store
            <ArrowRight className="w-5 h-5" />
          </button>
        </div>

        {/* Products Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {products.map((product, idx) => (
            <MotionDiv 
              key={product.id}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: idx * 0.15 }}
              className="group bg-white dark:bg-[#112417] border border-gray-200 dark:border-gray-800 rounded-3xl overflow-hidden hover:shadow-2xl hover:border-emerald-400 dark:hover:border-emerald-600 transition-all flex flex-col relative"
            >
              {/* Product Badge */}
              <div className="absolute top-4 left-4 z-20">
                <span className="bg-amber-500 text-gray-900 text-xs font-black uppercase tracking-wider px-3 py-1.5 rounded-full shadow-md flex items-center gap-1.5">
                  <Star className="w-3.5 h-3.5 fill-gray-900" />
                  {product.badge}
                </span>
              </div>

              {/* Verified Shield */}
              <div className="absolute top-4 right-4 z-20 bg-white/90 dark:bg-gray-900/90 backdrop-blur-sm p-2 rounded-full shadow-sm text-emerald-600 dark:text-emerald-400">
                <ShieldCheck className="w-5 h-5" />
              </div>

              {/* Product Image area */}
              <div className="h-56 w-full bg-gray-100 dark:bg-gray-800 overflow-hidden relative">
                <div className="absolute inset-0 bg-gradient-to-t from-black/40 to-transparent z-10 opacity-0 group-hover:opacity-100 transition-opacity" />
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img 
                  src={product.imageUrl} 
                  alt={product.name}
                  className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-700"
                />
              </div>

              {/* Product Info */}
              <div className="p-6 flex flex-col flex-1">
                <div className="flex items-center justify-between mb-3">
                  <span className="text-xs font-bold text-emerald-700 dark:text-emerald-400 bg-emerald-50 dark:bg-emerald-900/30 border border-emerald-200 dark:border-emerald-800/50 px-2.5 py-1 rounded-md">
                    {product.category}
                  </span>
                  <div className="flex items-center gap-1 text-sm text-gray-500 dark:text-gray-400 font-medium">
                    <Star className="w-4 h-4 text-amber-500 fill-amber-500" />
                    <span>{product.rating}</span>
                    <span>({product.reviews})</span>
                  </div>
                </div>
                
                <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-1 group-hover:text-emerald-700 dark:group-hover:text-emerald-400 transition-colors">
                  {product.name}
                </h3>
                <h4 className="text-lg font-serif text-emerald-800 dark:text-emerald-300 mb-4" dir="rtl">
                  {product.urduName}
                </h4>
                
                <p className="text-gray-600 dark:text-gray-400 text-sm leading-relaxed mb-6 flex-1">
                  {product.description}
                </p>

                <div className="flex items-center justify-between pt-4 border-t border-gray-100 dark:border-gray-800">
                  <span className="text-2xl font-black text-gray-900 dark:text-white">
                    {product.price}
                  </span>
                  <button className="text-emerald-700 dark:text-emerald-400 hover:text-white dark:hover:text-zari-bg bg-emerald-50 dark:bg-emerald-900/20 hover:bg-emerald-600 dark:hover:bg-zari-accent p-3 rounded-xl transition-all">
                    <ShoppingBag className="w-5 h-5" />
                  </button>
                </div>
              </div>
            </MotionDiv>
          ))}
        </div>

      </div>
    </section>
  );
}
