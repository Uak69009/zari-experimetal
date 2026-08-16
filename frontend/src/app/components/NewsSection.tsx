"use client";

import React, { useRef, useEffect } from "react";
import { Newspaper, ArrowRight, Clock, ChevronLeft, ChevronRight } from "lucide-react";
import dynamic from "next/dynamic";

const MotionDiv = dynamic(() => import("framer-motion").then((mod) => mod.motion.div), { ssr: false }) as any;

const newsData = [
  {
    id: 1,
    category: "Government Policy",
    title: "Punjab Government Announces Subsidy on Certified Wheat Seeds",
    urduTitle: "پنجاب حکومت کا گندم کے بیج پر سبسڈی کا اعلان",
    date: "Oct 15, 2026",
    readTime: "3 min read",
    imageUrl: "https://images.unsplash.com/photo-1574323347407-f5e1ad6d020b?auto=format&fit=crop&w=600&q=80",
    excerpt: "In a bid to boost the upcoming Rabi season yields, the agriculture department has rolled out a major subsidy for registered farmers..."
  },
  {
    id: 2,
    category: "Pathology Alert",
    title: "New Strain of Cotton Leaf Curl Virus (CLCuV) Detected in Sindh",
    urduTitle: "سندھ میں کپاس کے وائرس کی نئی قسم دریافت",
    date: "Oct 12, 2026",
    readTime: "4 min read",
    imageUrl: "https://images.unsplash.com/photo-1592982537447-6f2a6a0c5c1b?auto=format&fit=crop&w=600&q=80",
    excerpt: "Agricultural researchers have warned farmers in lower Sindh to take immediate precautionary measures against a highly resistant whitefly vector..."
  },
  {
    id: 3,
    category: "Tech Innovation",
    title: "ZARI.ai Expands Diagnostic Coverage to 153 Local Crop Varieties",
    urduTitle: "زاری اے آئی کی تشخیصی صلاحیت میں 153 فصلوں کا اضافہ",
    date: "Oct 10, 2026",
    readTime: "2 min read",
    imageUrl: "https://images.unsplash.com/photo-1586201375761-83865001e31c?auto=format&fit=crop&w=600&q=80",
    excerpt: "The indigenous AI platform has successfully integrated data from the NWRD dataset, enabling highly accurate offline detection for localized crop threats..."
  },
  {
    id: 4,
    category: "Weather Advisory",
    title: "Unexpected Monsoon Showers Threaten Tomato Crops Nationwide",
    urduTitle: "غیر متوقع مون سون بارشوں سے ٹماٹر کی فصل کو خطرہ",
    date: "Oct 08, 2026",
    readTime: "3 min read",
    imageUrl: "https://images.unsplash.com/photo-1515150144380-bca9f1650ed9?auto=format&fit=crop&w=600&q=80",
    excerpt: "Farmers are advised to improve drainage and apply preventative fungicides as unseasonal rains create ideal conditions for blights..."
  },
  {
    id: 5,
    category: "Market Trends",
    title: "Sugarcane Harvesting Commences Across Central Punjab",
    urduTitle: "وسطی پنجاب میں گنے کی کٹائی کا آغاز",
    date: "Oct 05, 2026",
    readTime: "5 min read",
    imageUrl: "https://images.unsplash.com/photo-1625244724120-1fd1d34d00f6?auto=format&fit=crop&w=600&q=80",
    excerpt: "As the crushing season approaches, sugar mills are negotiating rates with growers amid expectations of a bumper crop this year..."
  },
  {
    id: 6,
    category: "Outbreak Warning",
    title: "Severe Potato Blight Outbreak Reported in Okara District",
    urduTitle: "اوکاڑہ میں آلو کے جھلساؤ کی شدید بیماری کی رپورٹ",
    date: "Oct 02, 2026",
    readTime: "4 min read",
    imageUrl: "https://images.unsplash.com/photo-1518977676601-b53f82aba655?auto=format&fit=crop&w=600&q=80",
    excerpt: "ZARI.ai surveillance has detected a massive spike in Late Blight scans coming from Okara. Emergency advisories have been dispatched..."
  },
  {
    id: 7,
    category: "Policy & AI",
    title: "Government Launches Free Smart Farming Kits for Rural Areas",
    urduTitle: "حکومت کی جانب سے دیہی علاقوں کے لیے مفت سمارٹ فارمنگ کٹس",
    date: "Sep 28, 2026",
    readTime: "3 min read",
    imageUrl: "https://images.unsplash.com/photo-1620608573216-3e4b3e8e192c?auto=format&fit=crop&w=600&q=80",
    excerpt: "In collaboration with local tech firms, the Ministry of Agriculture is distributing smartphone kits pre-loaded with diagnostic tools..."
  },
  {
    id: 8,
    category: "Research",
    title: "New High-Yield, Drought-Resistant Basmati Rice Introduced",
    urduTitle: "نئی زیادہ پیداوار والی اور خشک سالی کے خلاف مزاحم باسمتی چاول متعارف",
    date: "Sep 25, 2026",
    readTime: "4 min read",
    imageUrl: "https://images.unsplash.com/photo-1530533527749-3663a8e9e1c3?auto=format&fit=crop&w=600&q=80",
    excerpt: "A breakthrough at the National Agricultural Research Centre promises a rice variety that requires 30% less water while resisting common pests..."
  }
];

export default function NewsSection() {
  const scrollRef = useRef<HTMLDivElement>(null);

  const scroll = (direction: 'left' | 'right') => {
    if (scrollRef.current) {
      const { clientWidth } = scrollRef.current;
      // Scroll by roughly the width of one or two cards (e.g. 400px)
      const scrollAmount = direction === 'left' ? -400 : 400;
      scrollRef.current.scrollBy({ left: scrollAmount, behavior: 'smooth' });
    }
  };

  useEffect(() => {
    const interval = setInterval(() => {
      if (scrollRef.current) {
         const { scrollLeft, scrollWidth, clientWidth } = scrollRef.current;
         // If we've scrolled all the way to the right, snap back to start
         if (Math.ceil(scrollLeft + clientWidth) >= scrollWidth - 10) {
           scrollRef.current.scrollTo({ left: 0, behavior: 'smooth' });
         } else {
           scroll('right');
         }
      }
    }, 4000); // Auto scroll every 4 seconds

    return () => clearInterval(interval);
  }, []);

  return (
    <section className="w-full py-24 bg-white dark:bg-zari-bg text-gray-900 dark:text-gray-100 border-t border-gray-200 dark:border-gray-800 transition-colors duration-300 overflow-hidden relative">
      <div className="max-w-[90rem] mx-auto px-6">
        
        {/* Header Section */}
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-6 mb-12">
          <div>
            <div className="flex items-center gap-2 text-emerald-700 dark:text-emerald-400 text-base font-semibold tracking-wider uppercase mb-2">
              <Newspaper className="w-5 h-5 text-emerald-600 dark:text-emerald-400" />
              <span>Agri-News & Updates</span>
            </div>
            <h2 className="text-3xl md:text-4xl font-extrabold text-gray-900 dark:text-white tracking-tight">
              Latest from the Fields
            </h2>
          </div>
        </div>

        {/* Scrolling News Container */}
        <div className="relative group/news">
          
          {/* Left Button */}
          <button 
            onClick={() => scroll('left')}
            className="absolute left-2 md:left-6 top-1/2 -translate-y-1/2 z-30 p-3 bg-white/80 dark:bg-[#112417]/80 backdrop-blur-md hover:bg-white dark:hover:bg-[#112417] text-gray-900 dark:text-white rounded-full border border-gray-200 dark:border-gray-700 transition-all shadow-xl hover:scale-110 opacity-0 group-hover/news:opacity-100 focus:opacity-100 focus:outline-none"
            aria-label="Scroll Left"
          >
            <ChevronLeft className="w-6 h-6" />
          </button>
          
          {/* Right Button */}
          <button 
            onClick={() => scroll('right')}
            className="absolute right-2 md:right-6 top-1/2 -translate-y-1/2 z-30 p-3 bg-white/80 dark:bg-[#112417]/80 backdrop-blur-md hover:bg-white dark:hover:bg-[#112417] text-gray-900 dark:text-white rounded-full border border-gray-200 dark:border-gray-700 transition-all shadow-xl hover:scale-110 opacity-0 group-hover/news:opacity-100 focus:opacity-100 focus:outline-none"
            aria-label="Scroll Right"
          >
            <ChevronRight className="w-6 h-6" />
          </button>

          <div 
            ref={scrollRef}
            className="flex overflow-x-auto gap-6 pb-8 snap-x snap-mandatory scrollbar-hide px-2 md:px-0"
            style={{ scrollbarWidth: 'none', msOverflowStyle: 'none' }}
          >
            {newsData.map((news, idx) => (
            <MotionDiv 
              key={news.id}
              initial={{ opacity: 0, scale: 0.95 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{ duration: 0.5, delay: idx * 0.05 }}
              className="group flex flex-col bg-gray-50 dark:bg-[#112417] border border-gray-200 dark:border-gray-800 rounded-3xl overflow-hidden hover:shadow-xl hover:border-emerald-300 dark:hover:border-emerald-700 transition-all cursor-pointer flex-shrink-0 snap-start w-[320px] md:w-[380px]"
            >
              {/* Image Header */}
              <div className="relative h-48 w-full overflow-hidden">
                <div className="absolute inset-0 bg-black/20 group-hover:bg-transparent transition-colors z-10" />
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img 
                  src={news.imageUrl} 
                  alt={news.title}
                  className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-700"
                />
                <div className="absolute top-4 left-4 z-20">
                  <span className="bg-emerald-600 text-white text-xs font-bold px-3 py-1.5 rounded-full shadow-md">
                    {news.category}
                  </span>
                </div>
              </div>

              {/* Content Body */}
              <div className="p-6 flex flex-col flex-1">
                <div className="flex items-center gap-4 text-xs text-gray-500 dark:text-gray-400 font-medium mb-3">
                  <div className="flex items-center gap-1.5">
                    <Clock className="w-3.5 h-3.5" />
                    <span>{news.date}</span>
                  </div>
                  <span>•</span>
                  <span>{news.readTime}</span>
                </div>
                
                <h3 className="text-xl font-bold text-gray-900 dark:text-white leading-snug mb-2 group-hover:text-emerald-700 dark:group-hover:text-emerald-400 transition-colors line-clamp-2">
                  {news.title}
                </h3>
                <h4 className="text-lg font-serif text-emerald-800 dark:text-emerald-300 mb-4 line-clamp-2" dir="rtl">
                  {news.urduTitle}
                </h4>
                
                <p className="text-gray-600 dark:text-gray-400 text-sm leading-relaxed mb-6 flex-1 border-b border-gray-200 dark:border-gray-800 pb-4 line-clamp-3">
                  {news.excerpt}
                </p>

                <div className="text-emerald-700 dark:text-emerald-400 text-sm font-bold flex items-center justify-between group-hover:text-emerald-800 dark:group-hover:text-emerald-300">
                  <span>Read full article</span>
                  <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                </div>
              </div>
            </MotionDiv>
          ))}
          </div>
        </div>

        {/* Tailwind specific custom scrollbar hiding block embedded as global css for this component */}
        <style dangerouslySetInnerHTML={{__html: `
          .scrollbar-hide::-webkit-scrollbar {
              display: none;
          }
        `}} />
      </div>
    </section>
  );
}
