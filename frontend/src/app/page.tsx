"use client";

import { useState, useEffect } from "react";
import dynamic from "next/dynamic";
import { Globe2, Leaf, MessageCircle, Activity, ScanLine } from "lucide-react";
import InferenceTester from "./components/InferenceTester";
import DiagnosisShowcase from "./components/DiagnosisShowcase";
import CropCards from "./components/CropCards";
import AgriStoreAds from "./components/AgriStoreAds";
import NewsSection from "./components/NewsSection";
import TeamAndContact from "./components/TeamAndContact";
import Footer from "./components/Footer";

const MotionDiv = dynamic(() => import("framer-motion").then((mod) => mod.motion.div), { ssr: false }) as any;
const MotionSpan = dynamic(() => import("framer-motion").then((mod) => mod.motion.span), { ssr: false }) as any;
const AnimatePresenceWrapper = dynamic(() => import("framer-motion").then((mod) => mod.AnimatePresence), { ssr: false }) as any;

// A custom component to flip English text to Urdu on hover
const FlipText = ({ english, urdu }: { english: string; urdu: string }) => {
  const [isHovered, setIsHovered] = useState(false);

  return (
    <span 
      className="relative inline-block cursor-default perspective-1000"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      onTouchStart={() => setIsHovered(!isHovered)}
    >
      <AnimatePresenceWrapper mode="wait">
        {!isHovered ? (
          <MotionSpan
            key="english"
            initial={{ rotateX: -90, opacity: 0 }}
            animate={{ rotateX: 0, opacity: 1 }}
            exit={{ rotateX: 90, opacity: 0 }}
            transition={{ duration: 0.3 }}
            className="inline-block"
          >
            {english}
          </MotionSpan>
        ) : (
          <MotionSpan
            key="urdu"
            initial={{ rotateX: -90, opacity: 0 }}
            animate={{ rotateX: 0, opacity: 1 }}
            exit={{ rotateX: 90, opacity: 0 }}
            transition={{ duration: 0.3 }}
            className="inline-block text-emerald-700 font-serif"
            dir="rtl"
          >
            {urdu}
          </MotionSpan>
        )}
      </AnimatePresenceWrapper>
    </span>
  );
};

export default function Home() {
  const [mounted, setMounted] = useState(false);
  
  useEffect(() => {
    setMounted(true);
  }, []);

  // Map pulse points
  const pulsePoints = [
    { top: "35%", left: "68%", duration: 3.5, delay: 0 },   
    { top: "38%", left: "66%", duration: 4.2, delay: 1.5 }, 
    { top: "45%", left: "18%", duration: 5.0, delay: 0.5 }, 
    { top: "55%", left: "75%", duration: 3.8, delay: 2.1 }, 
    { top: "60%", left: "45%", duration: 4.5, delay: 1.0 }, 
    { top: "25%", left: "52%", duration: 3.2, delay: 0.8 }, 
  ];

  return (
    <main className="min-h-screen bg-white dark:bg-zari-bg text-gray-900 dark:text-gray-100 relative overflow-x-hidden font-sans transition-colors duration-300">
      
      {/* 1. Light Map Background */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden z-0 flex items-center justify-center">
        <div 
          className="absolute w-full h-[800px] opacity-10 dark:opacity-5 bg-no-repeat bg-center bg-contain"
          style={{ 
            backgroundImage: "url('https://upload.wikimedia.org/wikipedia/commons/8/80/World_map_-_low_resolution.svg')",
            filter: "brightness(0) saturate(100%) invert(32%) sepia(85%) saturate(542%) hue-rotate(97deg) brightness(92%) contrast(92%)"
          }}
        ></div>
        
        {/* Soft Background Accent Circles */}
        <div className="absolute top-[-10%] left-[-10%] w-[500px] h-[500px] bg-emerald-100 dark:bg-emerald-900/20 opacity-60 rounded-full blur-[120px]" />
        <div className="absolute bottom-[-20%] right-[-10%] w-[600px] h-[600px] bg-emerald-50 dark:bg-emerald-900/10 opacity-80 rounded-full blur-[150px]" />

        {/* Pulse Points */}
        {mounted && pulsePoints.map((point, idx) => (
          <MotionDiv
            key={idx}
            className="absolute w-3 h-3 md:w-4 md:h-4 bg-emerald-600 dark:bg-zari-accent rounded-full shadow-[0_0_12px_#059669] dark:shadow-[0_0_12px_#00FFA3]"
            style={{ top: point.top, left: point.left }}
            animate={{ scale: [1, 1.8, 1], opacity: [0.7, 0.3, 0.7] }}
            transition={{ duration: point.duration, repeat: Infinity, delay: point.delay, ease: "easeInOut" }}
          />
        ))}
      </div>

      <div className="relative z-10 container mx-auto px-6 py-16 flex flex-col items-center">
        
        {/* Header Section */}
        <MotionDiv 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-center mb-16 max-w-4xl"
        >
          <div className="flex justify-center items-center gap-3 mb-4">
            <div className="p-3 bg-emerald-50 dark:bg-emerald-900/40 border border-emerald-200 dark:border-emerald-800/50 rounded-2xl shadow-sm">
              <Globe2 className="text-emerald-700 dark:text-emerald-400 w-8 h-8" />
            </div>
            <h1 className="text-5xl font-extrabold text-gray-900 dark:text-white tracking-tight">
              ZARI<span className="text-emerald-600 dark:text-zari-accent">.ai</span>
            </h1>
          </div>
          
          {/* Flip Title */}
          <h2 className="text-3xl md:text-4xl font-bold text-gray-900 dark:text-white mb-6 flex flex-col items-center gap-2">
            <FlipText 
              english="Autonomous agricultural intelligence." 
              urdu="خودمختار زرعی ذہانت۔" 
            />
          </h2>
          
          <div className="text-lg md:text-xl text-gray-600 dark:text-gray-400 mb-2">
            <FlipText 
              english="Upload a leaf scan for instant, edge-powered disease diagnostics and localized treatment protocols." 
              urdu="فوری، جدید بیماریوں کی تشخیص اور مقامی علاج کے لیے پتے کی تصویر اپ لوڈ کریں۔" 
            />
          </div>
          <p className="text-xs font-semibold text-emerald-700 dark:text-emerald-400 bg-emerald-50 dark:bg-emerald-900/30 border border-emerald-200 dark:border-emerald-800/50 inline-block px-3 py-1 rounded-full uppercase tracking-wider mt-4">
            [Hover text to translate to Urdu]
          </p>
        </MotionDiv>

        {/* Clean Diagnostic Zone */}
        <MotionDiv 
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="w-full max-w-4xl"
        >
          <div className="bg-white dark:bg-[#112417] border border-gray-200 dark:border-gray-800 rounded-3xl p-8 md:p-12 shadow-xl hover:shadow-2xl transition-all">
            <InferenceTester />
          </div>
        </MotionDiv>

        {/* Feature Highlights */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mt-20 w-full max-w-5xl">
          {[
            { title: "Computer Vision", desc: "Powered by EfficientNetV2-S for rapid, high-accuracy inference.", icon: <Activity className="w-6 h-6 text-emerald-700 dark:text-emerald-400" /> },
            { title: "LLM Advisory", desc: "Actionable, localized treatment protocols generated via Llama-3.3.", icon: <Globe2 className="w-6 h-6 text-emerald-700 dark:text-emerald-400" /> },
            { title: "Voice & Edge", desc: "Accessible globally via WhatsApp integration and Edge-TTS.", icon: <Leaf className="w-6 h-6 text-emerald-700 dark:text-emerald-400" /> }
          ].map((feature, idx) => (
            <MotionDiv 
              key={idx}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.4 + (idx * 0.1) }}
              className="bg-white dark:bg-[#0c1a11] border border-gray-200 dark:border-gray-800 rounded-2xl p-6 shadow-sm hover:shadow-md hover:border-emerald-300 dark:hover:border-emerald-700 transition-all"
            >
              <div className="bg-emerald-50 dark:bg-emerald-900/30 border border-emerald-200 dark:border-emerald-800/50 w-12 h-12 rounded-xl flex items-center justify-center mb-4">
                {feature.icon}
              </div>
              <h3 className="text-gray-900 dark:text-white font-bold text-lg mb-2">{feature.title}</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed">{feature.desc}</p>
            </MotionDiv>
          ))}
        </div>

        {/* WhatsApp Section */}
        <MotionDiv
          initial={{ opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8 }}
          className="mt-28 w-full max-w-5xl bg-emerald-900 dark:bg-[#07130a] text-white border border-emerald-800 dark:border-emerald-900/50 rounded-3xl p-10 md:p-16 flex flex-col md:flex-row items-center gap-10 shadow-xl"
        >
          <div className="flex-1">
            <div className="flex items-center gap-3 mb-6">
              <div className="bg-white/10 dark:bg-white/5 p-3 rounded-full">
                <MessageCircle className="w-8 h-8 text-[#25D366]" />
              </div>
              <h2 className="text-3xl font-bold text-white">WhatsApp Integration</h2>
            </div>
            <h3 className="text-xl font-semibold text-emerald-100 dark:text-emerald-400 mb-4">
              <FlipText 
                english="No internet? No web app? No problem." 
                urdu="انٹرنیٹ نہیں؟ ویب ایپ نہیں؟ کوئی مسئلہ نہیں۔" 
              />
            </h3>
            <p className="text-emerald-100/90 dark:text-gray-300 text-lg leading-relaxed mb-8">
              ZARI.ai runs a dedicated WhatsApp webhook node. Farmers directly in the field can simply take a photo of an infected crop and send it to our automated WhatsApp number. ZARI will reply instantly with voice notes detailing the exact diagnosis and cure in fluent Urdu.
            </p>
            <a 
              href="https://wa.me/+923170478541?text=Hello%20ZARI%21%20I%20need%20to%20detect%20a%20crop%20illness."
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex bg-[#25D366] hover:bg-[#1EBE5A] text-gray-900 font-bold px-8 py-3.5 rounded-full items-center gap-2.5 transition-transform hover:scale-105 shadow-md w-max"
            >
              <MessageCircle className="w-5 h-5" />
              Message ZARI on WhatsApp
            </a>
          </div>
          
          {/* Mockup */}
          <div className="w-full md:w-[300px] h-[450px] bg-[#0B141A] rounded-3xl border-8 border-gray-800 shadow-2xl relative overflow-hidden flex flex-col">
            <div className="bg-[#202C33] p-4 flex items-center gap-3">
              <div className="w-10 h-10 bg-emerald-600 rounded-full flex items-center justify-center">
                <Leaf className="w-5 h-5 text-white" />
              </div>
              <div>
                <h4 className="text-white font-semibold text-sm">ZARI.ai Bot</h4>
                <p className="text-[#25D366] text-xs">online</p>
              </div>
            </div>
            <div className="flex-1 p-4 bg-[#0B141A] bg-[url('https://i.imgur.com/4p99V1D.png')] bg-cover bg-blend-overlay flex flex-col gap-3">
              <div className="self-end bg-[#005C4B] text-white p-2 rounded-lg rounded-tr-none max-w-[80%] text-sm">
                <div className="flex items-center justify-center w-full h-24 bg-black/30 rounded mb-1 border border-white/10">
                  <ScanLine className="w-6 h-6 text-white/50" />
                </div>
                What's wrong with my potato crop?
              </div>
              <div className="self-start bg-[#202C33] text-white p-3 rounded-lg rounded-tl-none max-w-[90%] text-sm border border-white/5">
                <span className="text-[#25D366] font-bold block mb-1">Diagnosis: Late Blight (98%)</span>
                یہ فنگس کا حملہ ہے۔ فوری طور پر مینکو زیب سپرے کریں۔ (Voice Note Attached)
              </div>
            </div>
          </div>
        </MotionDiv>
      </div>

      <DiagnosisShowcase />
      <CropCards />
      
      {/* Recommended Agri-Chemicals Ads */}
      <AgriStoreAds />

      {/* Latest Agricultural News */}
      <NewsSection />
      
      {/* Team & Contact Section */}
      <TeamAndContact />
      
      <Footer />
    </main>
  );
}
