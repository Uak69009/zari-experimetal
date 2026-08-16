"use client";

import dynamic from "next/dynamic";
import { ScanLine, ArrowRight } from "lucide-react";

const MotionDiv = dynamic(() => import("framer-motion").then((mod) => mod.motion.div), { ssr: false }) as any;
const MotionButton = dynamic(() => import("framer-motion").then((mod) => mod.motion.button), { ssr: false }) as any;

export default function DiagnosisShowcase() {
  return (
    <section className="w-full py-24 bg-white dark:bg-zari-bg text-gray-900 dark:text-gray-100 border-t border-gray-200 dark:border-gray-800 overflow-hidden relative transition-colors duration-300">
      
      {/* Soft Background Accent */}
      <div className="absolute top-1/2 left-1/4 w-[400px] h-[400px] bg-emerald-100/50 dark:bg-emerald-900/20 rounded-full blur-[120px] pointer-events-none -translate-y-1/2" />

      <div className="container mx-auto px-6 lg:px-12">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
          
          {/* Left Column: Text & Animated Button */}
          <MotionDiv 
            initial={{ opacity: 0, x: -30 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className="flex flex-col items-start gap-6"
          >
            <h2 className="text-4xl lg:text-5xl font-extrabold text-gray-900 dark:text-white leading-tight">
              Diagnose your sick crop <br />
              <span className="text-emerald-700 dark:text-emerald-400">in seconds.</span>
            </h2>
            
            <p className="text-lg text-gray-600 dark:text-gray-400 max-w-md leading-relaxed">
              Upload a photo of your leaf and let our edge-powered computer vision model instantly identify the disease and generate a localized treatment protocol.
            </p>

            {/* Call to Action Button */}
            <MotionButton
              whileHover={{ scale: 1.03 }}
              whileTap={{ scale: 0.97 }}
              className="mt-4 bg-emerald-700 dark:bg-zari-accent hover:bg-emerald-800 dark:hover:bg-emerald-400 border border-emerald-800 dark:border-zari-accent text-white dark:text-zari-bg px-8 py-4 rounded-full font-bold text-lg flex items-center gap-3 shadow-lg transition-all group"
            >
              <ScanLine className="w-5 h-5 text-white dark:text-zari-bg" />
              Get a free diagnosis
              <ArrowRight className="w-5 h-5 ml-2 group-hover:translate-x-1 transition-transform" />
            </MotionButton>
          </MotionDiv>

          {/* Right Column: Animated Phone Mockup */}
          <MotionDiv 
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="flex justify-center lg:justify-end"
          >
            {/* Phone Frame */}
            <div className="relative w-[320px] h-[650px] bg-gray-900 dark:bg-black border-[12px] border-gray-800 dark:border-gray-900 rounded-[3rem] overflow-hidden shadow-2xl">
              
              {/* Phone Top Notch */}
              <div className="absolute top-0 inset-x-0 h-7 bg-gray-800 dark:bg-gray-900 rounded-b-2xl w-40 mx-auto z-20 flex justify-center items-end pb-2">
                <div className="w-12 h-1.5 bg-gray-700 dark:bg-gray-800 rounded-full"></div>
              </div>

              {/* Scrolling Screen Content */}
              <MotionDiv
                animate={{ y: [0, -250, 0] }}
                transition={{ 
                  repeat: Infinity, 
                  duration: 15, 
                  ease: "easeInOut",
                  repeatType: "reverse"
                }}
                className="absolute top-0 left-0 w-full p-4 pt-12 flex flex-col gap-4 bg-white dark:bg-[#0c1a11]"
              >
                {/* Mock Image Upload Area */}
                <div className="w-full h-48 bg-emerald-50 dark:bg-emerald-900/30 rounded-xl border border-emerald-200 dark:border-emerald-800/50 flex items-center justify-center overflow-hidden relative">
                   <div className="text-emerald-800 dark:text-emerald-400 flex flex-col items-center">
                      <ScanLine className="w-10 h-10 mb-2 text-emerald-600 dark:text-emerald-400" />
                      <span className="text-sm font-semibold">Analyzing late_blight...</span>
                   </div>
                   {/* Scanning Line Animation */}
                   <MotionDiv 
                     animate={{ y: [-100, 100] }}
                     transition={{ repeat: Infinity, duration: 2, ease: "linear" }}
                     className="absolute w-full h-1 bg-emerald-500 shadow-[0_0_8px_#10B981] opacity-80"
                   />
                </div>

                {/* Mock Diagnosis Results */}
                <div className="w-full bg-gray-50 dark:bg-[#112417] rounded-xl p-4 border border-gray-200 dark:border-gray-800">
                  <h3 className="text-emerald-800 dark:text-emerald-400 font-bold mb-2">Detected: Late Blight</h3>
                  <div className="h-2.5 w-full bg-gray-200 dark:bg-gray-800 rounded-full overflow-hidden mb-3">
                    <MotionDiv 
                      initial={{ width: 0 }}
                      whileInView={{ width: "94%" }}
                      className="h-full bg-emerald-600 dark:bg-emerald-500"
                    />
                  </div>
                  <p className="text-xs text-gray-600 dark:text-gray-400 font-medium">Confidence Score: 94%</p>
                </div>

                {/* Mock Symptoms Section */}
                <div className="w-full bg-gray-50 dark:bg-[#112417] rounded-xl p-4 border border-gray-200 dark:border-gray-800">
                  <h4 className="text-gray-900 dark:text-white text-sm font-bold mb-3">Symptoms</h4>
                  <div className="space-y-2">
                    <div className="h-3 w-3/4 bg-gray-300 dark:bg-gray-700 rounded"></div>
                    <div className="h-3 w-full bg-gray-300 dark:bg-gray-700 rounded"></div>
                    <div className="h-3 w-5/6 bg-gray-300 dark:bg-gray-700 rounded"></div>
                  </div>
                </div>

                {/* Mock Treatment Section */}
                <div className="w-full bg-gray-50 dark:bg-[#112417] rounded-xl p-4 border border-gray-200 dark:border-gray-800">
                  <h4 className="text-gray-900 dark:text-white text-sm font-bold mb-3">Llama-3 Treatment Plan</h4>
                  <div className="space-y-2">
                    <div className="h-3 w-full bg-emerald-100 dark:bg-emerald-900/60 rounded"></div>
                    <div className="h-3 w-4/5 bg-emerald-100 dark:bg-emerald-900/60 rounded"></div>
                    <div className="h-3 w-full bg-emerald-100 dark:bg-emerald-900/60 rounded"></div>
                  </div>
                </div>
                
              </MotionDiv>
            </div>
          </MotionDiv>
          
        </div>
      </div>
    </section>
  );
}
