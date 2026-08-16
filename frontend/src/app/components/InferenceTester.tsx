"use client";

import React, { useState } from "react";
import dynamic from "next/dynamic";
import { UploadCloud, CheckCircle, AlertTriangle, FileAudio, Activity, Globe, Shield, BookOpen, CheckSquare } from "lucide-react";

const MotionDiv = dynamic(() => import("framer-motion").then((mod) => mod.motion.div), { ssr: false }) as any;
const MotionButton = dynamic(() => import("framer-motion").then((mod) => mod.motion.button), { ssr: false }) as any;
const AnimatePresenceWrapper = dynamic(() => import("framer-motion").then((mod) => mod.AnimatePresence), { ssr: false }) as any;

interface PredictionResult {
  status: "accept" | "reject" | string;
  disease_class?: string;
  disease?: string;
  class_name?: string;
  confidence?: number;
  uncertainty?: number;
  scrc_threshold?: number;
  advisory?: string;
  treatment?: string;
  response?: string;
  symptoms?: string[];
  prevention?: string[];
  sources?: string[];
  audio_url?: string;
  message?: string;
}

export default function InferenceTester() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [language, setLanguage] = useState<string>("ur");
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setResult(null);
      setError(null);
    }
  };

  const diagnoseImage = async (imageFile: File, selectedLang: string): Promise<PredictionResult> => {
    const formData = new FormData();
    formData.append("file", imageFile);
    formData.append("image", imageFile);
    formData.append("language", selectedLang);

    const endpoints = [
      "http://localhost:8000/api/diagnose",
      "http://localhost:8000/predict"
    ];

    let lastError: Error | null = null;

    for (const url of endpoints) {
      try {
        const response = await fetch(url, {
          method: "POST",
          body: formData,
        });

        if (response.ok) {
          return await response.json();
        }
      } catch (err) {
        lastError = err as Error;
      }
    }

    throw lastError || new Error("Failed to connect to ZARI API backend server.");
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);

    try {
      const data = await diagnoseImage(file, language);
      setResult(data);
    } catch (err: unknown) {
      if (err instanceof Error) {
        setError(err.message || "Diagnosis failed. Please verify backend server is running.");
      } else {
        setError("Diagnosis failed. Please try again.");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col items-center w-full space-y-8">
      
      {/* Top Options Bar: Language Selector */}
      <div className="w-full max-w-3xl flex justify-between items-center bg-white dark:bg-[#112417] p-4 rounded-2xl border border-gray-200 dark:border-gray-800 shadow-sm">
        <div className="flex items-center space-x-2 text-gray-700 dark:text-gray-300 font-semibold text-sm">
          <Globe className="w-4 h-4 text-emerald-600 dark:text-emerald-400" />
          <span>Select Advisory Language / زبان منتخب کریں:</span>
        </div>

        <select
          value={language}
          onChange={(e) => setLanguage(e.target.value)}
          className="bg-emerald-50 dark:bg-emerald-950 border border-emerald-300 dark:border-emerald-700 text-emerald-900 dark:text-emerald-100 font-bold text-sm rounded-xl px-4 py-2 focus:outline-none focus:ring-2 focus:ring-emerald-500 cursor-pointer"
        >
          <option value="ur">Urdu (اردو)</option>
          <option value="ps">Pashto (پښتو)</option>
          <option value="en">English</option>
        </select>
      </div>

      {/* 3D Flip Upload Zone Wrapper */}
      <div className="w-full max-w-3xl flex flex-col items-center space-y-4">
        <div className="w-full perspective-[1200px] h-72 cursor-pointer group">
          <MotionDiv 
            className="w-full h-full relative"
            style={{ transformStyle: "preserve-3d" }}
            initial={false}
            animate={{ rotateY: 0 }}
            whileHover={{ rotateY: 180 }}
            transition={{ duration: 0.7, type: "spring", stiffness: 100, damping: 20 }}
          >
            {/* Front Face (Summary / Info) */}
            <div 
              className="absolute inset-0 w-full h-full flex flex-col items-center justify-center p-8 bg-white dark:bg-[#112417] border border-gray-200 dark:border-gray-800 rounded-2xl shadow-lg transition-colors"
              style={{ backfaceVisibility: "hidden" }}
            >
              <div className="flex items-center gap-3 mb-4">
                <Activity className="text-emerald-600 dark:text-emerald-400 w-6 h-6" />
                <h3 className="text-xl font-bold text-gray-900 dark:text-white">ZARI AI Field Diagnostics</h3>
              </div>
              <p className="text-gray-600 dark:text-gray-400 text-center text-sm mb-4 leading-relaxed px-4">
                Powered by Evidential Deep Learning (EDL) & Qdrant RAG. Upload a clear crop leaf image for real-time disease identification and Pakistan-specific IPM advice.
              </p>
              <h4 className="text-lg font-serif text-emerald-800 dark:text-emerald-300 text-center leading-relaxed" dir="rtl">
                زراعت پاکستان کی معیشت کی ریڑھ کی ہڈی ہے۔ جدید ٹیکنالوجی کے ذریعے فصلوں کی بروقت نگرانی یقینی بنائیں۔
              </h4>
              
              {/* Interactive hint */}
              <div className="absolute bottom-4 flex items-center justify-center w-full text-xs text-gray-400 dark:text-gray-500 font-bold uppercase tracking-widest animate-pulse">
                Hover to Upload Crop Image
              </div>
            </div>

            {/* Back Face (Actual Upload Zone) */}
            <div 
              className="absolute inset-0 w-full h-full flex flex-col items-center justify-center border-2 border-dashed border-emerald-300 dark:border-emerald-700 rounded-2xl p-10 bg-emerald-50/90 dark:bg-emerald-900/40 hover:bg-emerald-50 dark:hover:bg-emerald-900/60 transition-colors backdrop-blur-sm shadow-xl"
              style={{ backfaceVisibility: "hidden", transform: "rotateY(180deg)" }}
            >
              <input
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
              />
              <div className="text-center flex flex-col items-center">
                <div className="w-16 h-16 rounded-full bg-emerald-100 dark:bg-emerald-800/80 border border-emerald-200 dark:border-emerald-700 flex items-center justify-center mb-4 shadow-sm">
                  <UploadCloud className="text-emerald-700 dark:text-emerald-400 w-8 h-8" />
                </div>
                <p className="text-emerald-900 dark:text-emerald-100 font-bold text-xl mb-2">Drop your crop leaf image here</p>
                <p className="text-sm text-emerald-700/70 dark:text-emerald-400/70 font-medium bg-emerald-100/50 dark:bg-emerald-900/30 px-4 py-1.5 rounded-full">
                  Click to browse (JPG, PNG, WEBP)
                </p>
              </div>
            </div>
          </MotionDiv>
        </div>

        {/* Instructional Text Below Card */}
        <p className="text-sm text-gray-500 dark:text-gray-400 font-medium text-center">
          Hover over the card to reveal the upload dropzone. Upload a clear, well-lit image of your crop leaf to run AI diagnostics.
        </p>
      </div>

      {/* Preview & Action */}
      <AnimatePresenceWrapper>
        {preview && (
          <MotionDiv 
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="flex flex-col items-center space-y-6 w-full"
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img 
              src={preview} 
              alt="Crop Preview" 
              className="max-h-72 object-contain rounded-2xl shadow-md border border-gray-200 dark:border-gray-800" 
            />
            <MotionButton
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={handleAnalyze}
              disabled={loading}
              className="px-10 py-4 bg-emerald-700 hover:bg-emerald-800 border border-emerald-800 text-white font-extrabold rounded-full shadow-md disabled:opacity-70 disabled:cursor-not-allowed transition-all w-full max-w-sm flex items-center justify-center space-x-3"
            >
              {loading ? (
                <>
                  <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                  <span>Running AI Diagnostics...</span>
                </>
              ) : (
                <>
                  <Activity className="w-5 h-5 text-white" />
                  <span>Run AI Diagnostic ({language.toUpperCase()})</span>
                </>
              )}
            </MotionButton>
          </MotionDiv>
        )}
      </AnimatePresenceWrapper>

      {/* Error State */}
      {error && (
        <MotionDiv initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="p-4 bg-red-50 dark:bg-red-950/50 border-l-4 border-red-500 text-red-700 dark:text-red-300 rounded-lg w-full font-medium flex items-center space-x-3">
          <AlertTriangle size={24} className="text-red-500 shrink-0" />
          <span>{error}</span>
        </MotionDiv>
      )}

      {/* Results Dashboard */}
      <AnimatePresenceWrapper>
        {result && (
          <MotionDiv 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="w-full space-y-6 pt-4"
          >
            {/* Rejection Warning Banner */}
            {result.status === "reject" && (
              <div className="bg-amber-50 dark:bg-amber-950/60 border-l-4 border-amber-500 p-5 rounded-2xl text-amber-900 dark:text-amber-200 shadow-sm flex items-start space-x-4">
                <AlertTriangle className="text-amber-600 dark:text-amber-400 w-6 h-6 shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-extrabold text-lg">Image Unclear / High Model Uncertainty (SCRC Risk Control Triggered)</h4>
                  <p className="text-sm mt-1">
                    The AI model detected high uncertainty (u = {result.uncertainty?.toFixed(4) || "0.8500"} &gt; tau 0.8050). Please upload a clearer, close-up photo of the affected crop leaf to ensure safety.
                  </p>
                </div>
              </div>
            )}

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              
              {/* CV Inference Card */}
              <div className="bg-white dark:bg-[#112417] shadow-md rounded-2xl p-6 border border-gray-200 dark:border-gray-800 flex flex-col h-full transition-colors">
                <div className="flex items-center justify-between mb-6 border-b border-gray-100 dark:border-gray-800 pb-4">
                  <div className="flex items-center space-x-3">
                    <div className={`p-2 rounded-lg ${result.status === 'accept' ? 'bg-emerald-100 dark:bg-emerald-900/40' : 'bg-amber-100 dark:bg-amber-900/40'}`}>
                      {result.status === "accept" ? (
                        <CheckCircle className="text-emerald-700 dark:text-emerald-400" size={24} />
                      ) : (
                        <AlertTriangle className="text-amber-600 dark:text-amber-400" size={24} />
                      )}
                    </div>
                    <h3 className="text-lg font-bold text-gray-900 dark:text-white">Diagnostic Result</h3>
                  </div>

                  <span className={`px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider ${result.status === 'accept' ? 'bg-emerald-100 text-emerald-800 dark:bg-emerald-900/60 dark:text-emerald-300' : 'bg-amber-100 text-amber-800 dark:bg-amber-900/60 dark:text-amber-300'}`}>
                    {result.status === "accept" ? "VERIFIED ACCURATE" : "REJECTED (UNCERTAIN)"}
                  </span>
                </div>
                
                <div className="flex-1 flex flex-col justify-center">
                  <h4 className="text-3xl font-extrabold text-gray-900 dark:text-white mb-6 text-center">
                    {(result.disease_class || result.class_name || result.disease || "Unknown").replace(/_/g, " ")}
                  </h4>
                  
                  {/* Confidence Progress Bar */}
                  <div className="w-full bg-gray-100 dark:bg-gray-800 rounded-full h-3.5 mb-2 overflow-hidden shadow-inner border border-gray-200 dark:border-gray-700">
                    <MotionDiv 
                      initial={{ width: 0 }}
                      animate={{ width: `${(result.confidence || 0) * 100}%` }}
                      transition={{ duration: 1, delay: 0.2 }}
                      className={`h-full rounded-full ${result.confidence && result.confidence >= 0.85 ? 'bg-emerald-600 dark:bg-emerald-500' : 'bg-amber-500 dark:bg-amber-400'}`} 
                    ></MotionDiv>
                  </div>

                  <div className="flex justify-between text-sm font-semibold text-gray-600 dark:text-gray-400 mb-4">
                    <span>Model Confidence</span>
                    <span className="text-emerald-700 dark:text-emerald-400 font-bold">{((result.confidence || 0) * 100).toFixed(1)}%</span>
                  </div>

                  {/* Uncertainty & SCRC Indicator */}
                  {result.uncertainty !== undefined && (
                    <div className="p-3 bg-gray-50 dark:bg-gray-900/50 rounded-xl border border-gray-200 dark:border-gray-800 flex justify-between items-center text-xs font-medium text-gray-600 dark:text-gray-400">
                      <span>EDL Uncertainty Score:</span>
                      <span className={`font-mono font-bold ${result.uncertainty <= (result.scrc_threshold || 0.805) ? 'text-emerald-600 dark:text-emerald-400' : 'text-amber-600 dark:text-amber-400'}`}>
                        {result.uncertainty.toFixed(4)} (Threshold: {result.scrc_threshold || 0.805})
                      </span>
                    </div>
                  )}
                </div>
              </div>

              {/* LLM Advisory Card */}
              {(result.advisory || result.treatment || result.response) && (
                <div className="bg-white dark:bg-[#112417] shadow-md rounded-2xl p-6 border border-gray-200 dark:border-gray-800 flex flex-col h-full transition-colors">
                  <div className="flex items-center justify-between mb-6 border-b border-gray-100 dark:border-gray-800 pb-4">
                    <h3 className="text-lg font-bold text-gray-900 dark:text-white">ZARI RAG Expert Advisory</h3>
                    <div className="px-3 py-1 bg-emerald-100 dark:bg-emerald-900/40 text-emerald-800 dark:text-emerald-300 rounded-full text-xs font-bold tracking-wide border border-emerald-200 dark:border-emerald-800/50">
                      {language.toUpperCase()} ADVISORY
                    </div>
                  </div>
                  <div 
                    className="flex-1 whitespace-pre-line text-gray-800 dark:text-gray-200 text-sm md:text-base leading-relaxed font-serif" 
                    dir={language === "en" ? "ltr" : "rtl"}
                  >
                    {result.advisory || result.treatment || result.response}
                  </div>
                </div>
              )}

            </div>

            {/* Structured Evidence Cards Grid */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              
              {/* Symptoms Card */}
              {result.symptoms && result.symptoms.length > 0 && (
                <div className="bg-white dark:bg-[#112417] shadow-md rounded-2xl p-6 border border-gray-200 dark:border-gray-800">
                  <div className="flex items-center space-x-2 mb-4 text-emerald-800 dark:text-emerald-400 font-bold border-b border-gray-100 dark:border-gray-800 pb-3">
                    <CheckSquare className="w-5 h-5 text-emerald-600 dark:text-emerald-400" />
                    <h4>Symptoms to Verify</h4>
                  </div>
                  <ul className="space-y-2 text-xs text-gray-700 dark:text-gray-300">
                    {result.symptoms.map((sym, idx) => (
                      <li key={idx} className="flex items-start space-x-2">
                        <span className="text-emerald-600 dark:text-emerald-400 font-bold">•</span>
                        <span>{sym}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Prevention Card */}
              {result.prevention && result.prevention.length > 0 && (
                <div className="bg-white dark:bg-[#112417] shadow-md rounded-2xl p-6 border border-gray-200 dark:border-gray-800">
                  <div className="flex items-center space-x-2 mb-4 text-emerald-800 dark:text-emerald-400 font-bold border-b border-gray-100 dark:border-gray-800 pb-3">
                    <Shield className="w-5 h-5 text-emerald-600 dark:text-emerald-400" />
                    <h4>Prevention & Sanitation</h4>
                  </div>
                  <ul className="space-y-2 text-xs text-gray-700 dark:text-gray-300">
                    {result.prevention.map((prev, idx) => (
                      <li key={idx} className="flex items-start space-x-2">
                        <span className="text-emerald-600 dark:text-emerald-400 font-bold">•</span>
                        <span>{prev}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Verified Sources Badges */}
              {result.sources && result.sources.length > 0 && (
                <div className="bg-white dark:bg-[#112417] shadow-md rounded-2xl p-6 border border-gray-200 dark:border-gray-800">
                  <div className="flex items-center space-x-2 mb-4 text-emerald-800 dark:text-emerald-400 font-bold border-b border-gray-100 dark:border-gray-800 pb-3">
                    <BookOpen className="w-5 h-5 text-emerald-600 dark:text-emerald-400" />
                    <h4>Verified Citations</h4>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {result.sources.map((src, idx) => (
                      <span key={idx} className="px-3 py-1.5 bg-emerald-50 dark:bg-emerald-950 border border-emerald-200 dark:border-emerald-800 text-emerald-900 dark:text-emerald-200 text-xs font-bold rounded-xl">
                        {src}
                      </span>
                    ))}
                  </div>
                </div>
              )}

            </div>

          </MotionDiv>
        )}
      </AnimatePresenceWrapper>
    </div>
  );
}
