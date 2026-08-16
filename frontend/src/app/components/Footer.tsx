"use client";

import React from "react";
import { Leaf, Code2, Globe } from "lucide-react";

export default function Footer() {
  return (
    <footer className="w-full bg-white dark:bg-zari-bg border-t border-gray-200 dark:border-gray-800 text-gray-600 dark:text-gray-400 py-10 transition-colors duration-300">
      <div className="w-full mx-auto px-6 lg:px-12 flex flex-col md:flex-row items-center justify-between gap-6">
        
        {/* Left: Brand Identity */}
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-emerald-100 dark:bg-emerald-900/40 border border-emerald-200 dark:border-emerald-800/50 flex items-center justify-center">
            <Leaf className="w-5 h-5 text-emerald-700 dark:text-emerald-400" />
          </div>
          <div>
            <span className="text-gray-900 dark:text-white font-bold text-lg tracking-tight">
              ZARI<span className="text-emerald-600 dark:text-zari-accent">.ai</span>
            </span>
            <p className="text-xs text-gray-500 dark:text-gray-400">
              Autonomous Agricultural Intelligence Platform for Pakistan
            </p>
          </div>
        </div>

        {/* Center: Official Partner */}
        <div className="flex items-center gap-3 bg-white dark:bg-[#0B141A] border border-gray-200 dark:border-gray-800 px-5 py-2.5 rounded-full shadow-sm hover:shadow-md transition-shadow">
          <span className="text-xs font-bold text-gray-400 dark:text-gray-500 uppercase tracking-widest">Official Partner</span>
          <div className="h-6 w-px bg-gray-200 dark:bg-gray-800"></div>
          <a href="https://giki.edu.pk/" target="_blank" rel="noopener noreferrer" className="flex items-center gap-2 hover:opacity-80 transition-opacity" title="Visit GIKI Official Website">
            <img 
              src="https://upload.wikimedia.org/wikipedia/en/8/8e/Ghulam_Ishaq_Khan_Institute_of_Engineering_Sciences_and_Technology_%28insignia%29.png" 
              alt="GIKI Logo" 
              className="h-8 w-8 object-contain"
            />
            <span className="text-sm font-bold text-gray-900 dark:text-white tracking-tight">GIKI</span>
          </a>
        </div>

        {/* Right: Developer Attribution - Subtle & Non-Flashy */}
        <div className="flex items-center gap-4 text-xs text-gray-600 dark:text-gray-300 bg-gray-50 dark:bg-[#112417] border border-gray-200 dark:border-gray-800 px-4 py-2 rounded-full shadow-sm">
          <div className="flex items-center gap-1.5">
            <Code2 className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400" />
            <span>Developed by</span>
            <span className="text-gray-900 dark:text-white font-semibold tracking-wide">icode studios</span>
          </div>
          <span className="text-gray-300 dark:text-gray-600">|</span>
          <div className="flex items-center gap-1 text-gray-500 dark:text-gray-400 hover:text-emerald-700 dark:hover:text-emerald-300 transition-colors cursor-pointer">
            <Globe className="w-3.5 h-3.5" />
            <span>All Rights Reserved &copy; {new Date().getFullYear()}</span>
          </div>
        </div>

      </div>
    </footer>
  );
}
