"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { Leaf, Menu, X, Sun, Moon, ArrowRight, CloudSun, CloudRain, Cloud, CloudLightning, Loader2 } from "lucide-react";

export default function Navbar() {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [isWeatherOpen, setIsWeatherOpen] = useState(false);
  const [weatherData, setWeatherData] = useState<any>(null);
  const [isLoadingWeather, setIsLoadingWeather] = useState(true);

  const defaultCities = [
    { name: "Lahore", lat: 31.5204, lon: 74.3587 },
    { name: "Karachi", lat: 24.8607, lon: 67.0011 },
    { name: "Islamabad", lat: 33.6844, lon: 73.0479 },
    { name: "Multan", lat: 30.1978, lon: 71.4697 },
    { name: "Faisalabad", lat: 31.4181, lon: 73.0776 },
    { name: "Peshawar", lat: 34.0151, lon: 71.5249 },
    { name: "Quetta", lat: 30.1798, lon: 66.9750 }
  ];
  
  const [availableCities, setAvailableCities] = useState(defaultCities);
  const [selectedCity, setSelectedCity] = useState(defaultCities[0]);
  const [hasRequestedLocation, setHasRequestedLocation] = useState(false);

  // Sync initial dark mode state on mount
  useEffect(() => {
    if (document.documentElement.classList.contains('dark')) {
      setIsDarkMode(true);
    }
  }, []);

  // Automatically attempt to get user's real location
  useEffect(() => {
    if (typeof window !== "undefined" && navigator.geolocation && !hasRequestedLocation) {
      setHasRequestedLocation(true);
      navigator.geolocation.getCurrentPosition(
        async (position) => {
          const lat = position.coords.latitude;
          const lon = position.coords.longitude;
          let locationName = "Current Location";
          
          try {
            // Reverse Geocoding via BigDataCloud (much cleaner city names than OSM)
            const geoRes = await fetch(`https://api.bigdatacloud.net/data/reverse-geocode-client?latitude=${lat}&longitude=${lon}&localityLanguage=en`);
            const geoData = await geoRes.json();
            
            if (geoData && geoData.city) {
              locationName = geoData.city;
            } else if (geoData && geoData.locality) {
              locationName = geoData.locality;
            } else if (geoData && geoData.principalSubdivision) {
              locationName = geoData.principalSubdivision;
            }
          } catch (e) {
            console.warn("Reverse geocoding failed", e);
          }

          const userCity = { 
            name: `${locationName} (Auto)`, 
            lat: lat, 
            lon: lon 
          };
          
          setAvailableCities((prev) => {
            const filtered = prev.filter(c => !c.name.includes("(Auto)") && c.name !== "Current Location");
            return [userCity, ...filtered];
          });
          setSelectedCity(userCity);
        },
        (error) => {
          console.warn("Geolocation denied or failed. Falling back to default.", error);
        }
      );
    }
  }, [hasRequestedLocation]);

  // Fetch Live Weather Data from Open-Meteo (No API Key Required)
  useEffect(() => {
    async function fetchWeather() {
      setIsLoadingWeather(true);
      try {
        const lat = selectedCity.lat;
        const lon = selectedCity.lon;
        
        const res = await fetch(`https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&current=temperature_2m,weather_code&daily=weather_code,temperature_2m_max,temperature_2m_min&timezone=auto`);
        const data = await res.json();
        setWeatherData(data);
      } catch(e) {
        console.error("Failed to fetch weather data", e);
      } finally {
        setIsLoadingWeather(false);
      }
    }
    fetchWeather();
  }, [selectedCity]);

  const toggleMenu = () => {
    setIsMobileMenuOpen(!isMobileMenuOpen);
  };

  const toggleDarkMode = () => {
    if (isDarkMode) {
      document.documentElement.classList.remove('dark');
      localStorage.setItem('theme', 'light');
      setIsDarkMode(false);
    } else {
      document.documentElement.classList.add('dark');
      localStorage.setItem('theme', 'dark');
      setIsDarkMode(true);
    }
  };

  const navLinks = [
    { name: "Home", href: "/" },
    { name: "Diagnostics", href: "#diagnostics" },
    { name: "About ZARI", href: "#about" },
    { name: "Contact", href: "#contact" },
  ];

  // WMO Weather Interpretation Codes
  const getWeatherIcon = (code: number, size = 16) => {
    if (code === 0) return <Sun size={size} className="text-amber-500" />;
    if (code === 1 || code === 2) return <CloudSun size={size} className="text-amber-500" />;
    if (code === 3) return <Cloud size={size} className="text-gray-500" />;
    if (code >= 51 && code <= 67) return <CloudRain size={size} className="text-blue-500" />;
    if (code >= 80 && code <= 82) return <CloudRain size={size} className="text-blue-500" />;
    if (code >= 95) return <CloudLightning size={size} className="text-purple-500" />;
    return <Cloud size={size} className="text-gray-500" />;
  };

  const getWeatherDesc = (code: number) => {
    if (code === 0) return "Clear Sky";
    if (code === 1 || code === 2) return "Partly Cloudy";
    if (code === 3) return "Overcast";
    if (code >= 51 && code <= 67) return "Rain";
    if (code >= 80 && code <= 82) return "Showers";
    if (code >= 95) return "Thunderstorms";
    return "Cloudy";
  };

  const getDayName = (dateString: string) => {
    const date = new Date(dateString);
    const today = new Date();
    if (date.toDateString() === today.toDateString()) return "Today";
    return date.toLocaleDateString('en-US', { weekday: 'short' });
  };

  const renderWeatherDropdown = () => {
    if (isLoadingWeather) {
      return (
        <div className="absolute top-full right-0 mt-3 w-64 bg-white dark:bg-[#112417] border border-gray-200 dark:border-gray-800 rounded-2xl shadow-xl z-50 p-8 flex flex-col items-center justify-center">
          <Loader2 className="w-8 h-8 text-emerald-500 animate-spin mb-2" />
          <p className="text-sm text-gray-500 dark:text-gray-400">Fetching live weather...</p>
        </div>
      );
    }

    if (!weatherData) return null;

    return (
      <div className="absolute top-full right-0 mt-3 w-72 bg-white dark:bg-[#112417] border border-gray-200 dark:border-gray-800 rounded-2xl shadow-xl z-50 overflow-hidden">
        <div className="p-4 bg-emerald-50 dark:bg-emerald-900/20 border-b border-gray-100 dark:border-gray-800 flex items-center justify-between">
          <div>
            <h3 className="font-bold text-gray-900 dark:text-white flex items-center gap-2">
              {getWeatherIcon(weatherData.current.weather_code, 20)} {Math.round(weatherData.current.temperature_2m)}°C
            </h3>
            <div className="flex items-center mt-1">
              <span className="text-xs text-gray-500 dark:text-gray-400 mr-1">{getWeatherDesc(weatherData.current.weather_code)},</span>
              <select 
                value={selectedCity.name}
                onChange={(e) => {
                  const city = availableCities.find(c => c.name === e.target.value);
                  if (city) setSelectedCity(city);
                }}
                className="text-xs font-semibold text-emerald-700 dark:text-emerald-400 bg-transparent border-none outline-none cursor-pointer hover:text-emerald-900 dark:hover:text-emerald-300 p-0"
              >
                {availableCities.map(c => <option key={c.name} value={c.name} className="text-gray-900">{c.name}</option>)}
              </select>
            </div>
          </div>
          <button onClick={() => setIsWeatherOpen(false)} className="text-gray-400 hover:text-gray-700 dark:hover:text-white transition-colors">
            <X size={18} />
          </button>
        </div>
        <div className="p-2 flex flex-col gap-1 max-h-72 overflow-y-auto scrollbar-hide">
          {weatherData.daily.time.map((time: string, idx: number) => (
            <div key={idx} className="flex items-center justify-between p-2 hover:bg-gray-50 dark:hover:bg-gray-800/50 rounded-lg transition-colors">
              <span className="text-sm font-semibold text-gray-700 dark:text-gray-300 w-16">{getDayName(time)}</span>
              <div className="flex items-center gap-2 flex-1">
                {getWeatherIcon(weatherData.daily.weather_code[idx], 16)}
                <span className="text-xs text-gray-500 dark:text-gray-400 truncate">{getWeatherDesc(weatherData.daily.weather_code[idx])}</span>
              </div>
              <div className="flex items-center gap-2 text-sm">
                <span className="font-bold text-gray-900 dark:text-white">{Math.round(weatherData.daily.temperature_2m_max[idx])}°</span>
                <span className="text-gray-400 dark:text-gray-500">{Math.round(weatherData.daily.temperature_2m_min[idx])}°</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  const currentWeatherIcon = weatherData ? getWeatherIcon(weatherData.current.weather_code, 20) : <CloudSun size={20} />;

  return (
    <header className="w-full bg-white/90 dark:bg-zari-bg/90 backdrop-blur-md border-b border-gray-200 dark:border-gray-800/60 sticky top-0 z-50 transition-colors duration-300">
      <div className="w-full mx-auto px-6 lg:px-12 h-20 flex items-center justify-between">
        
        {/* Logo */}
        <Link href="/" className="flex items-center gap-2.5 group">
          <div className="w-10 h-10 rounded-xl bg-emerald-100 dark:bg-emerald-900/40 border border-emerald-200 dark:border-emerald-800/50 flex items-center justify-center group-hover:bg-emerald-600 dark:group-hover:bg-zari-accent transition-colors">
            <Leaf className="text-emerald-700 dark:text-emerald-400 group-hover:text-white dark:group-hover:text-zari-bg w-6 h-6 transition-colors" />
          </div>
          <h1 className="text-2xl font-extrabold text-gray-900 dark:text-white tracking-tight">
            ZARI<span className="text-emerald-600 dark:text-zari-accent">.ai</span>
          </h1>
        </Link>

        {/* Desktop Navigation Links & Actions */}
        <div className="hidden md:flex items-center gap-8">
          <nav className="flex items-center gap-8">
            {navLinks.map((link) => (
              <Link 
                key={link.name} 
                href={link.href} 
                className="text-sm font-semibold text-gray-700 dark:text-gray-300 hover:text-emerald-600 dark:hover:text-zari-accent transition-colors"
              >
                {link.name}
              </Link>
            ))}
          </nav>
          
          <div className="flex items-center gap-4 border-l border-gray-200 dark:border-gray-700 pl-8">
            {/* Weather Toggle & Dropdown */}
            <div className="relative">
              <button
                onClick={() => setIsWeatherOpen(!isWeatherOpen)}
                className="p-2 rounded-full hover:bg-gray-100 dark:hover:bg-gray-800 transition-all focus:outline-none flex items-center justify-center"
                title="View Live Weather Forecast"
              >
                {currentWeatherIcon}
              </button>
              
              {isWeatherOpen && renderWeatherDropdown()}
            </div>

            {/* Dynamic Dark Mode Toggle */}
            <button
              onClick={toggleDarkMode}
              className="p-2 rounded-full text-gray-500 hover:text-emerald-600 dark:text-gray-400 dark:hover:text-zari-accent hover:bg-gray-100 dark:hover:bg-gray-800 transition-all focus:outline-none"
              aria-label="Toggle Dark Mode"
            >
              {isDarkMode ? <Sun size={20} className="animate-spin-slow" /> : <Moon size={20} className="animate-pulse" />}
            </button>
            
            {/* Login Option with Tooltip */}
            <div className="relative group/login flex items-center">
              <button className="text-sm font-semibold text-gray-700 dark:text-gray-300 hover:text-emerald-600 dark:hover:text-zari-accent transition-colors px-3 py-2">
                Log In
              </button>
              {/* Tooltip */}
              <div className="absolute top-full right-1/2 translate-x-1/2 mt-2 w-52 p-3 bg-gray-900 dark:bg-[#112417] border border-gray-700 dark:border-emerald-800/50 text-gray-100 dark:text-gray-300 text-xs rounded-xl shadow-2xl opacity-0 group-hover/login:opacity-100 transition-opacity duration-300 pointer-events-none z-50 text-center leading-relaxed">
                Log in to securely save and track your crop diagnosis history and field logs.
                {/* Tooltip Arrow */}
                <div className="absolute -top-1.5 left-1/2 -translate-x-1/2 w-3 h-3 bg-gray-900 dark:bg-[#112417] border-t border-l border-gray-700 dark:border-emerald-800/50 rotate-45"></div>
              </div>
            </div>

            {/* Primary Action Button */}
            <Link 
              href="#diagnostics" 
              className="group flex items-center gap-2 px-5 py-2.5 rounded-full bg-emerald-600 hover:bg-emerald-700 dark:bg-zari-accent dark:text-zari-bg dark:hover:bg-emerald-400 text-white font-semibold text-sm transition-all transform hover:scale-105 shadow-md"
            >
              Start Diagnosis
              <ArrowRight size={16} className="group-hover:translate-x-1 transition-transform" />
            </Link>
          </div>
        </div>

        {/* Mobile Menu & Actions */}
        <div className="flex items-center gap-3 md:hidden">
          
          <div className="relative">
            <button
              onClick={() => setIsWeatherOpen(!isWeatherOpen)}
              className="p-2 rounded-full hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors focus:outline-none"
            >
              {currentWeatherIcon}
            </button>
            
            {isWeatherOpen && (
              <div className="absolute top-full right-0 mt-3">
                {renderWeatherDropdown()}
              </div>
            )}
          </div>

          <button
            onClick={toggleDarkMode}
            className="p-2 rounded-full text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors focus:outline-none"
          >
            {isDarkMode ? <Sun size={20} /> : <Moon size={20} />}
          </button>
          
          <button 
            onClick={toggleMenu}
            className="text-gray-700 dark:text-gray-300 hover:text-emerald-600 dark:hover:text-zari-accent transition-colors focus:outline-none"
          >
            {isMobileMenuOpen ? <X size={28} /> : <Menu size={28} />}
          </button>
        </div>
      </div>

      {/* Mobile Navigation Dropdown */}
      {isMobileMenuOpen && (
        <div className="md:hidden bg-white dark:bg-zari-bg border-b border-gray-200 dark:border-gray-800 overflow-hidden shadow-lg animate-in slide-in-from-top-2 duration-200">
          <div className="flex flex-col px-6 py-4 space-y-4">
            {navLinks.map((link) => (
              <Link 
                key={link.name} 
                href={link.href} 
                onClick={() => setIsMobileMenuOpen(false)}
                className="text-base font-semibold text-gray-700 dark:text-gray-300 hover:text-emerald-600 dark:hover:text-zari-accent transition-colors block border-b border-gray-100 dark:border-gray-800/50 pb-2"
              >
                {link.name}
              </Link>
            ))}
            
            {/* Mobile Login Option */}
            <div className="flex flex-col pt-2 border-b border-gray-100 dark:border-gray-800/50 pb-4">
              <button className="text-left text-base font-semibold text-emerald-700 dark:text-emerald-400">
                Log In
              </button>
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                Log in to securely save and track your crop diagnosis history and field logs.
              </p>
            </div>

            <Link 
              href="#diagnostics" 
              onClick={() => setIsMobileMenuOpen(false)}
              className="flex justify-center items-center gap-2 px-5 py-3 rounded-xl bg-emerald-600 dark:bg-zari-accent dark:text-zari-bg text-white font-semibold text-base mt-2 shadow-sm"
            >
              Start Diagnosis
              <ArrowRight size={18} />
            </Link>
          </div>
        </div>
      )}
    </header>
  );
}
