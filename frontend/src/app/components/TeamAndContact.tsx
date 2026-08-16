"use client";

import { Mail, Phone, MapPin, Users } from "lucide-react";

export default function TeamAndContact() {
  const team = [
    { name: "Umair Amjad Khan", role: "AI & ML Engineer", email: "umair@zari.ai" },
    { name: "Muhammad Hammaz Azam", role: "Software Engineer", email: "hammaz@zari.ai" },
    { name: "Muhammad Uzair", role: "Agricultural Specialist", email: "uzair@zari.ai" }
  ];

  return (
    <section id="contact" className="w-full bg-emerald-900 dark:bg-[#07130a] py-20 border-t border-emerald-800 dark:border-[#0f2113]">
      <div className="max-w-7xl mx-auto px-6">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-5xl font-extrabold text-white tracking-tight mb-4 flex items-center justify-center gap-3">
            <Users className="w-10 h-10 text-emerald-400" /> The Team Behind ZARI
          </h2>
          <p className="text-emerald-100/80 text-lg max-w-2xl mx-auto">
            We are dedicated to revolutionizing Pakistan's agriculture through cutting-edge Artificial Intelligence technology.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-12 items-center">
          {/* Team Members */}
          <div className="space-y-6">
            <h3 className="text-2xl font-bold text-white mb-6 border-b border-emerald-800/50 pb-4">Meet the Team</h3>
            <div className="flex flex-col gap-4">
              {team.map((member) => (
                <div key={member.name} className="bg-white/10 dark:bg-white/5 border border-emerald-700/50 dark:border-emerald-800/30 p-5 rounded-2xl flex items-center justify-between hover:bg-white/20 transition-colors">
                  <div className="flex flex-col gap-0.5">
                    <span className="text-lg font-bold text-white">{member.name}</span>
                    <a href={`mailto:${member.email}`} className="text-sm text-gray-300 hover:text-emerald-400 transition-colors flex items-center gap-1.5 mt-1">
                      <Mail className="w-3.5 h-3.5" /> {member.email}
                    </a>
                  </div>
                  <span className="text-emerald-300 text-sm font-semibold px-3 py-1 bg-emerald-950/50 rounded-full text-center max-w-[140px] leading-tight">{member.role}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Contact Info */}
          <div className="bg-white dark:bg-[#0B141A] rounded-3xl p-8 md:p-10 shadow-2xl border border-gray-200 dark:border-gray-800">
            <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-8">Contact Us</h3>
            
            <div className="space-y-6">
              <div className="flex items-center gap-4 group">
                <div className="w-12 h-12 bg-emerald-100 dark:bg-emerald-900/30 text-emerald-600 dark:text-emerald-400 rounded-full flex items-center justify-center group-hover:bg-emerald-600 group-hover:text-white transition-colors">
                  <Mail className="w-5 h-5" />
                </div>
                <div>
                  <p className="text-sm font-semibold text-gray-500 dark:text-gray-400">Email</p>
                  <p className="text-gray-900 dark:text-white font-bold">contact@zari.ai</p>
                </div>
              </div>

              <div className="flex items-center gap-4 group">
                <div className="w-12 h-12 bg-emerald-100 dark:bg-emerald-900/30 text-emerald-600 dark:text-emerald-400 rounded-full flex items-center justify-center group-hover:bg-emerald-600 group-hover:text-white transition-colors">
                  <Phone className="w-5 h-5" />
                </div>
                <div>
                  <p className="text-sm font-semibold text-gray-500 dark:text-gray-400">Phone & WhatsApp</p>
                  <p className="text-gray-900 dark:text-white font-bold">+92 317 0478541</p>
                </div>
              </div>

              <div className="flex items-center gap-4 group">
                <div className="w-12 h-12 bg-emerald-100 dark:bg-emerald-900/30 text-emerald-600 dark:text-emerald-400 rounded-full flex items-center justify-center group-hover:bg-emerald-600 group-hover:text-white transition-colors">
                  <MapPin className="w-5 h-5" />
                </div>
                <div>
                  <p className="text-sm font-semibold text-gray-500 dark:text-gray-400">Location</p>
                  <p className="text-gray-900 dark:text-white font-bold">Ghulam Ishaq Khan Institute (GIKI), Pakistan</p>
                </div>
              </div>
            </div>
            
            <a href="mailto:contact@zari.ai" className="mt-10 w-full inline-block text-center bg-emerald-600 hover:bg-emerald-700 text-white font-bold py-3.5 rounded-xl transition-transform hover:-translate-y-0.5 shadow-md">
              Send us an Email
            </a>
          </div>
        </div>
      </div>
    </section>
  );
}
