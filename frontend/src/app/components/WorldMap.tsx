"use client";

import dynamic from "next/dynamic";

// Dynamically import to avoid SSR issues
const ComposableMap = dynamic(() => import("react-simple-maps").then(m => m.ComposableMap), { ssr: false }) as any;
const Geographies = dynamic(() => import("react-simple-maps").then(m => m.Geographies), { ssr: false }) as any;
const Geography = dynamic(() => import("react-simple-maps").then(m => m.Geography), { ssr: false }) as any;

// Reliable public GeoJSON world topology source
const GEO_URL = "https://cdn.jsdelivr.net/npm/world-atlas@2/countries-110m.json";

// ISO numeric code for Pakistan is 586
const PAKISTAN_NUMERIC = "586";

export default function WorldMap() {
  return (
    <div className="absolute top-0 left-0 w-full h-[620px] pointer-events-none overflow-hidden z-0 opacity-90">
      <ComposableMap
        projectionConfig={{ scale: 145, center: [20, 20] }}
        style={{ width: "100%", height: "100%" }}
      >
        <Geographies geography={GEO_URL}>
          {({ geographies }: { geographies: any[] }) =>
            geographies.map((geo: any) => {
              const isPakistan = geo.id === PAKISTAN_NUMERIC;
              return (
                <Geography
                  key={geo.rsmKey}
                  geography={geo}
                  style={{
                    default: {
                      fill: isPakistan ? "#6EE7B7" : "#D1D5DB",
                      stroke: "#FFFFFF",
                      strokeWidth: 0.5,
                      outline: "none",
                    },
                    hover: {
                      fill: isPakistan ? "#6EE7B7" : "#D1D5DB",
                      stroke: "#FFFFFF",
                      strokeWidth: 0.5,
                      outline: "none",
                    },
                    pressed: {
                      fill: isPakistan ? "#6EE7B7" : "#D1D5DB",
                      outline: "none",
                    },
                  }}
                />
              );
            })
          }
        </Geographies>
      </ComposableMap>
    </div>
  );
}
