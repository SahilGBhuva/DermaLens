"use client";

import { Ref, useId } from "react";

export type ScopeMode = "original" | "attention" | "contour";

type Props = {
  mode?: ScopeMode;
  brightness?: number;
  contrast?: number;
  blur?: number;
  className?: string;
  svgRef?: Ref<SVGSVGElement>;
};

const BLOB =
  "M200 92 C252 88 300 118 312 162 C326 208 300 236 318 270 C332 300 290 330 246 320 C214 314 196 334 162 324 C118 312 92 280 96 238 C100 204 78 180 96 148 C116 110 156 96 200 92 Z";

const GLOBULES: [number, number, number][] = [
  [140, 180, 5], [128, 236, 4], [170, 300, 4.5], [270, 292, 4],
  [296, 196, 5], [250, 118, 3.5], [186, 114, 4], [300, 252, 3],
];

/**
 * A synthetic dermoscopy lesion drawn entirely in SVG. It is an illustration,
 * not a real lesion, and exists so the marketing surfaces never need to borrow
 * patient imagery.
 */
export default function LesionScope({
  mode = "original",
  brightness = 1,
  contrast = 1,
  blur = 0,
  className,
  svgRef,
}: Props) {
  const uid = useId().replace(/[^a-zA-Z0-9]/g, "");
  const id = (name: string) => `${name}${uid}`;
  const url = (name: string) => `url(#${id(name)})`;

  return (
    <div className={`lesionScope ${className ?? ""}`}>
      <svg
        ref={svgRef}
        xmlns="http://www.w3.org/2000/svg"
        viewBox="0 0 400 400"
        width="100%"
        height="100%"
        aria-hidden="true"
        style={{ filter: `brightness(${brightness}) contrast(${contrast}) blur(${blur}px)` }}
      >
        <defs>
          <radialGradient id={id("skin")} cx="44%" cy="40%" r="78%">
            <stop offset="0" stopColor="#F2D2B8" />
            <stop offset=".55" stopColor="#E0AE90" />
            <stop offset="1" stopColor="#B67D60" />
          </radialGradient>
          <radialGradient id={id("pig")} cx="54%" cy="54%" r="62%">
            <stop offset="0" stopColor="#4A2C1D" />
            <stop offset=".55" stopColor="#6E4430" />
            <stop offset=".85" stopColor="#946046" />
            <stop offset="1" stopColor="#A8735A" />
          </radialGradient>
          <radialGradient id={id("core")} cx="55%" cy="55%" r="60%">
            <stop offset="0" stopColor="#24130B" />
            <stop offset=".7" stopColor="#3B2216" />
            <stop offset="1" stopColor="#5A3625" stopOpacity="0" />
          </radialGradient>
          <radialGradient id={id("heatA")}>
            <stop offset="0" stopColor="#FF3B2F" stopOpacity=".95" />
            <stop offset=".3" stopColor="#FF9F1A" stopOpacity=".85" />
            <stop offset=".55" stopColor="#FFE45C" stopOpacity=".55" />
            <stop offset=".8" stopColor="#37B6FF" stopOpacity=".28" />
            <stop offset="1" stopColor="#2462E0" stopOpacity="0" />
          </radialGradient>
          <radialGradient id={id("heatB")}>
            <stop offset="0" stopColor="#FF9F1A" stopOpacity=".8" />
            <stop offset=".5" stopColor="#FFE45C" stopOpacity=".4" />
            <stop offset="1" stopColor="#37B6FF" stopOpacity="0" />
          </radialGradient>
          <radialGradient id={id("vig")}>
            <stop offset=".7" stopColor="#05080F" stopOpacity="0" />
            <stop offset=".93" stopColor="#05080F" stopOpacity=".55" />
            <stop offset="1" stopColor="#05080F" stopOpacity=".95" />
          </radialGradient>
          <pattern id={id("pores")} width="9" height="9" patternUnits="userSpaceOnUse">
            <circle cx="2" cy="3" r=".8" fill="#8A5A45" opacity=".28" />
            <circle cx="6.5" cy="7" r=".6" fill="#8A5A45" opacity=".2" />
          </pattern>
          <pattern
            id={id("net")}
            width="13"
            height="13"
            patternUnits="userSpaceOnUse"
            patternTransform="rotate(18)"
          >
            <path d="M0 6.5 L6.5 0 L13 6.5 L6.5 13 Z" fill="none" stroke="#1E0F08" strokeWidth="1.6" />
          </pattern>
          <clipPath id={id("clip")}>
            <path d={BLOB} />
          </clipPath>
          {/* Fractal displacement gives the lesion an organic, irregular border. */}
          <filter id={id("organic")} x="-20%" y="-20%" width="140%" height="140%">
            <feTurbulence type="fractalNoise" baseFrequency="0.022" numOctaves="2" seed="7" result="noise" />
            <feDisplacementMap
              in="SourceGraphic"
              in2="noise"
              scale="18"
              xChannelSelector="R"
              yChannelSelector="G"
              result="warped"
            />
            <feGaussianBlur in="warped" stdDeviation="1.6" />
          </filter>
          <filter id={id("grain")} x="0" y="0" width="100%" height="100%">
            <feTurbulence type="fractalNoise" baseFrequency=".85" numOctaves="2" stitchTiles="stitch" />
            <feColorMatrix values="0 0 0 0 .36  0 0 0 0 .2  0 0 0 0 .12  .9 0 0 0 -.3" />
          </filter>
          <filter id={id("halo")} x="-30%" y="-30%" width="160%" height="160%">
            <feGaussianBlur stdDeviation="12" />
          </filter>
          <filter id={id("veil")} x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="9" />
          </filter>
        </defs>

        <rect width="400" height="400" fill={url("skin")} />
        <rect width="400" height="400" fill={url("pores")} />
        <rect width="400" height="400" fill="#FFFFFF" filter={url("grain")} opacity=".55" />
        <path
          d={BLOB}
          fill="#A06A50"
          opacity=".55"
          transform="translate(205 210) scale(1.12) translate(-205 -210)"
          filter={url("halo")}
        />
        <g filter={url("organic")}>
          <path d={BLOB} fill={url("pig")} />
          <rect width="400" height="400" fill={url("net")} opacity=".38" clipPath={url("clip")} />
          <path
            d="M204 132 C246 128 276 158 272 200 C268 240 244 268 206 266 C170 264 140 240 144 202 C148 164 168 136 204 132 Z"
            fill={url("core")}
          />
          <path
            d="M222 176 C246 176 256 198 248 220 C240 240 214 246 198 232 C184 220 190 196 204 184 C210 178 216 176 222 176 Z"
            fill="#1A0C06"
            opacity=".85"
          />
          {GLOBULES.map(([cx, cy, r]) => (
            <circle key={`${cx}-${cy}`} cx={cx} cy={cy} r={r} fill="#2A1810" />
          ))}
        </g>
        <ellipse cx="176" cy="264" rx="30" ry="20" fill="#8FA3BF" opacity=".26" filter={url("veil")} />
        <g fill="none" stroke="#3A2518">
          <path d="M20 300 C90 280 150 330 230 380" opacity=".5" strokeWidth="1.3" />
          <path d="M330 20 C300 90 330 150 400 190" opacity=".45" strokeWidth="1.1" />
          <path d="M0 120 C60 110 90 60 120 0" opacity=".4" strokeWidth="1" />
        </g>

        <g className="scopeLayer" opacity={mode === "attention" ? 1 : 0}>
          <rect width="400" height="400" fill="#0B2A66" opacity=".42" />
          <circle cx="236" cy="214" r="128" fill={url("heatA")} />
          <circle cx="156" cy="174" r="74" fill={url("heatB")} />
          <g fill="none" stroke="#FFFFFF" strokeWidth="1" strokeDasharray="3 4">
            <path
              opacity=".45"
              d="M236 142 C284 146 304 190 296 226 C288 266 246 286 212 278 C176 270 166 232 176 200 C186 166 206 140 236 142 Z"
            />
            <path
              opacity=".6"
              d="M234 178 C258 180 268 202 262 222 C256 242 232 250 216 242 C200 234 198 212 206 198 C214 184 222 178 234 178 Z"
            />
          </g>
        </g>

        <g className="scopeLayer" opacity={mode === "contour" ? 1 : 0}>
          <rect width="400" height="400" fill="#07101F" opacity=".25" />
          <path d={BLOB} fill="none" stroke="#8CB8FF" strokeWidth="2" strokeDasharray="7 5" />
          <g transform="rotate(28 207 210)" stroke="#FFFFFF" strokeWidth="1" strokeDasharray="2 4" opacity=".75">
            <line x1="57" y1="210" x2="357" y2="210" />
            <line x1="207" y1="70" x2="207" y2="350" />
          </g>
          <circle cx="207" cy="210" r="5" fill="none" stroke="#8CB8FF" strokeWidth="1.5" />
          <path
            d="M78 88 L78 76 L90 76 M322 76 L334 76 L334 88 M334 324 L334 336 L322 336 M90 336 L78 336 L78 324"
            fill="none"
            stroke="#8CB8FF"
            strokeWidth="2"
          />
          <g fill="#DCE8FF" fontFamily="monospace" fontSize="11">
            <text x="336" y="140">AXIS 1</text>
            <text x="98" y="92">AXIS 2</text>
          </g>
        </g>

        <rect width="400" height="400" fill={url("vig")} />
      </svg>
    </div>
  );
}
