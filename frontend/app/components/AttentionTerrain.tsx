"use client";

import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import LesionScope from "./LesionScope";
import { svgToCanvas } from "../lib/rasterize";

/*
 * The synthetic lesion raised into a 3D surface: the higher the ground, the
 * more that region contributed to the (illustrative) top score. The attention
 * field mirrors the two hot spots drawn by LesionScope's attention layer.
 */

type Surface = "skin" | "attention";

const SIZE = 4; // world units across the disc's bounding square
const SEGMENTS = 180;
const RADIUS = SIZE / 2 - 0.04;
const MAX_HEIGHT = 1.15;

const HEAT_STOPS = [0, 0.3, 0.55, 0.8, 1];
const HEAT_COLORS = ["#2462E0", "#37B6FF", "#FFE45C", "#FF9F1A", "#FF3B2F"].map(
  (hex) => new THREE.Color(hex)
);

function gaussian(x: number, y: number, cx: number, cy: number, sigma: number) {
  return Math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma * sigma));
}

/** Illustrative attention at a point in the lesion's 400 × 400 SVG space. */
export function attentionAt(sx: number, sy: number) {
  const heat =
    gaussian(sx, sy, 236, 214, 52) +
    0.6 * gaussian(sx, sy, 156, 174, 34) +
    0.12 * gaussian(sx, sy, 205, 210, 120);
  return Math.min(heat / 1.05, 1);
}

function heatColor(value: number, out: THREE.Color) {
  for (let i = 1; i < HEAT_STOPS.length; i++) {
    if (value <= HEAT_STOPS[i]) {
      const t = (value - HEAT_STOPS[i - 1]) / (HEAT_STOPS[i] - HEAT_STOPS[i - 1]);
      return out.copy(HEAT_COLORS[i - 1]).lerp(HEAT_COLORS[i], t);
    }
  }
  return out.copy(HEAT_COLORS[HEAT_COLORS.length - 1]);
}

/** World-space plane coordinates (before rotation) → lesion SVG coordinates. */
const toSvg = (x: number, y: number) => [((x + SIZE / 2) / SIZE) * 400, ((SIZE / 2 - y) / SIZE) * 400];

function buildGeometry() {
  const plane = new THREE.PlaneGeometry(SIZE, SIZE, SEGMENTS, SEGMENTS);
  const pos = plane.attributes.position;
  const uv = plane.attributes.uv;
  const heat = new Float32Array(pos.count);
  const colors = new Float32Array(pos.count * 3);
  const outside = new Uint8Array(pos.count);
  const color = new THREE.Color();

  for (let i = 0; i < pos.count; i++) {
    // Snap grid points beyond the rim onto it, so the disc edge is a smooth
    // circle rather than a staircase of square cells.
    const r = Math.hypot(pos.getX(i), pos.getY(i));
    if (r > RADIUS) {
      outside[i] = 1;
      pos.setXY(i, (pos.getX(i) / r) * RADIUS, (pos.getY(i) / r) * RADIUS);
      uv.setXY(i, (pos.getX(i) + SIZE / 2) / SIZE, (pos.getY(i) + SIZE / 2) / SIZE);
    }
    const [sx, sy] = toSvg(pos.getX(i), pos.getY(i));
    const rimFade = Math.min(1, (RADIUS - Math.min(r, RADIUS)) / 0.3);
    heat[i] = attentionAt(sx, sy) * rimFade;
    // Low attention stays a dim blue so the relief reads end to end.
    heatColor(heat[i], color).lerp(new THREE.Color("#0E1B33"), 0.35 * (1 - heat[i]));
    colors.set([color.r, color.g, color.b], i * 3);
  }
  plane.setAttribute("color", new THREE.BufferAttribute(colors, 3));

  // Drop cells that lie wholly beyond the rim; cells straddling it now end on
  // the snapped circle.
  const index = plane.index!;
  const kept: number[] = [];
  for (let t = 0; t < index.count; t += 3) {
    const a = index.getX(t), b = index.getX(t + 1), c = index.getX(t + 2);
    if (!(outside[a] && outside[b] && outside[c])) kept.push(a, b, c);
  }
  plane.setIndex(kept);
  return { geometry: plane, heat };
}

export default function AttentionTerrain() {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const sourceRef = useRef<SVGSVGElement | null>(null);
  const [surface, setSurface] = useState<Surface>("attention");
  const [height, setHeight] = useState(0.7);
  const [readout, setReadout] = useState<{ x: number; y: number; value: number } | null>(null);
  const [failed, setFailed] = useState(false);

  // Live values the render loop reads without re-running the setup effect.
  const live = useRef({ surface, height, resetView: () => {} });
  live.current.surface = surface;
  live.current.height = height;

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    } catch {
      setFailed(true);
      return;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    mount.appendChild(renderer.domElement);
    renderer.domElement.setAttribute("role", "img");
    renderer.domElement.setAttribute(
      "aria-label",
      "3D terrain of a synthetic lesion where height shows illustrative Grad-CAM attention"
    );

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(34, 1, 0.1, 100);
    const home = new THREE.Vector3(0, 3.4, 5.4);
    camera.position.copy(home);

    scene.add(new THREE.HemisphereLight(0xdfe9ff, 0x0c1626, 1.15));
    const sun = new THREE.DirectionalLight(0xffffff, 1.7);
    sun.position.set(3, 5, 2);
    scene.add(sun);

    const { geometry, heat } = buildGeometry();
    const base = new Float32Array(geometry.attributes.position.array);

    const skinMaterial = new THREE.MeshStandardMaterial({ color: 0xd9b49a, roughness: 0.85 });
    const heatMaterial = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.55 });
    const mesh = new THREE.Mesh(geometry, heatMaterial);
    mesh.rotation.x = -Math.PI / 2;
    scene.add(mesh);

    const grid = new THREE.Mesh(
      geometry,
      new THREE.MeshBasicMaterial({ color: 0xffffff, wireframe: true, transparent: true, opacity: 0.06 })
    );
    grid.rotation.x = -Math.PI / 2;
    scene.add(grid);

    const pedestal = new THREE.Mesh(
      new THREE.CylinderGeometry(RADIUS + 0.06, RADIUS + 0.18, 0.16, 96, 1, true),
      new THREE.MeshStandardMaterial({ color: 0x16233a, roughness: 0.7, side: THREE.DoubleSide })
    );
    pedestal.position.y = -0.08;
    scene.add(pedestal);

    const rings = new THREE.Group();
    [RADIUS + 0.35, RADIUS + 0.7].forEach((r, i) => {
      const ring = new THREE.Mesh(
        new THREE.RingGeometry(r, r + 0.008, 160),
        new THREE.MeshBasicMaterial({ color: 0x3a4e6d, transparent: true, opacity: 0.7 - i * 0.3 })
      );
      ring.rotation.x = -Math.PI / 2;
      ring.position.y = -0.16;
      rings.add(ring);
    });
    scene.add(rings);

    // Skin texture comes from the same synthetic lesion drawing used site-wide.
    let disposed = false;
    if (sourceRef.current) {
      svgToCanvas(sourceRef.current, 1024)
        .then((canvas) => {
          if (disposed) return;
          const texture = new THREE.CanvasTexture(canvas);
          texture.colorSpace = THREE.SRGBColorSpace;
          texture.anisotropy = renderer.capabilities.getMaxAnisotropy();
          skinMaterial.map = texture;
          skinMaterial.color.set(0xffffff);
          skinMaterial.needsUpdate = true;
        })
        .catch(() => {});
    }

    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0.25, 0);
    controls.enableDamping = true;
    controls.enableZoom = false; // keep page scrolling predictable
    controls.enablePan = false;
    controls.minPolarAngle = 0.25;
    controls.maxPolarAngle = 1.32;
    controls.autoRotate = !reduceMotion;
    controls.autoRotateSpeed = 0.7;
    controls.addEventListener("start", () => {
      controls.autoRotate = false;
    });
    live.current.resetView = () => {
      camera.position.copy(home);
      controls.target.set(0, 0.25, 0);
      controls.autoRotate = !reduceMotion;
    };

    function resize() {
      const { clientWidth: w, clientHeight: h } = mount!;
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      // On narrow (phone) stages, widen the view so the disc's sides stay in frame.
      camera.zoom = Math.min(1, camera.aspect / 1.35);
      camera.updateProjectionMatrix();
    }
    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(mount);
    resize();

    // Hover readout: attention value under the pointer.
    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    let pointerEvent: PointerEvent | null = null;
    const onPointerMove = (event: PointerEvent) => {
      pointerEvent = event;
    };
    const onPointerLeave = () => {
      pointerEvent = null;
      setReadout(null);
    };
    renderer.domElement.addEventListener("pointermove", onPointerMove);
    renderer.domElement.addEventListener("pointerleave", onPointerLeave);

    // Only render while on screen.
    let visible = false;
    const visibility = new IntersectionObserver(([entry]) => {
      visible = entry.isIntersecting;
    });
    visibility.observe(mount);

    // Start below zero so the first visible frame always writes heights: the
    // terrain rises in, or with reduced motion snaps straight to its height.
    let currentHeight = -1;
    let frame = 0;
    const positions = geometry.attributes.position;

    function tick() {
      frame = requestAnimationFrame(tick);
      if (!visible) return;

      const target = live.current.height;
      if (Math.abs(target - currentHeight) > 0.001) {
        if (currentHeight < 0) currentHeight = reduceMotion ? target : 0;
        else currentHeight += (target - currentHeight) * (reduceMotion ? 1 : 0.08);
        for (let i = 0; i < positions.count; i++) {
          positions.setZ(i, base[i * 3 + 2] + heat[i] * MAX_HEIGHT * currentHeight);
        }
        positions.needsUpdate = true;
        geometry.computeVertexNormals();
        geometry.computeBoundingSphere();
      }

      const wantHeat = live.current.surface === "attention";
      mesh.material = wantHeat ? heatMaterial : skinMaterial;
      grid.visible = wantHeat;

      if (pointerEvent) {
        const rect = renderer.domElement.getBoundingClientRect();
        pointer.set(
          ((pointerEvent.clientX - rect.left) / rect.width) * 2 - 1,
          -((pointerEvent.clientY - rect.top) / rect.height) * 2 + 1
        );
        raycaster.setFromCamera(pointer, camera);
        const hit = raycaster.intersectObject(mesh)[0];
        if (hit?.uv) {
          const value = attentionAt(hit.uv.x * 400, (1 - hit.uv.y) * 400);
          setReadout({
            x: pointerEvent.clientX - rect.left,
            y: pointerEvent.clientY - rect.top,
            value,
          });
        } else {
          setReadout(null);
        }
        pointerEvent = null;
      }

      controls.update();
      renderer.render(scene, camera);
    }
    tick();

    return () => {
      disposed = true;
      cancelAnimationFrame(frame);
      visibility.disconnect();
      resizeObserver.disconnect();
      renderer.domElement.removeEventListener("pointermove", onPointerMove);
      renderer.domElement.removeEventListener("pointerleave", onPointerLeave);
      controls.dispose();
      scene.traverse((object) => {
        if (object instanceof THREE.Mesh) {
          object.geometry.dispose();
          const materials = Array.isArray(object.material) ? object.material : [object.material];
          materials.forEach((material) => {
            (material as THREE.MeshStandardMaterial).map?.dispose();
            material.dispose();
          });
        }
      });
      skinMaterial.map?.dispose();
      skinMaterial.dispose();
      heatMaterial.dispose();
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, []);

  return (
    <div className="terrain">
      <div className="terrainStage" ref={mountRef}>
        {failed && (
          <p className="terrainFallback">
            3D view needs WebGL, which this browser has turned off.
          </p>
        )}
        {readout && (
          <span className="terrainReadout" style={{ left: readout.x, top: readout.y }}>
            Attention <b>{readout.value.toFixed(2)}</b>
          </span>
        )}
        <span className="monoLabel terrainTag">Illustrative · drag to orbit</span>
      </div>

      <div className="terrainControls">
        <div className="terrainSwitch" role="group" aria-label="Surface">
          {(["attention", "skin"] as const).map((value) => (
            <button
              key={value}
              aria-pressed={surface === value}
              className={surface === value ? "active" : ""}
              onClick={() => setSurface(value)}
            >
              {value === "attention" ? "Attention colors" : "Skin"}
            </button>
          ))}
        </div>
        <label className="terrainHeight">
          <span>Relief</span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={height}
            onChange={(event) => setHeight(parseFloat(event.target.value))}
          />
        </label>
        <button className="terrainReset" onClick={() => live.current.resetView()}>
          Reset view
        </button>
      </div>

      {/* Off-screen source for the skin texture. */}
      <div className="offscreen" aria-hidden="true">
        <LesionScope svgRef={sourceRef} />
      </div>
    </div>
  );
}
