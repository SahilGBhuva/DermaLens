// Quick photo-quality check, run in the browser before any upload.
//
// The image is shrunk to 256 px on its longer side, converted to grey, and
// three simple measures are taken:
//   brightness - mean grey level (0-255)
//   glare      - share of nearly white pixels (blown-out highlights, flash)
//   sharpness  - variance of the Laplacian (fine detail; low means blurry)
// The thresholds are heuristics (see QUALITY_LIMITS), not measured accuracy.
// They only produce advice; the visitor can always analyse anyway.

export type PhotoQuality = {
  brightness: number;
  glare: number;
  sharpness: number;
  warnings: string[];
};

// Checked 2026-10-21 against the CC0 sample (4288 px and 1200 px copies):
// sharp originals 187-1438, blurred copies down to 5-30; mean grey ~121, a
// 30%-brightness copy 36, a 2.2x copy 210. Sharpness is measured at roughly
// the size the model sees, so blur that vanishes at 224 px is not flagged.
// To do (roadmap year 2): set these from the HAM10000 / PAD-UFES-20 training
// distributions instead of a single image.
export const QUALITY_LIMITS = {
  darkBelow: 55,
  brightAbove: 195,
  glareAbove: 0.03,
  blurBelow: 35,
};

const SIZE = 256;

export function assessGrey(grey: Float32Array, width: number, height: number): PhotoQuality {
  let sum = 0;
  let nearWhite = 0;
  for (let i = 0; i < grey.length; i += 1) {
    sum += grey[i];
    if (grey[i] >= 245) nearWhite += 1;
  }
  const brightness = sum / grey.length;

  // 4-neighbour Laplacian over the interior pixels.
  let lapSum = 0;
  let lapSq = 0;
  let count = 0;
  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      const i = y * width + x;
      const lap = grey[i - 1] + grey[i + 1] + grey[i - width] + grey[i + width] - 4 * grey[i];
      lapSum += lap;
      lapSq += lap * lap;
      count += 1;
    }
  }
  const mean = count ? lapSum / count : 0;
  const sharpness = count ? lapSq / count - mean * mean : 0;
  const glare = nearWhite / grey.length;

  const warnings: string[] = [];
  if (brightness < QUALITY_LIMITS.darkBelow) warnings.push("Looks too dark: retake in brighter, even light.");
  if (brightness > QUALITY_LIMITS.brightAbove) warnings.push("Looks washed out: avoid direct sunlight or flash.");
  else if (glare > QUALITY_LIMITS.glareAbove) warnings.push("Glare or shine on the skin: avoid flash and angle away from lights.");
  // A dark photo also has little visible detail; only call it blurry when the light is fine.
  if (sharpness < QUALITY_LIMITS.blurBelow && brightness >= QUALITY_LIMITS.darkBelow) {
    warnings.push("May be blurry: hold steady and tap to focus on the spot.");
  }

  return { brightness, glare, sharpness, warnings };
}

export async function checkPhotoQuality(file: File): Promise<PhotoQuality | null> {
  try {
    const bitmap = await createImageBitmap(file);
    const scale = SIZE / Math.max(bitmap.width, bitmap.height);
    const width = Math.max(3, Math.round(bitmap.width * scale));
    const height = Math.max(3, Math.round(bitmap.height * scale));
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) return null;
    ctx.drawImage(bitmap, 0, 0, width, height);
    bitmap.close();
    const { data } = ctx.getImageData(0, 0, width, height);
    const grey = new Float32Array(width * height);
    for (let i = 0; i < grey.length; i += 1) {
      grey[i] = 0.299 * data[i * 4] + 0.587 * data[i * 4 + 1] + 0.114 * data[i * 4 + 2];
    }
    return assessGrey(grey, width, height);
  } catch {
    return null; // unreadable here; the API still validates the upload
  }
}
