/** Load an inline <svg> element as a decoded image. */
export async function svgToImage(svg: SVGSVGElement): Promise<HTMLImageElement> {
  const markup = new XMLSerializer().serializeToString(svg);
  const blobUrl = URL.createObjectURL(new Blob([markup], { type: "image/svg+xml" }));
  try {
    const image = new Image();
    image.src = blobUrl;
    await image.decode();
    return image;
  } finally {
    URL.revokeObjectURL(blobUrl);
  }
}

/** Draw an inline <svg> onto a square canvas. */
export async function svgToCanvas(svg: SVGSVGElement, size = 600): Promise<HTMLCanvasElement> {
  const image = await svgToImage(svg);
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  canvas.getContext("2d")?.drawImage(image, 0, 0, size, size);
  return canvas;
}

/** Rasterize an inline <svg> to a PNG file, e.g. to upload the synthetic sample. */
export async function rasterizeSvg(svg: SVGSVGElement, name = "synthetic_sample.png"): Promise<File> {
  const canvas = await svgToCanvas(svg);
  const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/png"));
  if (!blob) throw new Error("Could not create the sample image.");
  return new File([blob], name, { type: "image/png" });
}
