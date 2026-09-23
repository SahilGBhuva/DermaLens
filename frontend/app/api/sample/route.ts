export const runtime = "nodejs";

const SAMPLE_URL =
  "https://isic-archive.s3.amazonaws.com/images/ISIC_0016128.jpg";

export async function GET() {
  try {
    const response = await fetch(SAMPLE_URL, {
      cache: "force-cache",
    });

    if (!response.ok) {
      return Response.json(
        { detail: "Sample image is temporarily unavailable." },
        { status: 502 }
      );
    }

    const image = await response.arrayBuffer();
    return new Response(image, {
      headers: {
        "Content-Type": response.headers.get("content-type") ?? "image/jpeg",
        "Cache-Control": "public, max-age=86400, s-maxage=86400",
      },
    });
  } catch {
    return Response.json(
      { detail: "Sample image is temporarily unavailable." },
      { status: 502 }
    );
  }
}
