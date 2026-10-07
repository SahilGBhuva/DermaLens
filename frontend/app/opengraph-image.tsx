import { ImageResponse } from "next/og";

export const alt = "DermaLens — see what the model sees. Explainable skin-lesion AI research.";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

// Link-preview card shown when the site URL is shared.
export default function OpenGraphImage() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "72px 80px",
          background: "#0C1626",
          color: "#E6EDF7",
          fontFamily: "sans-serif",
        }}
      >
        <div style={{ display: "flex", flexDirection: "column", gap: 28, maxWidth: 620 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 18 }}>
            <div
              style={{
                width: 52,
                height: 52,
                borderRadius: 26,
                border: "4px solid #E6EDF7",
                display: "flex",
                alignItems: "flex-start",
                justifyContent: "flex-end",
                padding: 6,
              }}
            >
              <div style={{ width: 22, height: 22, borderRadius: 11, background: "#5B8CFF" }} />
            </div>
            <div style={{ fontSize: 44, fontWeight: 700, letterSpacing: -1 }}>DermaLens</div>
          </div>
          <div style={{ fontSize: 76, lineHeight: 1.02, fontWeight: 700, letterSpacing: -2, color: "#FFFFFF" }}>
            See what the model sees.
          </div>
          <div style={{ fontSize: 30, lineHeight: 1.35, color: "#AFC0D6" }}>
            Probabilities, Grad-CAM attention and robustness for skin-lesion AI — with honest limits.
          </div>
          <div
            style={{
              display: "flex",
              alignSelf: "flex-start",
              padding: "10px 18px",
              borderRadius: 999,
              border: "2px solid #2A3B56",
              fontSize: 22,
              color: "#8DB0FF",
            }}
          >
            Research prototype · not a diagnosis
          </div>
        </div>

        <div
          style={{
            width: 400,
            height: 400,
            borderRadius: 200,
            border: "10px solid #1F2E45",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            background: "#16233A",
          }}
        >
          <div
            style={{
              width: 320,
              height: 320,
              borderRadius: 160,
              display: "flex",
              backgroundImage:
                "radial-gradient(circle at 58% 50%, #FF3B2F 0%, #FF9F1A 18%, #FFE45C 32%, #37B6FF 52%, #2462E0 70%, #0E1B33 100%)",
            }}
          />
        </div>
      </div>
    ),
    size
  );
}
