import { useParams, Link } from "react-router-dom";
import { useState, useRef, useCallback, useEffect } from "react";
import {
  Upload,
  Sparkles,
  ArrowLeft,
  RotateCcw,
  Download,
  Settings,
  ChevronDown,
  ChevronUp,
  Cloud,
  Cpu,
  Check,
  Share2,
  ShoppingBag,
  Wand2,
  Palette,
} from "lucide-react";
import { products } from "../data/products";

type EngineMode = "cloud" | "local";
type Stage = "idle" | "PARSING_BODY" | "DETECTING_GARMENT" | "TPS_WARPING" | "AUTO_PROMPT" | "DIFFUSION_REFINE" | "SUCCESS" | "FAILED";

const STAGE_LABELS: Record<Exclude<Stage, "idle" | "FAILED">, string> = {
  PARSING_BODY: "Phân tích cơ thể",
  DETECTING_GARMENT: "Nhận diện trang phục",
  TPS_WARPING: "Căn chỉnh form dáng",
  AUTO_PROMPT: "Mô tả thử đồ",
  DIFFUSION_REFINE: "Tạo ảnh thử đồ",
  SUCCESS: "Hoàn thiện kết quả",
};
const STAGE_ORDER: Array<keyof typeof STAGE_LABELS> = [
  "PARSING_BODY",
  "DETECTING_GARMENT",
  "TPS_WARPING",
  "AUTO_PROMPT",
  "DIFFUSION_REFINE",
  "SUCCESS",
];

type SliderPreset = { fitScale: number; alpha: number; yOffset: number };

const CATEGORY_SLIDER_PRESETS: Record<string, SliderPreset> = {
  auto: { fitScale: 1.1, alpha: 0.7, yOffset: 0.0 },
  // Upper body
  top: { fitScale: 1.08, alpha: 0.72, yOffset: -0.02 },
  tshirt: { fitScale: 1.1, alpha: 0.7, yOffset: -0.02 },
  hoodie: { fitScale: 1.18, alpha: 0.68, yOffset: -0.01 },
  jacket: { fitScale: 1.2, alpha: 0.66, yOffset: 0.0 },
  outer: { fitScale: 1.22, alpha: 0.64, yOffset: 0.0 },
  // Lower body
  pants: { fitScale: 1.18, alpha: 0.65, yOffset: 0.03 },
  jeans: { fitScale: 1.2, alpha: 0.64, yOffset: 0.03 },
  shorts: { fitScale: 1.12, alpha: 0.68, yOffset: -0.01 },
  skirt: { fitScale: 1.1, alpha: 0.64, yOffset: 0.02 },
  // Full body
  dress: { fitScale: 1.04, alpha: 0.62, yOffset: 0.01 },
  // Accessories
  belt: { fitScale: 1.02, alpha: 0.82, yOffset: 0.0 },
  bag: { fitScale: 1.0, alpha: 0.78, yOffset: 0.0 },
  scarf: { fitScale: 1.04, alpha: 0.76, yOffset: -0.03 },
  hat: { fitScale: 1.0, alpha: 0.8, yOffset: -0.04 },
  sunglasses: { fitScale: 1.0, alpha: 0.85, yOffset: 0.0 },
  shoes: { fitScale: 1.04, alpha: 0.72, yOffset: 0.02 },
  boots: { fitScale: 1.08, alpha: 0.7, yOffset: 0.01 },
  generic: { fitScale: 1.1, alpha: 0.7, yOffset: 0.0 },
};

type DiffusionPreset = { steps: number; guidance: number; texture: number };

const CATEGORY_DIFFUSION_PRESETS: Record<string, DiffusionPreset> = {
  auto: { steps: 22, guidance: 5.0, texture: 0.72 },
  // Upper
  top: { steps: 22, guidance: 5.0, texture: 0.72 },
  tshirt: { steps: 22, guidance: 5.0, texture: 0.7 },
  hoodie: { steps: 24, guidance: 5.2, texture: 0.66 },
  jacket: { steps: 24, guidance: 5.3, texture: 0.64 },
  outer: { steps: 26, guidance: 5.4, texture: 0.62 },
  // Lower
  pants: { steps: 24, guidance: 4.8, texture: 0.66 },
  jeans: { steps: 24, guidance: 4.6, texture: 0.64 },
  shorts: { steps: 22, guidance: 4.8, texture: 0.68 },
  skirt: { steps: 24, guidance: 4.9, texture: 0.62 },
  // Full body
  dress: { steps: 26, guidance: 5.0, texture: 0.6 },
  // Accessories
  belt: { steps: 18, guidance: 5.2, texture: 0.82 },
  bag: { steps: 20, guidance: 5.1, texture: 0.78 },
  scarf: { steps: 20, guidance: 5.2, texture: 0.74 },
  hat: { steps: 18, guidance: 5.3, texture: 0.8 },
  sunglasses: { steps: 18, guidance: 5.4, texture: 0.84 },
  shoes: { steps: 22, guidance: 5.0, texture: 0.7 },
  boots: { steps: 24, guidance: 5.0, texture: 0.68 },
  generic: { steps: 22, guidance: 5.0, texture: 0.72 },
};

function clamp(v: number, lo: number, hi: number) {
  return Math.min(hi, Math.max(lo, v));
}

function adjustPresetWithGemini(
  preset: SliderPreset,
  gemini: Record<string, unknown> | null | undefined,
): SliderPreset {
  if (!gemini) return preset;
  const fit = String(gemini.fit ?? "").toLowerCase();
  const silhouette = String(gemini.silhouette ?? "").toLowerCase();
  const length = String(gemini.length ?? "").toLowerCase();

  let { fitScale, alpha, yOffset } = preset;

  if (fit === "oversized" || fit === "loose") {
    fitScale += 0.04;
    alpha -= 0.02;
  } else if (fit === "slim") {
    fitScale -= 0.02;
    alpha += 0.02;
  }

  if (["wide_leg", "oversized"].includes(silhouette)) {
    fitScale += 0.04;
    alpha -= 0.02;
  } else if (["a_line", "fit_and_flare"].includes(silhouette)) {
    alpha -= 0.02;
  } else if (["skinny", "fitted", "bodycon", "sheath", "shift"].includes(silhouette)) {
    fitScale -= 0.03;
    alpha += 0.02;
  }

  if (length === "ankle" || length === "maxi") {
    yOffset += 0.02;
  } else if (length === "cropped" || length === "mini" || length === "hip") {
    yOffset -= 0.02;
  }

  return {
    fitScale: clamp(fitScale, 0.95, 1.3),
    alpha: clamp(alpha, 0.55, 0.86),
    yOffset: clamp(yOffset, -0.06, 0.06),
  };
}

function adjustDiffusionPresetWithGemini(
  preset: DiffusionPreset,
  gemini: Record<string, unknown> | null | undefined,
): DiffusionPreset {
  if (!gemini) return preset;
  const fit = String(gemini.fit ?? "").toLowerCase();
  const silhouette = String(gemini.silhouette ?? "").toLowerCase();
  const length = String(gemini.length ?? "").toLowerCase();

  let { steps, guidance, texture } = preset;

  if (["wide_leg", "a_line", "fit_and_flare", "oversized"].includes(silhouette)) {
    steps += 2;
    texture -= 0.04;
  } else if (["skinny", "fitted", "bodycon", "sheath", "shift"].includes(silhouette)) {
    steps = Math.max(22, steps);
    texture += 0.02;
  }

  if (fit === "oversized" || fit === "loose") {
    texture -= 0.02;
    guidance -= 0.1;
  } else if (fit === "slim") {
    texture += 0.02;
  }

  if (length === "maxi" || length === "ankle") {
    steps += 1;
  }

  return {
    steps: Math.round(clamp(steps, 16, 30)),
    guidance: clamp(guidance, 3.8, 6.0),
    texture: clamp(texture, 0.5, 0.88),
  };
}

export default function TryOn() {
  const { id } = useParams();
  const product = id ? products.find((p) => p.id === Number(id)) : null;

  const [personImage, setPersonImage] = useState<string | null>(null);
  const [clothImage, setClothImage] = useState<string | null>(product?.image ?? null);
  const [result, setResult] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string>("");
  const [backendBadge, setBackendBadge] = useState<string>("");
  const [stage, setStage] = useState<Stage>("idle");
  const [progress, setProgress] = useState(0);
  const [showSettings, setShowSettings] = useState(false);
  const [engineMode, setEngineMode] = useState<EngineMode>("cloud");

  // Pipeline settings
  const [fitScale, setFitScale] = useState(1.08);
  const [alpha, setAlpha] = useState(0.78);
  const [yOffset, setYOffset] = useState(-0.01);
  const [autoPresetByCategory, setAutoPresetByCategory] = useState(true);
  const [stylePrompt, setStylePrompt] = useState("");
  const [genSteps, setGenSteps] = useState(24);
  const [genGuidance, setGenGuidance] = useState(5.2);
  const [preserveStrength, setPreserveStrength] = useState(0.82);
  const [qualityPreset, setQualityPreset] = useState<"fast" | "balanced" | "hq">("hq");
  const [refinerMode, setRefinerMode] = useState("dpm++");
  const [clothType, setClothType] = useState("auto");
  const [useGeminiPrompt, setUseGeminiPrompt] = useState(true);
  const [promptMode, setPromptMode] = useState<"auto" | "fallback" | "manual">("auto");

  // Outfit recommendation
  type RecommendedItem = { type: string; name: string; reason: string };
  type Recommendation = {
    main_color: string;
    garment_type: string;
    suitable_colors: string[];
    avoid_colors: string[];
    recommended_items: RecommendedItem[];
    style_summary: string;
    short_tip: string;
    source?: string;
  };
  const [recommendation, setRecommendation] = useState<Recommendation | null>(null);
  const [recommendLoading, setRecommendLoading] = useState(false);
  const [recommendError, setRecommendError] = useState<string | null>(null);
  const [occasion, setOccasion] = useState("casual");
  const [styleVibe, setStyleVibe] = useState("minimal");

  const personRef = useRef<HTMLInputElement>(null);
  const clothRef = useRef<HTMLInputElement>(null);
  const personFileRef = useRef<File | null>(null);
  const clothFileRef = useRef<File | null>(null);
  const progressRef = useRef(0);

  const advanceStage = useCallback((nextStage: Stage, nextProgress: number, force = false) => {
    if (!force && nextProgress < progressRef.current) return;
    progressRef.current = nextProgress;
    setStage(nextStage);
    setProgress(nextProgress);
  }, []);

  const handleFile = useCallback(
    (setter: (v: string) => void, fileRef: React.MutableRefObject<File | null>) =>
      (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;
        fileRef.current = file;
        const reader = new FileReader();
        reader.onload = () => setter(reader.result as string);
        reader.readAsDataURL(file);
      },
    []
  );

  const toBlob = async (src: string): Promise<Blob> => {
    if (src.startsWith("data:")) {
      const [header, data] = src.split(",");
      const mime = header.match(/:(.*?);/)?.[1] ?? "image/png";
      const bin = atob(data);
      const arr = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
      return new Blob([arr], { type: mime });
    }
    const resp = await fetch(src);
    return resp.blob();
  };

  // Simulated stage progression for the first three (passive) stages.
  // AUTO_PROMPT and DIFFUSION_REFINE are driven by real API calls.
  useEffect(() => {
    if (!loading) return;
    advanceStage("PARSING_BODY", 8, true);
    const seq: Array<{ s: Stage; p: number; delay: number }> = [
      { s: "PARSING_BODY", p: 18, delay: 800 },
      { s: "DETECTING_GARMENT", p: 32, delay: 1800 },
      { s: "TPS_WARPING", p: 46, delay: 2800 },
    ];
    const timers = seq.map((step) =>
      setTimeout(() => {
        advanceStage(step.s, step.p);
      }, step.delay)
    );
    return () => timers.forEach(clearTimeout);
  }, [advanceStage, loading]);

  const handleTryOn = async () => {
    if (!personImage || !clothImage) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setBackendBadge("");
    setRecommendation(null);
    setRecommendError(null);
    advanceStage("PARSING_BODY", 5, true);

    try {
      // Reusable file blobs for describe + main call
      const personBlob = personFileRef.current
        ? personFileRef.current
        : await toBlob(personImage);
      const clothBlob = clothFileRef.current
        ? clothFileRef.current
        : await toBlob(clothImage);
      const personName = personFileRef.current?.name ?? "person.png";
      const clothName = clothFileRef.current?.name ?? "cloth.png";

      // ─── Step "Mô tả thử đồ" — call Gemini to autofill prompt ───
      let effectiveStylePrompt = stylePrompt;
      let geminiDescribeOk = false;
      let effectiveFitScale = fitScale;
      let effectiveAlpha = alpha;
      let effectiveYOffset = yOffset;
      let effectiveSteps = genSteps;
      let effectiveGuidance = genGuidance;
      let effectivePreserve = preserveStrength;
      if (promptMode === "auto" && useGeminiPrompt) {
        advanceStage("AUTO_PROMPT", 58);
        try {
          const describeForm = new FormData();
          describeForm.append("person", personBlob, personName);
          describeForm.append("cloth", clothBlob, clothName);
          describeForm.append("category_lock", clothType);
          describeForm.append("user_prompt", stylePrompt);
          const dResp = await fetch("/api/tryon/describe", {
            method: "POST",
            body: describeForm,
            signal: AbortSignal.timeout(90_000),
          });
          if (dResp.ok) {
            const dJson = await dResp.json();
            const auto = (
              dJson.positive_prompt ||
              dJson.positivePrompt ||
              dJson.prompt ||
              ""
            ).trim();
            if (auto) {
              effectiveStylePrompt = stylePrompt.trim()
                ? `${auto}, ${stylePrompt.trim()}`
                : auto;
              setStylePrompt(effectiveStylePrompt);
              geminiDescribeOk = true;
            }
            // Gemini-adjust: refine the category preset with fit/silhouette/length
            // hints from the JSON envelope so straight jeans / skinny jeans /
            // wide-leg jeans get distinct slider values, etc.
            if (autoPresetByCategory) {
              const geminiCategory = String(dJson.category ?? "").toLowerCase();
              const lookupKey =
                geminiCategory && CATEGORY_SLIDER_PRESETS[geminiCategory]
                  ? geminiCategory
                  : clothType;
              const base =
                CATEGORY_SLIDER_PRESETS[lookupKey] ?? CATEGORY_SLIDER_PRESETS.generic;
              const tuned = adjustPresetWithGemini(base, dJson);
              effectiveFitScale = tuned.fitScale;
              effectiveAlpha = tuned.alpha;
              effectiveYOffset = tuned.yOffset;
              setFitScale(tuned.fitScale);
              setAlpha(tuned.alpha);
              setYOffset(tuned.yOffset);
              const dbase =
                CATEGORY_DIFFUSION_PRESETS[lookupKey] ?? CATEGORY_DIFFUSION_PRESETS.generic;
              const dtuned = adjustDiffusionPresetWithGemini(dbase, dJson);
              effectiveSteps = dtuned.steps;
              effectiveGuidance = dtuned.guidance;
              effectivePreserve = dtuned.texture;
              setGenSteps(dtuned.steps);
              setGenGuidance(dtuned.guidance);
              setPreserveStrength(dtuned.texture);
            }
          } else {
            try {
              const dJson = await dResp.json();
              console.warn("Gemini describe skipped:", dJson.error || dResp.status);
            } catch {
              console.warn("Gemini describe skipped:", dResp.status);
            }
          }
          // describe failures are non-fatal — fall through to main call
        } catch (describeErr) {
          console.warn("Gemini describe failed:", describeErr);
          // ignore — main pipeline will still run
        }
        advanceStage("AUTO_PROMPT", 70);
      }

      const formData = new FormData();
      formData.append("person", personBlob, personName);
      formData.append("cloth", clothBlob, clothName);

      formData.append("fit_scale", effectiveFitScale.toString());
      formData.append("alpha", effectiveAlpha.toString());
      formData.append("y_offset", effectiveYOffset.toString());
      formData.append("use_gen", "true");
      formData.append("style_prompt", effectiveStylePrompt);
      formData.append("gen_steps", effectiveSteps.toString());
      formData.append("gen_guidance", effectiveGuidance.toString());
      formData.append("preserve_strength", effectivePreserve.toString());
      formData.append("quality_preset", qualityPreset);
      formData.append("refiner_mode", refinerMode);
      formData.append("cloth_type", clothType);
      formData.append("use_catvton_cloud", (engineMode === "cloud").toString());
      // Only skip server-side Gemini when the describe endpoint actually worked.
      formData.append("use_gemini_prompt", (promptMode === "auto" && useGeminiPrompt && !geminiDescribeOk).toString());
      formData.append("prompt_mode", promptMode);

      advanceStage("DIFFUSION_REFINE", 85);
      const resp = await fetch("/api/tryon", {
        method: "POST",
        body: formData,
        signal: AbortSignal.timeout(1_200_000),
      });

      if (!resp.ok) {
        let errMsg: string;
        try {
          const json = await resp.json();
          errMsg = json.error || `Server error ${resp.status}`;
        } catch {
          errMsg = (await resp.text()) || `Server error ${resp.status}`;
        }
        throw new Error(errMsg);
      }

      const pipelineInfo = resp.headers.get("X-Pipeline-Info") ?? "";
      const backend = resp.headers.get("X-Backend") ?? "";
      const warning = resp.headers.get("X-Warning") ?? "";
      setBackendBadge(backend);
      setInfo([pipelineInfo && `Pipeline: ${pipelineInfo}`, warning].filter(Boolean).join("\n"));

      const blob = await resp.blob();
      setResult(URL.createObjectURL(blob));
      advanceStage("SUCCESS", 100);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Lỗi khi thử đồ");
      advanceStage("FAILED", progressRef.current, true);
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setPersonImage(null);
    setClothImage(product?.image ?? null);
    setResult(null);
    setError(null);
    setInfo("");
    setBackendBadge("");
    setRecommendation(null);
    setRecommendError(null);
    advanceStage("idle", 0, true);
    personFileRef.current = null;
    clothFileRef.current = null;
  };

  const handleRecommend = async () => {
    if (!result) return;
    setRecommendLoading(true);
    setRecommendError(null);
    try {
      const blob = await toBlob(result);
      const form = new FormData();
      form.append("result", blob, "result.png");
      form.append("category", clothType);
      form.append("occasion", occasion);
      form.append("style", styleVibe);
      const resp = await fetch("/api/tryon/recommend", {
        method: "POST",
        body: form,
        signal: AbortSignal.timeout(120_000),
      });
      const data = await resp.json();
      if (data?.success && data.recommendation) {
        setRecommendation(data.recommendation as Recommendation);
      } else {
        setRecommendError(data?.error || "Không lấy được gợi ý phối đồ");
      }
    } catch (err) {
      setRecommendError(err instanceof Error ? err.message : "Lỗi gợi ý phối đồ");
    } finally {
      setRecommendLoading(false);
    }
  };

  return (
    <div className="bg-[var(--color-bg-warm)] min-h-screen">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Link
          to="/"
          className="inline-flex items-center gap-1 text-sm text-[var(--color-ink-muted)] hover:text-[var(--color-ink)] mb-6"
        >
          <ArrowLeft className="w-4 h-4" /> Quay lại cửa hàng
        </Link>

        {/* Header */}
        <div className="text-center mb-10">
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-white border border-[var(--color-accent)]/40 text-[11px] font-semibold uppercase tracking-wider">
            <Sparkles className="w-3 h-3 text-[var(--color-accent)]" /> AI Virtual Try-On
          </span>
          <h1 className="font-display text-4xl md:text-5xl text-[var(--color-ink)] mt-4">
            Thử đồ ảo thông minh
          </h1>
          <p className="text-[var(--color-ink-muted)] max-w-xl mx-auto mt-3">
            Tải ảnh của bạn, chọn trang phục yêu thích — AI sẽ tạo ảnh thử đồ chân thực
            chỉ trong vài giây.
          </p>
        </div>

        {/* Engine Selector */}
        <div className="max-w-md mx-auto mb-8">
          <div className="bg-white rounded-2xl p-1.5 border border-[var(--color-line)] flex gap-1">
            <button
              onClick={() => setEngineMode("cloud")}
              className={`flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl text-sm font-medium transition-all ${
                engineMode === "cloud"
                  ? "bg-[var(--color-primary)] text-white"
                  : "text-[var(--color-ink-muted)] hover:bg-[var(--color-bg-soft)]"
              }`}
            >
              <Cloud className="w-4 h-4" /> Cloud AI
            </button>
            <button
              onClick={() => setEngineMode("local")}
              className={`flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl text-sm font-medium transition-all ${
                engineMode === "local"
                  ? "bg-[var(--color-primary)] text-white"
                  : "text-[var(--color-ink-muted)] hover:bg-[var(--color-bg-soft)]"
              }`}
            >
              <Cpu className="w-4 h-4" /> Local SD
            </button>
          </div>
          <p className="text-xs text-[var(--color-ink-muted)] mt-2 text-center px-4">
            {engineMode === "cloud"
              ? "Cloud: Kwai-Kolors / CatVTON / IDM-VTON — chất lượng cao, 30-90s"
              : "Local SD-Inpaint — chạy offline, 15-60s, phù hợp test"}
          </p>
        </div>

        {/* 3-Column Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
          {/* 1. Person upload */}
          <div className="card p-5">
            <div className="flex items-center gap-2 mb-4">
              <span className="w-7 h-7 rounded-full bg-[var(--color-bg-warm)] flex items-center justify-center text-xs font-bold text-[var(--color-ink)] border border-[var(--color-accent)]/30">
                1
              </span>
              <h3 className="font-semibold text-[var(--color-ink)]">Ảnh của bạn</h3>
            </div>
            <input
              ref={personRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={handleFile(setPersonImage, personFileRef)}
            />
            {personImage ? (
              <div className="relative group">
                <img
                  src={personImage}
                  alt="Person"
                  className="w-full h-[26rem] object-contain bg-[var(--color-bg-soft)] rounded-xl"
                />
                <button
                  onClick={() => {
                    setPersonImage(null);
                    if (personRef.current) personRef.current.value = "";
                  }}
                  className="absolute top-2 right-2 p-2 bg-white/95 rounded-full opacity-0 group-hover:opacity-100 shadow-sm transition-opacity"
                >
                  <RotateCcw className="w-4 h-4" />
                </button>
              </div>
            ) : (
              <button
                onClick={() => personRef.current?.click()}
                className="w-full h-[26rem] border-2 border-dashed border-[var(--color-line-strong)] rounded-xl flex flex-col items-center justify-center gap-3 hover:border-[var(--color-accent)] hover:bg-[var(--color-bg-warm)] transition-all"
              >
                <div className="w-14 h-14 rounded-full bg-[var(--color-bg-warm)] flex items-center justify-center">
                  <Upload className="w-6 h-6 text-[var(--color-accent)]" />
                </div>
                <span className="text-sm font-medium text-[var(--color-ink)]">
                  Tải ảnh lên hoặc kéo thả
                </span>
                <span className="text-xs text-[var(--color-ink-muted)] text-center px-6">
                  Nên dùng ảnh nửa thân trên, đứng thẳng, ánh sáng tốt
                </span>
              </button>
            )}
          </div>

          {/* 2. Garment selection */}
          <div className="card p-5">
            <div className="flex items-center gap-2 mb-4">
              <span className="w-7 h-7 rounded-full bg-[var(--color-bg-warm)] flex items-center justify-center text-xs font-bold text-[var(--color-ink)] border border-[var(--color-accent)]/30">
                2
              </span>
              <h3 className="font-semibold text-[var(--color-ink)]">Trang phục</h3>
            </div>
            <input
              ref={clothRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={handleFile(setClothImage, clothFileRef)}
            />
            {clothImage ? (
              <div className="relative group">
                <img
                  src={clothImage}
                  alt="Cloth"
                  className="w-full h-[26rem] object-contain bg-[var(--color-bg-soft)] rounded-xl"
                />
                <button
                  onClick={() => {
                    setClothImage(null);
                    if (clothRef.current) clothRef.current.value = "";
                  }}
                  className="absolute top-2 right-2 p-2 bg-white/95 rounded-full opacity-0 group-hover:opacity-100 shadow-sm transition-opacity"
                >
                  <RotateCcw className="w-4 h-4" />
                </button>
              </div>
            ) : (
              <button
                onClick={() => clothRef.current?.click()}
                className="w-full h-[26rem] border-2 border-dashed border-[var(--color-line-strong)] rounded-xl flex flex-col items-center justify-center gap-3 hover:border-[var(--color-accent)] hover:bg-[var(--color-bg-warm)] transition-all"
              >
                <div className="w-14 h-14 rounded-full bg-[var(--color-bg-warm)] flex items-center justify-center">
                  <Upload className="w-6 h-6 text-[var(--color-accent)]" />
                </div>
                <span className="text-sm font-medium text-[var(--color-ink)]">
                  Tải ảnh trang phục
                </span>
                <span className="text-xs text-[var(--color-ink-muted)] text-center px-6">
                  Ảnh phẳng, nền trắng cho kết quả tốt nhất
                </span>
              </button>
            )}
            {product && (
              <p className="text-xs text-[var(--color-accent)] mt-2 text-center font-medium">
                Sản phẩm: {product.name}
              </p>
            )}
          </div>

          {/* 3. Result */}
          <div className="card p-5">
            <div className="flex items-center justify-between gap-2 mb-4">
              <div className="flex items-center gap-2">
                <span className="w-7 h-7 rounded-full bg-[var(--color-bg-warm)] flex items-center justify-center text-xs font-bold text-[var(--color-ink)] border border-[var(--color-accent)]/30">
                  3
                </span>
                <h3 className="font-semibold text-[var(--color-ink)]">Kết quả</h3>
              </div>
              {result && (
                <a
                  href={result}
                  download={`tryon_${Date.now()}.png`}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-[var(--color-ink)] text-white text-xs font-semibold hover:opacity-90 transition-opacity"
                >
                  <Download className="w-3.5 h-3.5" /> Tải ảnh
                </a>
              )}
            </div>
            {result ? (
              <div className="relative group">
                <img
                  src={result}
                  alt="Try-on result"
                  className="w-full h-[26rem] object-contain bg-[var(--color-bg-soft)] rounded-xl"
                />
                <div className="absolute top-2 right-2 flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                  <button className="p-2 bg-white/95 rounded-full shadow-sm hover:scale-110 transition-transform" title="Chia sẻ">
                    <Share2 className="w-4 h-4" />
                  </button>
                </div>
                {backendBadge && (
                  <span className="absolute bottom-2 left-2 inline-flex items-center gap-1 px-2.5 py-1 bg-white/95 rounded-full text-[11px] font-medium text-[var(--color-ink)]">
                    {backendBadge.toLowerCase().includes("local") ? (
                      <Cpu className="w-3 h-3" />
                    ) : (
                      <Cloud className="w-3 h-3" />
                    )}
                    {backendBadge}
                  </span>
                )}
              </div>
            ) : (
              <div className="w-full h-[26rem] bg-[var(--color-bg-soft)] rounded-xl flex items-center justify-center p-6">
                {loading ? (
                  <AILoading stage={stage} progress={progress} engineMode={engineMode} />
                ) : (
                  <div className="text-center">
                    <Sparkles className="w-10 h-10 text-[var(--color-ink-disabled)] mx-auto mb-3" />
                    <p className="text-sm text-[var(--color-ink-muted)]">
                      Kết quả thử đồ sẽ hiển thị ở đây
                    </p>
                  </div>
                )}
              </div>
            )}

            {result && (
              <div className="flex gap-2 mt-3">
                <Link
                  to={product ? `/product/${product.id}` : "/"}
                  className="flex-1 btn-secondary"
                >
                  <ShoppingBag className="w-4 h-4" /> Thêm vào giỏ
                </Link>
                <button onClick={reset} className="btn-ghost border border-[var(--color-line)]">
                  <RotateCcw className="w-4 h-4" /> Thử lại
                </button>
              </div>
            )}

            <p className="mt-3 text-[11px] leading-relaxed text-[var(--color-ink-muted)] flex items-start gap-1.5">
              <span aria-hidden>🔒</span>
              <span>
                Ảnh thử đồ <b>không được lưu</b> trên hệ thống. Hãy bấm "Tải ảnh" nếu bạn muốn giữ lại — kết quả sẽ mất khi bạn rời trang.
              </span>
            </p>
          </div>
        </div>

        {/* Action bar */}
        <div className="flex flex-wrap justify-center gap-3 mt-8">
          <button
            onClick={handleTryOn}
            disabled={!personImage || !clothImage || loading}
            className="btn-tryon"
          >
            <Sparkles className="w-5 h-5" />
            {loading ? "Đang xử lý..." : "Thử đồ ngay"}
          </button>
          <button
            onClick={() => setShowSettings((v) => !v)}
            className="btn-secondary"
          >
            <Settings className="w-4 h-4" />
            Cài đặt nâng cao
            {showSettings ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>
          <button onClick={reset} className="btn-ghost border border-[var(--color-line)]">
            <RotateCcw className="w-4 h-4" /> Làm mới
          </button>
        </div>

        {/* Settings panel */}
        {showSettings && (
          <div className="max-w-4xl mx-auto mt-6 card p-6 space-y-6 animate-fade-up">
            <h3 className="text-lg font-semibold flex items-center gap-2">
              <Settings className="w-5 h-5 text-[var(--color-accent)]" />
              Tinh chỉnh pipeline
            </h3>

            {/* Basic fitting */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <RangeField label={`Độ rộng (${fitScale.toFixed(2)})`} min={0.8} max={1.5} step={0.01} value={fitScale} onChange={setFitScale} />
              <RangeField label={`Độ hòa trộn (${alpha.toFixed(2)})`} min={0.4} max={1.0} step={0.01} value={alpha} onChange={setAlpha} />
              <RangeField label={`Dịch dọc (${yOffset.toFixed(2)})`} min={-0.15} max={0.2} step={0.01} value={yOffset} onChange={setYOffset} />
            </div>
            <label className="flex items-center gap-2 mt-2 text-sm text-[var(--color-ink)] cursor-pointer select-none">
              <input
                type="checkbox"
                checked={autoPresetByCategory}
                onChange={(e) => {
                  const enabled = e.target.checked;
                  setAutoPresetByCategory(enabled);
                  if (enabled) {
                    const preset =
                      CATEGORY_SLIDER_PRESETS[clothType] ?? CATEGORY_SLIDER_PRESETS.generic;
                    setFitScale(preset.fitScale);
                    setAlpha(preset.alpha);
                    setYOffset(preset.yOffset);
                    const dpreset =
                      CATEGORY_DIFFUSION_PRESETS[clothType] ?? CATEGORY_DIFFUSION_PRESETS.generic;
                    setGenSteps(dpreset.steps);
                    setGenGuidance(dpreset.guidance);
                    setPreserveStrength(dpreset.texture);
                  }
                }}
              />
              Tự chỉnh thông số theo loại trang phục
            </label>

            {/* Quality preset */}
            <div>
              <span className="text-sm font-medium text-[var(--color-ink)] mb-2 block">
                Chất lượng
              </span>
              <div className="flex gap-2">
                {(["fast", "balanced", "hq"] as const).map((p) => (
                  <button
                    key={p}
                    onClick={() => setQualityPreset(p)}
                    className={qualityPreset === p ? "chip-active" : "chip"}
                  >
                    {p === "fast" ? "Nhanh" : p === "balanced" ? "Cân bằng" : "Chất lượng cao"}
                  </button>
                ))}
              </div>
            </div>

            {/* Prompt */}
            <label className="block">
              <span className="text-sm font-medium text-[var(--color-ink)] mb-1.5 block">
                Mô tả trang phục (prompt)
              </span>
              <input
                type="text"
                value={stylePrompt}
                onChange={(e) => setStylePrompt(e.target.value)}
                placeholder="Để trống để giữ nguyên áo gốc"
                className="input"
              />
            </label>

            {/* Diffusion params */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 p-4 bg-[var(--color-bg-warm)] rounded-xl border border-[var(--color-accent)]/20">
              <RangeField label={`Steps (${genSteps})`} min={4} max={30} step={1} value={genSteps} onChange={(v) => setGenSteps(v)} />
              <RangeField label={`Guidance (${genGuidance.toFixed(1)})`} min={0.5} max={8.0} step={0.1} value={genGuidance} onChange={setGenGuidance} />
              <RangeField label={`Giữ texture (${preserveStrength.toFixed(2)})`} min={0.25} max={1.0} step={0.01} value={preserveStrength} onChange={setPreserveStrength} />
            </div>

            {/* Dropdowns */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <label className="block">
                <span className="text-sm font-medium text-[var(--color-ink)] mb-1.5 block">
                  Refiner mode
                </span>
                <select value={refinerMode} onChange={(e) => setRefinerMode(e.target.value)} className="select">
                  <option value="lcm">LCM (nhanh, VRAM thấp)</option>
                  <option value="hypersd">HyperSD</option>
                  <option value="dpm++">DPM++ (chi tiết cao)</option>
                  <option value="euler">Euler</option>
                  <option value="base">Base</option>
                </select>
              </label>
              <label className="block">
                <span className="text-sm font-medium text-[var(--color-ink)] mb-1.5 block">
                  Khóa loại trang phục
                </span>
                <select
                  value={clothType}
                  onChange={(e) => {
                    const next = e.target.value;
                    setClothType(next);
                    if (autoPresetByCategory) {
                      const preset =
                        CATEGORY_SLIDER_PRESETS[next] ?? CATEGORY_SLIDER_PRESETS.generic;
                      setFitScale(preset.fitScale);
                      setAlpha(preset.alpha);
                      setYOffset(preset.yOffset);
                      const dpreset =
                        CATEGORY_DIFFUSION_PRESETS[next] ?? CATEGORY_DIFFUSION_PRESETS.generic;
                      setGenSteps(dpreset.steps);
                      setGenGuidance(dpreset.guidance);
                      setPreserveStrength(dpreset.texture);
                    }
                  }}
                  className="select"
                >
                  <optgroup label="Tự động">
                    <option value="auto">Tự động (hệ thống tự nhận)</option>
                  </optgroup>
                  <optgroup label="Áo">
                    <option value="top">Áo (chung)</option>
                    <option value="tshirt">T-shirt</option>
                    <option value="hoodie">Hoodie</option>
                    <option value="jacket">Jacket</option>
                    <option value="outer">Áo khoác ngoài</option>
                  </optgroup>
                  <optgroup label="Quần / Váy">
                    <option value="pants">Quần (chung)</option>
                    <option value="jeans">Jeans</option>
                    <option value="shorts">Quần short</option>
                    <option value="dress">Váy liền</option>
                    <option value="skirt">Chân váy</option>
                  </optgroup>
                  <optgroup label="Phụ kiện">
                    <option value="belt">Thắt lưng</option>
                    <option value="bag">Túi</option>
                    <option value="scarf">Khăn</option>
                    <option value="hat">Mũ / nón</option>
                    <option value="sunglasses">Kính</option>
                    <option value="shoes">Giày</option>
                    <option value="boots">Boots</option>
                  </optgroup>
                  <option value="generic">Khác</option>
                </select>
                <span className="text-[11px] text-[var(--color-ink-muted)] mt-1 block">
                  Auto = hệ thống tự nhận. Chọn cụ thể để khóa pipeline, tránh nhận nhầm áo thành váy.
                </span>
              </label>
            </div>

            <div>
              <span className="text-sm font-medium text-[var(--color-ink)] mb-2 block">
                Chế độ prompt
              </span>
              <div className="flex flex-wrap gap-2">
                {([
                  { v: "auto", label: "Gemini Auto", hint: "AI Vision phân tích ảnh & sinh prompt" },
                  { v: "fallback", label: "Theo loại", hint: "Dùng prompt mẫu theo danh mục, không gọi Gemini" },
                  { v: "manual", label: "Thủ công", hint: "Chỉ dùng đúng prompt bạn nhập bên dưới" },
                ] as const).map((opt) => (
                  <button
                    key={opt.v}
                    type="button"
                    onClick={() => setPromptMode(opt.v)}
                    className={promptMode === opt.v ? "chip-active" : "chip"}
                    title={opt.hint}
                  >
                    {opt.label}
                  </button>
                ))}
              </div>
              <p className="text-[11px] text-[var(--color-ink-muted)] mt-1.5">
                {promptMode === "auto" && "Gemini Vision phân tích người + áo, tự sinh positive/negative prompt. Khi Gemini lỗi (503/quota) hệ thống tự rơi về 'Theo loại'."}
                {promptMode === "fallback" && "Không gọi Gemini. Dùng prompt mẫu theo loại trang phục đã khóa — nhanh, ổn định, miễn phí."}
                {promptMode === "manual" && "Chỉ dùng prompt bạn nhập ở ô 'Mô tả trang phục'. Bỏ qua mọi auto prompt."}
              </p>
            </div>

            {promptMode === "auto" && (
              <label className="flex items-start gap-2 cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={useGeminiPrompt}
                  onChange={(e) => setUseGeminiPrompt(e.target.checked)}
                  className="mt-1"
                />
                <span className="text-sm">
                  <span className="font-medium text-[var(--color-ink)]">Bật Gemini Vision</span>
                  <span className="text-[11px] text-[var(--color-ink-muted)] block">
                    Tắt nếu muốn tạm thời không gọi Gemini ở chế độ Auto.
                  </span>
                </span>
              </label>
            )}

            <p className="text-xs text-[var(--color-ink-muted)]">
              💡 Tip: Trang phục có họa tiết mạnh → giữ texture cao và dùng DPM++ để khóa hoa văn.
            </p>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="max-w-2xl mx-auto mt-6 p-4 bg-[var(--color-danger)]/10 border border-[var(--color-danger)]/30 rounded-xl text-[var(--color-danger)] text-sm text-center">
            {error}
          </div>
        )}

        {/* Pipeline info */}
        {info && (
          <div className="max-w-2xl mx-auto mt-6 p-4 bg-white border border-[var(--color-line)] rounded-xl">
            <pre className="whitespace-pre-wrap font-mono text-xs text-[var(--color-ink-muted)]">
              {info}
            </pre>
          </div>
        )}

        {/* Outfit recommendation */}
        {result && (
          <div className="max-w-4xl mx-auto mt-8 card p-6 space-y-5">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <h3 className="text-lg font-semibold flex items-center gap-2 text-[var(--color-ink)]">
                  <Wand2 className="w-5 h-5 text-[var(--color-accent)]" />
                  Stylist AI — gợi ý phối đồ
                </h3>
                <p className="text-xs text-[var(--color-ink-muted)] mt-1">
                  Gemini phân tích ảnh kết quả và đề xuất màu phối, item nên mặc cùng và phong cách phù hợp.
                </p>
              </div>
              <button
                onClick={handleRecommend}
                disabled={recommendLoading}
                className="btn-secondary"
              >
                <Palette className="w-4 h-4" />
                {recommendLoading
                  ? "Đang phân tích..."
                  : recommendation
                  ? "Gợi ý lại"
                  : "Gợi ý phối đồ"}
              </button>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <label className="block">
                <span className="text-xs font-medium text-[var(--color-ink)] mb-1.5 block">
                  Dịp sử dụng
                </span>
                <select
                  value={occasion}
                  onChange={(e) => setOccasion(e.target.value)}
                  className="select"
                >
                  <option value="casual">Hằng ngày / casual</option>
                  <option value="đi học">Đi học</option>
                  <option value="đi làm / công sở">Đi làm / công sở</option>
                  <option value="đi chơi">Đi chơi</option>
                  <option value="hẹn hò">Hẹn hò</option>
                  <option value="dự tiệc">Dự tiệc</option>
                  <option value="du lịch">Du lịch</option>
                </select>
              </label>
              <label className="block">
                <span className="text-xs font-medium text-[var(--color-ink)] mb-1.5 block">
                  Phong cách
                </span>
                <select
                  value={styleVibe}
                  onChange={(e) => setStyleVibe(e.target.value)}
                  className="select"
                >
                  <option value="minimal">Minimal</option>
                  <option value="nữ tính">Nữ tính</option>
                  <option value="streetwear">Streetwear</option>
                  <option value="công sở">Công sở</option>
                  <option value="sang trọng">Sang trọng</option>
                  <option value="trẻ trung">Trẻ trung</option>
                  <option value="Hàn Quốc">Hàn Quốc</option>
                  <option value="vintage">Vintage</option>
                </select>
              </label>
            </div>

            {recommendError && (
              <div className="p-3 bg-[var(--color-danger)]/10 border border-[var(--color-danger)]/30 rounded-xl text-[var(--color-danger)] text-sm">
                {recommendError}
              </div>
            )}

            {recommendation && (
              <div className="space-y-4">
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                  <InfoTile label="Màu chính" value={recommendation.main_color || "—"} />
                  <InfoTile label="Loại trang phục" value={recommendation.garment_type || "—"} />
                  <InfoTile
                    label="Nguồn"
                    value={recommendation.source === "gemini" ? "Gemini Vision" : recommendation.source === "cache" ? "Cache" : "Mẫu theo loại"}
                  />
                </div>

                {recommendation.suitable_colors?.length > 0 && (
                  <div>
                    <span className="text-xs font-semibold text-[var(--color-ink-muted)] uppercase tracking-wider">
                      Nên phối với
                    </span>
                    <div className="flex flex-wrap gap-2 mt-2">
                      {recommendation.suitable_colors.map((c) => (
                        <span key={c} className="chip">{c}</span>
                      ))}
                    </div>
                  </div>
                )}

                {recommendation.avoid_colors?.length > 0 && (
                  <div>
                    <span className="text-xs font-semibold text-[var(--color-ink-muted)] uppercase tracking-wider">
                      Nên tránh
                    </span>
                    <div className="flex flex-wrap gap-2 mt-2">
                      {recommendation.avoid_colors.map((c) => (
                        <span
                          key={c}
                          className="px-3 py-1 rounded-full text-xs border border-[var(--color-danger)]/30 text-[var(--color-danger)] bg-[var(--color-danger)]/5"
                        >
                          {c}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {recommendation.recommended_items?.length > 0 && (
                  <div>
                    <span className="text-xs font-semibold text-[var(--color-ink-muted)] uppercase tracking-wider">
                      Nên mặc cùng
                    </span>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 mt-2">
                      {recommendation.recommended_items.map((item, idx) => (
                        <div
                          key={`${item.type}-${idx}`}
                          className="p-3 rounded-xl bg-[var(--color-bg-warm)] border border-[var(--color-line)]"
                        >
                          <p className="text-sm font-semibold text-[var(--color-ink)]">
                            <span className="uppercase text-[10px] tracking-wider text-[var(--color-accent)] mr-1.5">
                              {item.type}
                            </span>
                            {item.name}
                          </p>
                          <p className="text-xs text-[var(--color-ink-muted)] mt-1 leading-relaxed">
                            {item.reason}
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {recommendation.style_summary && (
                  <p className="text-sm text-[var(--color-ink)] leading-relaxed">
                    {recommendation.style_summary}
                  </p>
                )}

                {recommendation.short_tip && (
                  <div className="p-3 rounded-xl bg-[var(--color-accent)]/10 border border-[var(--color-accent)]/30 text-sm text-[var(--color-ink)]">
                    <span aria-hidden className="mr-1.5">💡</span>
                    {recommendation.short_tip}
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Quick product picker */}
        {!id && (
          <div className="mt-12">
            <h3 className="font-display text-2xl text-[var(--color-ink)] mb-2 text-center">
              Hoặc chọn nhanh từ cửa hàng
            </h3>
            <p className="text-sm text-[var(--color-ink-muted)] text-center mb-6">
              Click vào sản phẩm để dùng làm trang phục thử
            </p>
            <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-3">
              {products.map((p) => (
                <button
                  key={p.id}
                  onClick={() => setClothImage(p.image)}
                  className={`group rounded-xl overflow-hidden border-2 transition-all hover:-translate-y-1 ${
                    clothImage === p.image
                      ? "border-[var(--color-accent)]"
                      : "border-[var(--color-line)]"
                  }`}
                >
                  <img src={p.image} alt={p.name} className="w-full h-24 object-cover" />
                  <p className="text-xs text-[var(--color-ink)] p-1.5 truncate bg-white">
                    {p.name}
                  </p>
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

/* ---------------------------------------------------------------- */

function InfoTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="p-3 rounded-xl bg-[var(--color-bg-soft)] border border-[var(--color-line)]">
      <span className="text-[10px] uppercase tracking-wider text-[var(--color-ink-muted)] font-semibold">
        {label}
      </span>
      <p className="text-sm font-medium text-[var(--color-ink)] mt-1">{value}</p>
    </div>
  );
}

function RangeField({
  label,
  min,
  max,
  step,
  value,
  onChange,
}: {
  label: string;
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <label className="block">
      <span className="text-sm font-medium text-[var(--color-ink)] mb-1.5 block">{label}</span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full"
      />
      <div className="flex justify-between text-[11px] text-[var(--color-ink-muted)] mt-0.5">
        <span>{min}</span>
        <span>{max}</span>
      </div>
    </label>
  );
}

function AILoading({
  stage,
  progress,
  engineMode,
}: {
  stage: Stage;
  progress: number;
  engineMode: EngineMode;
}) {
  const currentIdx = stage === "idle" ? -1 : STAGE_ORDER.indexOf(stage as keyof typeof STAGE_LABELS);

  return (
    <div className="w-full max-w-xs">
      <div className="text-center mb-4">
        <Sparkles className="w-8 h-8 text-[var(--color-accent)] mx-auto mb-2 animate-pulse" />
        <p className="font-medium text-[var(--color-ink)]">AI đang xử lý ảnh của bạn</p>
        <p className="text-xs text-[var(--color-ink-muted)] mt-1">
          {engineMode === "cloud" ? "Cloud: 30-90s" : "Local: 15-60s"}
        </p>
      </div>

      <div className="ai-progress-bar mb-4">
        <div className="ai-progress-fill" style={{ width: `${progress}%` }} />
      </div>
      <p className="text-center text-xs font-semibold text-[var(--color-ink)] mb-4">
        {progress}%
      </p>

      <ul className="space-y-2 text-sm">
        {STAGE_ORDER.map((s, i) => {
          const done = currentIdx > i;
          const active = currentIdx === i;
          return (
            <li
              key={s}
              className={`flex items-center gap-2 ${
                done || active ? "text-[var(--color-ink)]" : "text-[var(--color-ink-disabled)]"
              }`}
            >
              <span className="w-4 h-4 flex items-center justify-center">
                {done ? (
                  <Check className="w-4 h-4 text-[var(--color-success)]" />
                ) : active ? (
                  <span className="w-2 h-2 bg-[var(--color-accent)] rounded-full animate-pulse" />
                ) : (
                  <span className="w-2 h-2 border border-[var(--color-line-strong)] rounded-full" />
                )}
              </span>
              <span className="text-xs">{STAGE_LABELS[s]}</span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
