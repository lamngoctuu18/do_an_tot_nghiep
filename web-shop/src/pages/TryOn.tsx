import { useParams, Link } from "react-router-dom";
import { useState, useRef, useCallback } from "react";
import { Upload, Shirt, Loader2, ArrowLeft, RotateCcw, Download, Settings, ChevronDown, ChevronUp, Cloud, Cpu, Zap } from "lucide-react";
import { products } from "../data/products";

type EngineMode = "cloud" | "local" | "cpu";

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
  const [status, setStatus] = useState<string>("");
  const [showSettings, setShowSettings] = useState(false);

  // Engine mode: cloud (default), local (SD-inpaint), cpu (geometric only)
  const [engineMode, setEngineMode] = useState<EngineMode>("cloud");

  // Pipeline settings (matching Gradio UI)
  const [fitScale, setFitScale] = useState(1.08);
  const [alpha, setAlpha] = useState(0.78);
  const [yOffset, setYOffset] = useState(-0.01);
  const [stylePrompt, setStylePrompt] = useState("");
  const [genSteps, setGenSteps] = useState(20);
  const [genGuidance, setGenGuidance] = useState(2.5);
  const [preserveStrength, setPreserveStrength] = useState(0.60);
  const [qualityPreset, setQualityPreset] = useState<"fast" | "balanced" | "hq">("hq");
  const [refinerMode, setRefinerMode] = useState("dpm++");
  const [clothType, setClothType] = useState("auto");

  const personRef = useRef<HTMLInputElement>(null);
  const clothRef = useRef<HTMLInputElement>(null);

  // Store raw File objects for direct upload
  const personFileRef = useRef<File | null>(null);
  const clothFileRef = useRef<File | null>(null);

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

  /** Convert a data-URL or remote URL to a Blob. */
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

  const handleTryOn = async () => {
    if (!personImage || !clothImage) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setBackendBadge("");

    const useGen = engineMode !== "cpu";
    const useCatvtonCloud = engineMode === "cloud";

    const statusMsg = engineMode === "cloud"
      ? "Sending to Cloud AI (IDM-VTON / CatVTON)..."
      : engineMode === "local"
        ? "Processing with local SD-Inpaint..."
        : "Running CPU geometric pipeline...";
    setStatus(statusMsg);

    try {
      const formData = new FormData();

      // Person image
      if (personFileRef.current) {
        formData.append("person", personFileRef.current);
      } else {
        formData.append("person", await toBlob(personImage), "person.png");
      }

      // Cloth image
      if (clothFileRef.current) {
        formData.append("cloth", clothFileRef.current);
      } else {
        formData.append("cloth", await toBlob(clothImage), "cloth.png");
      }

      // Pipeline params
      formData.append("fit_scale", fitScale.toString());
      formData.append("alpha", alpha.toString());
      formData.append("y_offset", yOffset.toString());
      formData.append("use_gen", useGen.toString());
      formData.append("style_prompt", stylePrompt);
      formData.append("gen_steps", genSteps.toString());
      formData.append("gen_guidance", genGuidance.toString());
      formData.append("preserve_strength", preserveStrength.toString());
      formData.append("quality_preset", qualityPreset);
      formData.append("refiner_mode", refinerMode);
      formData.append("cloth_type", clothType);
      formData.append("use_catvton_cloud", useCatvtonCloud.toString());

      if (engineMode === "cloud") {
        setStatus("Cloud AI processing (30-90s)...");
      }

      const resp = await fetch("/api/tryon", {
        method: "POST",
        body: formData,
        signal: AbortSignal.timeout(600_000), // 10 minutes for GPU diffusion
      });

      if (!resp.ok) {
        let errMsg: string;
        try {
          const json = await resp.json();
          errMsg = json.error || `Server error ${resp.status}`;
        } catch {
          errMsg = await resp.text() || `Server error ${resp.status}`;
        }
        throw new Error(errMsg);
      }

      // Read pipeline info from response headers
      const pipelineInfo = resp.headers.get("X-Pipeline-Info") ?? "";
      const backend = resp.headers.get("X-Backend") ?? "";
      const warning = resp.headers.get("X-Warning") ?? "";
      setBackendBadge(backend);
      setInfo(
        [
          pipelineInfo ? `Pipeline: ${pipelineInfo}` : "",
          warning,
        ]
          .filter(Boolean)
          .join("\n")
      );

      // Result is a PNG image
      const blob = await resp.blob();
      setResult(URL.createObjectURL(blob));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Error during try-on");
    } finally {
      setLoading(false);
      setStatus("");
    }
  };

  const reset = () => {
    setPersonImage(null);
    setClothImage(product?.image ?? null);
    setResult(null);
    setError(null);
    setInfo("");
    setBackendBadge("");
    setStatus("");
    personFileRef.current = null;
    clothFileRef.current = null;
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <Link to="/" className="inline-flex items-center gap-1 text-gray-400 hover:text-white mb-6 transition-colors">
        <ArrowLeft className="w-4 h-4" /> Quay lai cua hang
      </Link>

      <div className="text-center mb-10">
        <h1 className="text-3xl md:text-4xl font-bold text-white mb-3">
          <Shirt className="inline w-8 h-8 text-primary mr-2" />
          Thu do AI
        </h1>
        <p className="text-gray-400 max-w-xl mx-auto">
          Upload anh chan dung cua ban va chon trang phuc muon thu.
          AI se ghep trang phuc len nguoi ban.
        </p>
      </div>

      {/* AI Engine Selector */}
      <div className="max-w-2xl mx-auto mb-8">
        <span className="text-sm text-gray-400 mb-3 block text-center">AI Engine</span>
        <div className="flex gap-3 justify-center">
          <button
            onClick={() => setEngineMode("cloud")}
            className={`flex items-center gap-2 px-5 py-3 rounded-xl text-sm font-medium transition-all ${
              engineMode === "cloud"
                ? "bg-primary text-white shadow-lg shadow-primary/25"
                : "bg-surface-light text-gray-400 hover:text-white"
            }`}
          >
            <Cloud className="w-4 h-4" />
            Cloud AI (Best)
          </button>
          <button
            onClick={() => setEngineMode("local")}
            className={`flex items-center gap-2 px-5 py-3 rounded-xl text-sm font-medium transition-all ${
              engineMode === "local"
                ? "bg-primary text-white shadow-lg shadow-primary/25"
                : "bg-surface-light text-gray-400 hover:text-white"
            }`}
          >
            <Cpu className="w-4 h-4" />
            Local SD
          </button>
          <button
            onClick={() => setEngineMode("cpu")}
            className={`flex items-center gap-2 px-5 py-3 rounded-xl text-sm font-medium transition-all ${
              engineMode === "cpu"
                ? "bg-primary text-white shadow-lg shadow-primary/25"
                : "bg-surface-light text-gray-400 hover:text-white"
            }`}
          >
            <Zap className="w-4 h-4" />
            CPU Only
          </button>
        </div>
        <p className="text-xs text-gray-600 mt-2 text-center">
          {engineMode === "cloud" && "IDM-VTON / CatVTON / Fal.ai — best quality, needs internet (30-90s)"}
          {engineMode === "local" && "Stable Diffusion inpaint — offline, needs GPU 4-6GB (15-60s)"}
          {engineMode === "cpu" && "Geometric warp only — fastest, no GPU needed (5-15s)"}
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Person upload */}
        <div className="bg-surface rounded-2xl p-6 border border-white/5">
          <h3 className="font-semibold text-white mb-4">Anh nguoi mau</h3>
          <input ref={personRef} type="file" accept="image/*" className="hidden" onChange={handleFile(setPersonImage, personFileRef)} />
          {personImage ? (
            <div className="relative group">
              <img
                src={personImage}
                alt="Person"
                className="w-full h-80 sm:h-[30rem] object-contain bg-black/20 rounded-xl"
              />
              <button
                onClick={() => { setPersonImage(null); if (personRef.current) personRef.current.value = ""; }}
                className="absolute top-2 right-2 p-2 bg-black/50 rounded-full opacity-0 group-hover:opacity-100 transition-opacity"
              >
                <RotateCcw className="w-4 h-4" />
              </button>
            </div>
          ) : (
            <button
              onClick={() => personRef.current?.click()}
              className="w-full h-80 sm:h-[30rem] border-2 border-dashed border-white/10 rounded-xl flex flex-col items-center justify-center gap-3 hover:border-primary/30 hover:bg-primary/5 transition-all"
            >
              <Upload className="w-10 h-10 text-gray-500" />
              <span className="text-sm text-gray-400">Click de upload anh</span>
              <span className="text-xs text-gray-600">Nen dung anh nua than tren, nen sang</span>
            </button>
          )}
        </div>

        {/* Cloth upload */}
        <div className="bg-surface rounded-2xl p-6 border border-white/5">
          <h3 className="font-semibold text-white mb-4">Anh trang phuc</h3>
          <input ref={clothRef} type="file" accept="image/*" className="hidden" onChange={handleFile(setClothImage, clothFileRef)} />
          {clothImage ? (
            <div className="relative group">
              <img
                src={clothImage}
                alt="Cloth"
                className="w-full h-80 sm:h-[30rem] object-contain bg-black/20 rounded-xl"
              />
              <button
                onClick={() => { setClothImage(null); if (clothRef.current) clothRef.current.value = ""; }}
                className="absolute top-2 right-2 p-2 bg-black/50 rounded-full opacity-0 group-hover:opacity-100 transition-opacity"
              >
                <RotateCcw className="w-4 h-4" />
              </button>
            </div>
          ) : (
            <button
              onClick={() => clothRef.current?.click()}
              className="w-full h-80 sm:h-[30rem] border-2 border-dashed border-white/10 rounded-xl flex flex-col items-center justify-center gap-3 hover:border-primary/30 hover:bg-primary/5 transition-all"
            >
              <Upload className="w-10 h-10 text-gray-500" />
              <span className="text-sm text-gray-400">Click de upload anh ao</span>
              <span className="text-xs text-gray-600">Nen dung anh phang, nen trang</span>
            </button>
          )}
          {product && (
            <p className="text-xs text-primary mt-2 text-center">
              Dang dung anh tu: {product.name}
            </p>
          )}
        </div>

        {/* Result */}
        <div className="bg-surface rounded-2xl p-6 border border-white/5">
          <h3 className="font-semibold text-white mb-4">Ket qua</h3>
          {result ? (
            <div className="relative group">
              <img
                src={result}
                alt="Try-on result"
                className="w-full h-80 sm:h-[30rem] object-contain bg-black/20 rounded-xl"
              />
              <a
                href={result}
                download="tryon_result.png"
                className="absolute bottom-2 right-2 p-2 bg-primary rounded-full opacity-0 group-hover:opacity-100 transition-opacity"
              >
                <Download className="w-4 h-4" />
              </a>
            </div>
          ) : (
            <div className="w-full h-80 sm:h-[30rem] bg-surface-light rounded-xl flex items-center justify-center">
              {loading ? (
                <div className="text-center">
                  <Loader2 className="w-10 h-10 text-primary animate-spin mx-auto mb-3" />
                  <p className="text-sm text-gray-400">{status || "Dang xu ly AI..."}</p>
                  <p className="text-xs text-gray-600 mt-1">
                    {engineMode === "cloud" ? "Cloud: 30-90s" : engineMode === "local" ? "Local: 15-60s" : "CPU: 5-15s"}
                  </p>
                </div>
              ) : (
                <p className="text-sm text-gray-500">Ket qua se hien o day</p>
              )}
            </div>
          )}
          {/* Backend badge */}
          {result && backendBadge && (
            <div className="text-center mt-3">
              <span className="inline-flex items-center gap-1.5 px-3 py-1 bg-green-500/20 text-green-400 rounded-full text-xs font-medium">
                <Cloud className="w-3 h-3" />
                Generated by: {backendBadge}
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Action buttons */}
      <div className="flex justify-center gap-4 mt-8">
        <button
          onClick={handleTryOn}
          disabled={!personImage || !clothImage || loading}
          className="flex items-center gap-2 px-8 py-4 bg-gradient-to-r from-primary to-primary-dark text-white font-semibold rounded-2xl hover:shadow-lg hover:shadow-primary/25 transition-all disabled:opacity-40 disabled:cursor-not-allowed"
        >
          {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Shirt className="w-5 h-5" />}
          {loading ? "Dang thu do..." : "Thu do ngay"}
        </button>
        <button
          onClick={() => setShowSettings((v) => !v)}
          className="flex items-center gap-2 px-6 py-4 bg-surface-light text-gray-300 hover:text-white rounded-2xl transition-colors"
        >
          <Settings className="w-5 h-5" />
          Cai dat
          {showSettings ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </button>
        <button
          onClick={reset}
          className="flex items-center gap-2 px-6 py-4 bg-surface-light text-gray-300 hover:text-white rounded-2xl transition-colors"
        >
          <RotateCcw className="w-5 h-5" /> Lam moi
        </button>
      </div>

      {/* Settings panel */}
      {showSettings && (
        <div className="max-w-4xl mx-auto mt-6 bg-surface rounded-2xl border border-white/5 p-6 space-y-6">
          <h3 className="text-lg font-semibold text-white flex items-center gap-2">
            <Settings className="w-5 h-5 text-primary" /> Cai dat pipeline
          </h3>

          {/* Row 1: Basic fitting */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <label className="space-y-1">
              <span className="text-sm text-gray-400">Do rong trang phuc ({fitScale.toFixed(2)})</span>
              <input type="range" min={0.8} max={1.5} step={0.01} value={fitScale}
                onChange={(e) => setFitScale(Number(e.target.value))}
                className="w-full accent-primary" />
              <div className="flex justify-between text-xs text-gray-600"><span>0.80</span><span>1.50</span></div>
            </label>
            <label className="space-y-1">
              <span className="text-sm text-gray-400">Do hoa tron ({alpha.toFixed(2)})</span>
              <input type="range" min={0.4} max={1.0} step={0.01} value={alpha}
                onChange={(e) => setAlpha(Number(e.target.value))}
                className="w-full accent-primary" />
              <div className="flex justify-between text-xs text-gray-600"><span>0.40</span><span>1.00</span></div>
            </label>
            <label className="space-y-1">
              <span className="text-sm text-gray-400">Dich doc ({yOffset.toFixed(2)})</span>
              <input type="range" min={-0.15} max={0.2} step={0.01} value={yOffset}
                onChange={(e) => setYOffset(Number(e.target.value))}
                className="w-full accent-primary" />
              <div className="flex justify-between text-xs text-gray-600"><span>-0.15</span><span>0.20</span></div>
            </label>
          </div>

          {/* Row 2: Quality preset */}
          <div>
            <span className="text-sm text-gray-400 mb-2 block">Preset chat luong</span>
            <div className="flex gap-3">
              {(["fast", "balanced", "hq"] as const).map((p) => (
                <button key={p} onClick={() => setQualityPreset(p)}
                  className={`px-4 py-2 rounded-xl text-sm font-medium transition-all ${
                    qualityPreset === p
                      ? "bg-primary text-white"
                      : "bg-surface-light text-gray-400 hover:text-white"
                  }`}>
                  {p === "fast" ? "Nhanh" : p === "balanced" ? "Can bang" : "Chat luong cao"}
                </button>
              ))}
            </div>
          </div>

          {/* Row 3: Prompt */}
          <label className="block space-y-1">
            <span className="text-sm text-gray-400">Mo ta trang phuc (prompt)</span>
            <input type="text" value={stylePrompt} onChange={(e) => setStylePrompt(e.target.value)}
              placeholder="De trong de giu nguyen ao goc"
              className="w-full px-4 py-2.5 bg-surface-light border border-white/10 rounded-xl text-sm text-white placeholder:text-gray-600 focus:outline-none focus:border-primary/50" />
          </label>

          {/* Row 4: Diffusion params (visible when NOT cpu mode) */}
          {engineMode !== "cpu" && (
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 p-4 bg-surface-light/50 rounded-xl border border-white/5">
              <label className="space-y-1">
                <span className="text-sm text-gray-400">Refine steps ({genSteps})</span>
                <input type="range" min={4} max={30} step={1} value={genSteps}
                  onChange={(e) => setGenSteps(Number(e.target.value))}
                  className="w-full accent-primary" />
                <div className="flex justify-between text-xs text-gray-600"><span>4</span><span>30</span></div>
              </label>
              <label className="space-y-1">
                <span className="text-sm text-gray-400">Refine guidance ({genGuidance.toFixed(1)})</span>
                <input type="range" min={0.5} max={8.0} step={0.1} value={genGuidance}
                  onChange={(e) => setGenGuidance(Number(e.target.value))}
                  className="w-full accent-primary" />
                <div className="flex justify-between text-xs text-gray-600"><span>0.5</span><span>8.0</span></div>
              </label>
              <label className="space-y-1">
                <span className="text-sm text-gray-400">Giu texture ao goc ({preserveStrength.toFixed(2)})</span>
                <input type="range" min={0.25} max={1.0} step={0.01} value={preserveStrength}
                  onChange={(e) => setPreserveStrength(Number(e.target.value))}
                  className="w-full accent-primary" />
                <div className="flex justify-between text-xs text-gray-600"><span>0.25</span><span>1.00</span></div>
              </label>
            </div>
          )}

          {/* Row 5: Dropdowns */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <label className="space-y-1">
              <span className="text-sm text-gray-400">Refiner mode</span>
              <select value={refinerMode} onChange={(e) => setRefinerMode(e.target.value)}
                className="w-full px-4 py-2.5 bg-surface-light border border-white/10 rounded-xl text-sm text-white focus:outline-none focus:border-primary/50">
                <option value="lcm">LCM (nhanh, VRAM thap)</option>
                <option value="hypersd">HyperSD</option>
                <option value="dpm++">DPM++ (chi tiet cao)</option>
                <option value="euler">Euler</option>
                <option value="base">Base</option>
              </select>
            </label>
            <label className="space-y-1">
              <span className="text-sm text-gray-400">Loai trang phuc</span>
              <select value={clothType} onChange={(e) => setClothType(e.target.value)}
                className="w-full px-4 py-2.5 bg-surface-light border border-white/10 rounded-xl text-sm text-white focus:outline-none focus:border-primary/50">
                <option value="auto">Tu dong</option>
                <option value="tshirt">T-shirt</option>
                <option value="hoodie">Hoodie</option>
                <option value="jacket">Jacket</option>
                <option value="dress">Dress</option>
                <option value="generic">Generic</option>
              </select>
            </label>
          </div>

          <p className="text-xs text-gray-600">
            Tip: Cloud AI cho ket qua tot nhat. Local SD can GPU 4-6GB. CPU mode nhanh nhat nhung chat luong thap.
          </p>
        </div>
      )}

      {error && (
        <div className="max-w-2xl mx-auto mt-6 p-4 bg-red-500/10 border border-red-500/20 rounded-xl text-red-400 text-sm text-center">
          {error}
        </div>
      )}

      {info && (
        <div className="max-w-2xl mx-auto mt-6 p-4 bg-primary/10 border border-primary/20 rounded-xl text-sm text-gray-300">
          <pre className="whitespace-pre-wrap font-mono text-xs">{info}</pre>
        </div>
      )}

      {/* Quick product selector */}
      {!id && (
        <div className="mt-12">
          <h3 className="text-xl font-semibold text-white mb-6 text-center">
            Hoac chon nhanh tu cua hang
          </h3>
          <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-3">
            {products.map((p) => (
              <button
                key={p.id}
                onClick={() => setClothImage(p.image)}
                className={`rounded-xl overflow-hidden border-2 transition-all hover:scale-105 ${
                  clothImage === p.image ? "border-primary" : "border-transparent"
                }`}
              >
                <img src={p.image} alt={p.name} className="w-full h-24 object-cover" />
                <p className="text-xs text-gray-400 p-1 truncate">{p.name}</p>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
