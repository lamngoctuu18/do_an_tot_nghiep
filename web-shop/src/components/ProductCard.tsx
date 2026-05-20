import { Link, useNavigate } from "react-router-dom";
import { Star, Sparkles, Heart } from "lucide-react";
import { useState } from "react";
import { wishlistApi, ApiError } from "../lib/api";
import { useAuth } from "../context/AuthContext";
import { useToast } from "./Toast";

export type BackendProduct = {
  id: number;
  name: string;
  slug: string;
  price: string | number;
  originalPrice?: string | number | null;
  tryOnEnabled?: boolean;
  ratingAvg?: string | number;
  ratingCount?: number;
  badge?: string | null;
  category?: { name?: string; slug?: string } | null;
  images?: { url: string; position?: number }[];
};

const fmt = (v: any) =>
  new Intl.NumberFormat("vi-VN").format(Math.round(Number(v) || 0)) + "đ";

interface Props {
  product: BackendProduct;
  initialLiked?: boolean;
}

export default function ProductCard({ product, initialLiked = false }: Props) {
  const { user } = useAuth();
  const toast = useToast();
  const nav = useNavigate();
  const [liked, setLiked] = useState(initialLiked);
  const [popKey, setPopKey] = useState(0);
  const [busy, setBusy] = useState(false);

  const price = Number(product.price) || 0;
  const original = product.originalPrice ? Number(product.originalPrice) : 0;
  const discount = original > price ? Math.round((1 - price / original) * 100) : 0;
  const image = product.images?.[0]?.url ?? "https://placehold.co/600x800?text=No+Image";
  const rating = Number(product.ratingAvg) || 0;
  const reviewsCount = product.ratingCount ?? 0;

  const toggleLike = async (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (!user) {
      toast.show("Vui lòng đăng nhập để dùng danh sách yêu thích", "info");
      nav("/login");
      return;
    }
    if (busy) return;
    setBusy(true);
    const wasLiked = liked;
    setLiked(!wasLiked);
    setPopKey((k) => k + 1);
    try {
      if (wasLiked) {
        await wishlistApi.remove(product.id);
        toast.show(`Đã bỏ "${product.name}" khỏi yêu thích`, "info");
      } else {
        await wishlistApi.add(product.id);
        toast.show(`Đã thêm "${product.name}" vào yêu thích`, "success");
      }
    } catch (err: any) {
      setLiked(wasLiked);
      toast.show(err instanceof ApiError ? err.message : "Lỗi cập nhật yêu thích", "error");
    } finally {
      setBusy(false);
    }
  };

  return (
    <article className="group card-hover">
      <Link
        to={`/product/${product.slug}`}
        className="block relative overflow-hidden bg-[var(--color-bg-soft)]"
      >
        <img
          src={image}
          alt={product.name}
          className="w-full aspect-[3/4] object-cover transition-transform duration-500 group-hover:scale-[1.06]"
        />

        <div className="absolute top-3 left-3 flex flex-col gap-1.5">
          {product.badge === "Mới" && <span className="badge-new">{product.badge}</span>}
          {product.badge && product.badge !== "Mới" && product.badge !== "Hot" && (
            <span className="badge-new">{product.badge}</span>
          )}
          {product.badge === "Hot" && <span className="badge-hot">Hot</span>}
          {product.tryOnEnabled && (
            <span className="badge-tryon">
              <Sparkles className="w-3 h-3" /> Try-On
            </span>
          )}
        </div>

        {discount > 0 && (
          <span className="badge-sale absolute top-3 right-3">-{discount}%</span>
        )}

        <button
          onClick={toggleLike}
          disabled={busy}
          aria-label="Yêu thích"
          className="absolute bottom-3 right-3 w-10 h-10 rounded-full bg-white/95 backdrop-blur flex items-center justify-center shadow-sm hover:scale-110 transition-transform disabled:opacity-60"
        >
          <Heart
            key={popKey}
            className={`w-4 h-4 transition-colors ${
              liked
                ? "fill-[var(--color-danger)] text-[var(--color-danger)] animate-heart-pop"
                : "text-[var(--color-ink-muted)]"
            }`}
          />
        </button>

        {product.tryOnEnabled && (
          <div className="absolute inset-x-3 bottom-3 flex gap-2 opacity-0 translate-y-3 group-hover:opacity-100 group-hover:translate-y-0 transition-all duration-300 pr-14">
            <Link
              to={`/try-on/${product.slug}`}
              onClick={(e) => e.stopPropagation()}
              className="flex-1 flex items-center justify-center gap-1.5 px-3 py-2.5 rounded-full bg-[var(--color-accent)] text-[var(--color-ink)] text-xs font-semibold hover:-translate-y-0.5 transition-transform"
            >
              <Sparkles className="w-3.5 h-3.5" /> Thử ngay
            </Link>
          </div>
        )}
      </Link>

      <div className="p-4">
        <p className="text-[11px] text-[var(--color-ink-muted)] uppercase tracking-wider mb-1">
          {product.category?.name ?? ""}
        </p>
        <Link to={`/product/${product.slug}`}>
          <h3 className="font-medium text-[var(--color-ink)] line-clamp-1 group-hover:text-[var(--color-primary)] transition-colors">
            {product.name}
          </h3>
        </Link>

        <div className="flex items-center gap-1.5 mt-1.5 text-xs text-[var(--color-ink-muted)]">
          <Star className="w-3.5 h-3.5 fill-[var(--color-warning)] text-[var(--color-warning)]" />
          <span className="text-[var(--color-ink)] font-medium">{rating.toFixed(1)}</span>
          <span>({reviewsCount})</span>
        </div>

        <div className="flex items-baseline gap-2 mt-2">
          <span className="text-lg font-semibold text-[var(--color-ink)]">{fmt(price)}</span>
          {original > 0 && (
            <span className="text-sm text-[var(--color-ink-disabled)] line-through">
              {fmt(original)}
            </span>
          )}
        </div>
      </div>
    </article>
  );
}
