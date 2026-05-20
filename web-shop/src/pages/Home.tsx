import { useEffect, useMemo, useState } from "react";
import { Search, Sparkles, ArrowRight, ShieldCheck, Truck, RefreshCw } from "lucide-react";
import { Link } from "react-router-dom";
import ProductCard, { type BackendProduct } from "../components/ProductCard";
import { ProductCardSkeleton } from "../components/Skeleton";
import { productsApi, wishlistApi } from "../lib/api";
import { useAuth } from "../context/AuthContext";

const FEATURE_ITEMS = [
  { icon: Truck, title: "Miễn phí vận chuyển", sub: "Đơn từ 500K" },
  { icon: RefreshCw, title: "Đổi trả 30 ngày", sub: "Hoàn tiền nhanh" },
  { icon: ShieldCheck, title: "Bảo hành chính hãng", sub: "100% chính phẩm" },
  { icon: Sparkles, title: "AI Thử đồ ảo", sub: "Trải nghiệm mới" },
];

export default function Home() {
  const { user } = useAuth();
  const [search, setSearch] = useState("");
  const [category, setCategory] = useState("Tất cả");
  const [sortBy, setSortBy] = useState("default");
  const [priceRange, setPriceRange] = useState<string>("all");
  const [minRating, setMinRating] = useState(0);
  const [page, setPage] = useState(1);
  const pageSize = 12;
  const [products, setProducts] = useState<BackendProduct[]>([]);
  const [categories, setCategories] = useState<{ name: string; slug: string }[]>([]);
  const [wishIds, setWishIds] = useState<Set<number>>(new Set());
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      productsApi.list({ size: 60 }).catch(() => ({ items: [] as any[] })),
      productsApi.categories().catch(() => [] as any[]),
    ]).then(([res, cats]) => {
      setProducts((res as any).items ?? []);
      setCategories(cats as any[]);
      setLoading(false);
    });
  }, []);

  useEffect(() => {
    if (!user) {
      setWishIds(new Set());
      return;
    }
    wishlistApi
      .list()
      .then((list) => setWishIds(new Set(list.map((w: any) => w.product?.id).filter(Boolean))))
      .catch(() => {});
  }, [user]);

  const filtered = useMemo(() => {
    const [minP, maxP] =
      priceRange === "0-200"
        ? [0, 200_000]
        : priceRange === "200-500"
          ? [200_000, 500_000]
          : priceRange === "500-1000"
            ? [500_000, 1_000_000]
            : priceRange === "1000+"
              ? [1_000_000, Infinity]
              : [0, Infinity];
    let list = products.filter((p) => {
      const matchSearch = p.name.toLowerCase().includes(search.toLowerCase());
      const matchCat = category === "Tất cả" || p.category?.name === category;
      const price = Number(p.price) || 0;
      const matchPrice = price >= minP && price <= maxP;
      const rating = Number(p.ratingAvg) || 0;
      const matchRating = rating >= minRating;
      return matchSearch && matchCat && matchPrice && matchRating;
    });
    if (sortBy === "price-asc")
      list = [...list].sort((a, b) => Number(a.price) - Number(b.price));
    if (sortBy === "price-desc")
      list = [...list].sort((a, b) => Number(b.price) - Number(a.price));
    if (sortBy === "rating")
      list = [...list].sort((a, b) => Number(b.ratingAvg) - Number(a.ratingAvg));
    return list;
  }, [search, category, sortBy, priceRange, minRating, products]);

  useEffect(() => {
    setPage(1);
  }, [search, category, sortBy, priceRange, minRating]);

  const totalPages = Math.max(1, Math.ceil(filtered.length / pageSize));
  const paged = filtered.slice((page - 1) * pageSize, page * pageSize);

  const newest = products.slice(0, 4);
  const categoryChips = ["Tất cả", ...categories.map((c) => c.name)];

  return (
    <div className="bg-white">
      <section className="relative overflow-hidden">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16 lg:py-24">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
            <div className="animate-fade-up">
              <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-[var(--color-bg-warm)] border border-[var(--color-accent)]/40 text-[11px] font-semibold uppercase tracking-wider text-[var(--color-ink)]">
                <Sparkles className="w-3 h-3 text-[var(--color-accent)]" /> Bộ sưu tập 2026
              </span>
              <h1 className="font-display text-4xl sm:text-5xl md:text-6xl lg:text-7xl text-[var(--color-ink)] mt-4 leading-[1.05]">
                Mặc thử
                <br />
                trước khi <em className="text-[var(--color-accent)] not-italic">mua</em>.
              </h1>
              <p className="text-lg text-[var(--color-ink-muted)] mt-6 max-w-md leading-relaxed">
                Trải nghiệm công nghệ Virtual Try-On AI. Tải ảnh của bạn, chọn trang phục
                yêu thích, xem kết quả thật nhất chỉ trong vài giây.
              </p>

              <div className="flex flex-wrap gap-3 mt-8">
                <Link to="/try-on" className="btn-tryon">
                  <Sparkles className="w-5 h-5" /> Thử đồ ngay
                </Link>
                <a href="#shop" className="btn-secondary">
                  Khám phá BST <ArrowRight className="w-4 h-4" />
                </a>
              </div>

              <div className="flex items-center gap-6 sm:gap-8 mt-10 pt-8 border-t border-[var(--color-line)] flex-wrap">
                <div>
                  <p className="text-2xl font-semibold text-[var(--color-ink)]">10K+</p>
                  <p className="text-xs text-[var(--color-ink-muted)] uppercase tracking-wider">
                    Khách hàng
                  </p>
                </div>
                <div>
                  <p className="text-2xl font-semibold text-[var(--color-ink)]">50K+</p>
                  <p className="text-xs text-[var(--color-ink-muted)] uppercase tracking-wider">
                    Lượt thử đồ
                  </p>
                </div>
                <div>
                  <p className="text-2xl font-semibold text-[var(--color-ink)]">4.9★</p>
                  <p className="text-xs text-[var(--color-ink-muted)] uppercase tracking-wider">
                    Đánh giá
                  </p>
                </div>
              </div>
            </div>

            <div className="relative">
              <div className="aspect-[4/5] rounded-3xl overflow-hidden bg-[var(--color-bg-soft)]">
                <img
                  src="https://images.unsplash.com/photo-1490481651871-ab68de25d43d?w=900"
                  alt="Hero"
                  className="w-full h-full object-cover"
                />
              </div>
              <div
                className="absolute -bottom-6 -left-6 bg-white rounded-2xl p-4 flex items-center gap-3 max-w-[260px] animate-fade-up"
                style={{ boxShadow: "0 18px 40px rgba(17,24,39,0.12)" }}
              >
                <div className="w-12 h-12 rounded-full bg-[var(--color-bg-warm)] flex items-center justify-center">
                  <Sparkles className="w-5 h-5 text-[var(--color-accent)]" />
                </div>
                <div>
                  <p className="text-xs text-[var(--color-ink-muted)]">AI đang xử lý</p>
                  <p className="font-semibold text-sm text-[var(--color-ink)]">
                    Try-On Result Ready
                  </p>
                  <div className="ai-progress-bar mt-1.5">
                    <div className="ai-progress-fill" style={{ width: "82%" }} />
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="border-y border-[var(--color-line)] bg-[var(--color-bg-soft)]">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 grid grid-cols-2 lg:grid-cols-4 gap-6">
          {FEATURE_ITEMS.map((f) => (
            <div key={f.title} className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-full bg-white flex items-center justify-center border border-[var(--color-line)]">
                <f.icon className="w-4 h-4 text-[var(--color-ink)]" />
              </div>
              <div>
                <p className="font-semibold text-sm text-[var(--color-ink)]">{f.title}</p>
                <p className="text-xs text-[var(--color-ink-muted)]">{f.sub}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {categories.length > 0 && (
        <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
          <h2 className="section-title text-center">Danh mục nổi bật</h2>
          <p className="section-sub text-center">Khám phá phong cách phù hợp với bạn</p>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-8">
            {categories.slice(0, 4).map((cat, i) => (
              <button
                key={cat.slug}
                onClick={() => {
                  setCategory(cat.name);
                  document.getElementById("shop")?.scrollIntoView({ behavior: "smooth" });
                }}
                className="group text-center"
              >
                <div className="aspect-square rounded-2xl bg-[var(--color-bg-soft)] overflow-hidden border border-[var(--color-line)] group-hover:border-[var(--color-accent)] transition-all group-hover:-translate-y-1">
                  <img
                    src={
                      products[i]?.images?.[0]?.url ??
                      "https://placehold.co/400?text=" + encodeURIComponent(cat.name)
                    }
                    alt={cat.name}
                    className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-500"
                  />
                </div>
                <p className="text-sm font-medium mt-2 text-[var(--color-ink)]">{cat.name}</p>
              </button>
            ))}
          </div>
        </section>
      )}

      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="rounded-3xl overflow-hidden grid grid-cols-1 lg:grid-cols-2 bg-[var(--color-bg-warm)] border border-[var(--color-accent)]/30">
          <div className="p-10 lg:p-14 flex flex-col justify-center">
            <span className="text-xs font-semibold uppercase tracking-[0.2em] text-[var(--color-accent)] mb-3">
              ⊹ AI Virtual Try-On
            </span>
            <h3 className="font-display text-2xl sm:text-3xl md:text-4xl text-[var(--color-ink)] leading-tight">
              Xem trước phong cách của bạn
              <br />
              trước khi quyết định.
            </h3>
            <p className="text-[var(--color-ink-muted)] mt-4 max-w-md">
              Công nghệ AI sử dụng TPS Warp + Diffusion để tạo ảnh thử đồ chân thực,
              giữ form dáng và họa tiết gốc của trang phục.
            </p>
            <div className="flex gap-3 mt-6">
              <Link to="/try-on" className="btn-primary">
                <Sparkles className="w-4 h-4" /> Bắt đầu thử đồ
              </Link>
              <a href="#shop" className="btn-ghost">
                Xem sản phẩm hỗ trợ
              </a>
            </div>
          </div>
          <div className="aspect-square lg:aspect-auto bg-white">
            <img
              src="https://images.unsplash.com/photo-1483985988355-763728e1935b?w=900"
              alt="Try-On"
              className="w-full h-full object-cover"
            />
          </div>
        </div>
      </section>

      {newest.length > 0 && (
        <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
          <div className="flex items-end justify-between mb-8">
            <div>
              <h2 className="section-title">Mới về</h2>
              <p className="section-sub mb-0">Bộ sưu tập mới nhất tuần này</p>
            </div>
            <a href="#shop" className="text-sm font-medium underline-offset-4 hover:underline">
              Xem tất cả →
            </a>
          </div>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-5">
            {newest.map((p) => (
              <ProductCard key={p.id} product={p} initialLiked={wishIds.has(p.id)} />
            ))}
          </div>
        </section>
      )}

      <section id="shop" className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <h2 className="section-title">Tất cả sản phẩm</h2>
        <p className="section-sub">Tìm trang phục hoàn hảo cho phong cách của bạn</p>

        <div className="flex flex-col md:flex-row gap-3 mb-6">
          <div className="relative flex-1">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-[var(--color-ink-muted)]" />
            <input
              type="text"
              placeholder="Tìm kiếm sản phẩm..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="input pl-11"
            />
          </div>
          <select
            value={priceRange}
            onChange={(e) => setPriceRange(e.target.value)}
            className="select md:w-44"
          >
            <option value="all">Khoảng giá: Tất cả</option>
            <option value="0-200">Dưới 200K</option>
            <option value="200-500">200K - 500K</option>
            <option value="500-1000">500K - 1tr</option>
            <option value="1000+">Trên 1tr</option>
          </select>
          <select
            value={minRating}
            onChange={(e) => setMinRating(Number(e.target.value))}
            className="select md:w-40"
          >
            <option value={0}>Đánh giá: Tất cả</option>
            <option value={3}>Từ 3★</option>
            <option value={4}>Từ 4★</option>
            <option value={4.5}>Từ 4.5★</option>
          </select>
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value)}
            className="select md:w-52"
          >
            <option value="default">Sắp xếp: Mặc định</option>
            <option value="price-asc">Giá: Thấp → Cao</option>
            <option value="price-desc">Giá: Cao → Thấp</option>
            <option value="rating">Đánh giá cao nhất</option>
          </select>
        </div>

        <div className="flex flex-wrap gap-2 mb-8">
          {categoryChips.map((cat) => (
            <button
              key={cat}
              onClick={() => setCategory(cat)}
              className={category === cat ? "chip-active" : "chip"}
            >
              {cat}
            </button>
          ))}
        </div>

        {loading ? (
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-5">
            {Array.from({ length: 8 }).map((_, i) => (
              <ProductCardSkeleton key={i} />
            ))}
          </div>
        ) : filtered.length > 0 ? (
          <>
            <div className="text-sm text-[var(--color-ink-muted)] mb-3">
              {filtered.length} sản phẩm
            </div>
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-5">
              {paged.map((product) => (
                <ProductCard
                  key={product.id}
                  product={product}
                  initialLiked={wishIds.has(product.id)}
                />
              ))}
            </div>
            {totalPages > 1 && (
              <div className="flex justify-center items-center gap-1 mt-10">
                <button
                  onClick={() => setPage((p) => Math.max(1, p - 1))}
                  disabled={page === 1}
                  className="px-3 py-1.5 border border-gray-200 rounded-md text-sm disabled:opacity-40"
                >
                  ←
                </button>
                {Array.from({ length: totalPages }).map((_, i) => {
                  const n = i + 1;
                  const show =
                    n === 1 ||
                    n === totalPages ||
                    Math.abs(n - page) <= 1;
                  if (!show) {
                    if (n === 2 || n === totalPages - 1)
                      return <span key={n} className="px-2 text-gray-400">…</span>;
                    return null;
                  }
                  return (
                    <button
                      key={n}
                      onClick={() => setPage(n)}
                      className={`px-3 py-1.5 rounded-md text-sm border ${
                        n === page
                          ? 'bg-[var(--color-ink)] text-white border-[var(--color-ink)]'
                          : 'border-gray-200 hover:bg-gray-50'
                      }`}
                    >
                      {n}
                    </button>
                  );
                })}
                <button
                  onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                  disabled={page === totalPages}
                  className="px-3 py-1.5 border border-gray-200 rounded-md text-sm disabled:opacity-40"
                >
                  →
                </button>
              </div>
            )}
          </>
        ) : (
          <div className="text-center py-20">
            <p className="text-[var(--color-ink-muted)] text-lg">
              Không tìm thấy sản phẩm phù hợp.
            </p>
          </div>
        )}
      </section>
    </div>
  );
}
