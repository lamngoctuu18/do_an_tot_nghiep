import { useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import {
  Star,
  ShoppingBag,
  Sparkles,
  ArrowLeft,
  Heart,
  Truck,
  ShieldCheck,
  RefreshCw,
} from 'lucide-react';
import { productsApi, ApiError } from '../lib/api';
import { useCart } from '../context/CartContext';
import { useAuth } from '../context/AuthContext';
import { useToast } from '../components/Toast';
import ProductReviews from '../components/ProductReviews';

const fmt = (v: any) => new Intl.NumberFormat('vi-VN').format(+v) + 'đ';

export default function ProductDetail() {
  const { id = '' } = useParams();
  const { user } = useAuth();
  const { addItem } = useCart();
  const toast = useToast();
  const [product, setProduct] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [notFound, setNotFound] = useState(false);
  const [color, setColor] = useState<string>('');
  const [size, setSize] = useState<string>('');
  const [qty, setQty] = useState(1);
  const [adding, setAdding] = useState(false);
  const [tab, setTab] = useState<'desc' | 'reviews'>('desc');

  useEffect(() => {
    setLoading(true);
    setNotFound(false);
    productsApi
      .detail(id)
      .then((p) => {
        setProduct(p);
        const v = p.variants?.[0];
        if (v) {
          setColor(v.color ?? '');
          setSize(v.size ?? '');
        }
      })
      .catch(() => setNotFound(true))
      .finally(() => setLoading(false));
  }, [id]);

  if (loading) return <div className="py-20 text-center text-gray-500">Đang tải...</div>;
  if (notFound || !product) {
    return (
      <div className="max-w-7xl mx-auto px-4 py-20 text-center">
        <p className="text-gray-500 text-lg">Sản phẩm không tồn tại.</p>
        <Link to="/" className="inline-block mt-4 underline">
          ← Quay lại trang chủ
        </Link>
      </div>
    );
  }

  const variants: any[] = product.variants ?? [];
  const colors = Array.from(new Set(variants.map((v) => v.color).filter(Boolean)));
  const sizes = Array.from(new Set(variants.map((v) => v.size).filter(Boolean)));
  const selectedVariant = variants.find(
    (v) => (color ? v.color === color : true) && (size ? v.size === size : true),
  );
  const images: any[] = product.images ?? [];
  const mainImg = images[0]?.url;
  const price = +product.price + (selectedVariant ? +selectedVariant.priceDelta : 0);
  const original = product.originalPrice ? +product.originalPrice : 0;
  const discount = original > price ? Math.round((1 - price / original) * 100) : 0;

  const handleAdd = async () => {
    if (!user) {
      toast.show('Vui lòng đăng nhập để mua hàng.', 'info');
      return;
    }
    if (!selectedVariant) {
      toast.show('Vui lòng chọn phân loại.', 'error');
      return;
    }
    setAdding(true);
    try {
      await addItem(selectedVariant.id, qty);
      toast.show(`Đã thêm "${product.name}" vào giỏ`, 'success');
    } catch (e: any) {
      toast.show(e instanceof ApiError ? e.message : 'Lỗi thêm giỏ hàng', 'error');
    } finally {
      setAdding(false);
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <Link to="/" className="inline-flex items-center gap-1 text-sm text-gray-600 hover:text-black mb-4">
        <ArrowLeft className="w-4 h-4" /> Quay lại
      </Link>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-10">
        <div className="aspect-[3/4] rounded-xl overflow-hidden bg-[var(--color-bg-soft)]">
          {mainImg ? (
            <img src={mainImg} alt={product.name} className="w-full h-full object-cover" />
          ) : (
            <div className="w-full h-full flex items-center justify-center text-gray-400">No image</div>
          )}
        </div>
        <div>
          {product.badge && (
            <span className="inline-block bg-[var(--color-accent)]/20 px-2 py-1 text-xs rounded mb-2">
              {product.badge}
            </span>
          )}
          <h1 className="font-display text-3xl md:text-4xl mb-2">{product.name}</h1>
          <div className="flex items-center gap-2 mb-4 text-sm text-gray-600">
            <Star className="w-4 h-4 text-yellow-500 fill-yellow-500" />
            <span>
              {(+product.ratingAvg || 0).toFixed(1)} ({product.ratingCount ?? 0} đánh giá)
            </span>
          </div>
          <div className="flex items-baseline gap-3 mb-6">
            <span className="text-3xl font-semibold">{fmt(price)}</span>
            {original > price && (
              <>
                <span className="text-gray-400 line-through">{fmt(original)}</span>
                <span className="text-red-600 text-sm">-{discount}%</span>
              </>
            )}
          </div>

          {colors.length > 0 && (
            <div className="mb-4">
              <div className="text-sm mb-2">Màu sắc</div>
              <div className="flex flex-wrap gap-2">
                {colors.map((c) => (
                  <button
                    key={c}
                    onClick={() => setColor(c)}
                    className={`px-3 py-1.5 border rounded-md text-sm ${
                      color === c ? 'border-[var(--color-ink)] bg-[var(--color-bg-soft)]' : 'border-gray-200'
                    }`}
                  >
                    {c}
                  </button>
                ))}
              </div>
            </div>
          )}

          {sizes.length > 0 && (
            <div className="mb-4">
              <div className="text-sm mb-2">Kích cỡ</div>
              <div className="flex flex-wrap gap-2">
                {sizes.map((s) => (
                  <button
                    key={s}
                    onClick={() => setSize(s)}
                    className={`px-3 py-1.5 border rounded-md text-sm ${
                      size === s ? 'border-[var(--color-ink)] bg-[var(--color-bg-soft)]' : 'border-gray-200'
                    }`}
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}

          <div className="flex items-center gap-3 mb-6">
            <div className="inline-flex items-center border border-gray-200 rounded-md">
              <button onClick={() => setQty((q) => Math.max(1, q - 1))} className="w-10 h-10">
                −
              </button>
              <span className="w-12 text-center">{qty}</span>
              <button onClick={() => setQty((q) => q + 1)} className="w-10 h-10">
                +
              </button>
            </div>
            {selectedVariant && (
              <span className="text-sm text-gray-500">Tồn kho: {selectedVariant.stock}</span>
            )}
          </div>

          <div className="flex flex-wrap gap-3 mb-4">
            <button
              onClick={handleAdd}
              disabled={adding}
              className="btn-primary inline-flex items-center gap-2 disabled:opacity-50"
            >
              <ShoppingBag className="w-4 h-4" /> {adding ? 'Đang thêm...' : 'Thêm vào giỏ'}
            </button>
            {product.tryOnEnabled && (
              <Link to={`/try-on/${product.id}`} className="btn-secondary inline-flex items-center gap-2">
                <Sparkles className="w-4 h-4" /> Thử đồ ảo
              </Link>
            )}
            <button className="p-2.5 border border-gray-200 rounded-md">
              <Heart className="w-5 h-5" />
            </button>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs text-gray-600 mt-6 pt-6 border-t border-gray-100">
            <div className="flex items-center gap-1.5">
              <Truck className="w-4 h-4" /> Free ship 500K
            </div>
            <div className="flex items-center gap-1.5">
              <RefreshCw className="w-4 h-4" /> Đổi trả 30 ngày
            </div>
            <div className="flex items-center gap-1.5">
              <ShieldCheck className="w-4 h-4" /> Chính hãng
            </div>
          </div>
        </div>
      </div>

      <section className="mt-10">
        <div className="flex gap-6 border-b border-gray-100 mb-4">
          {(['desc', 'reviews'] as const).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`pb-3 text-sm ${
                tab === t ? 'border-b-2 border-[var(--color-ink)] font-semibold' : 'text-gray-500'
              }`}
            >
              {t === 'desc' ? 'Mô tả' : 'Đánh giá'}
            </button>
          ))}
        </div>
        {tab === 'desc' && (
          <div className="prose max-w-none text-sm text-gray-700 whitespace-pre-line">
            {product.description || 'Chưa có mô tả.'}
          </div>
        )}
        {tab === 'reviews' && <ProductReviews productId={product.id} />}
      </section>
    </div>
  );
}
