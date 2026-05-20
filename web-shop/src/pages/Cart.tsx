import { Link, useNavigate } from 'react-router-dom';
import { Trash2, Plus, Minus, ShoppingBag, ArrowLeft, Sparkles, ShieldCheck } from 'lucide-react';
import { useCart } from '../context/CartContext';
import { useAuth } from '../context/AuthContext';
import { useToast } from '../components/Toast';
import { ApiError } from '../lib/api';

const fmt = (v: number) => new Intl.NumberFormat('vi-VN').format(v) + 'đ';

export default function Cart() {
  const { user } = useAuth();
  const { items, loading, removeItem, updateQuantity, totalPrice } = useCart();
  const toast = useToast();
  const nav = useNavigate();
  const shipping = totalPrice >= 500_000 ? 0 : totalPrice > 0 ? 30_000 : 0;
  const total = totalPrice + shipping;

  const onUpdateQty = async (item: any, next: number) => {
    if (next < 1) return;
    try {
      await updateQuantity(item.id, next);
    } catch (e: any) {
      toast.show(e instanceof ApiError ? e.message : 'Lỗi cập nhật số lượng', 'error');
    }
  };

  const onRemove = async (item: any) => {
    if (!confirm(`Xoá "${item.variant.product.name}" khỏi giỏ?`)) return;
    try {
      await removeItem(item.id);
      toast.show('Đã xoá khỏi giỏ', 'info');
    } catch (e: any) {
      toast.show(e instanceof ApiError ? e.message : 'Lỗi xoá', 'error');
    }
  };

  if (!user) {
    return (
      <div className="max-w-3xl mx-auto px-4 py-24 text-center">
        <h2 className="text-2xl font-semibold mb-3">Vui lòng đăng nhập</h2>
        <p className="text-gray-500 mb-6">Đăng nhập để xem giỏ hàng của bạn.</p>
        <Link to="/login" state={{ from: '/cart' }} className="bg-[var(--color-ink)] text-white px-6 py-3 rounded-md">
          Đăng nhập
        </Link>
      </div>
    );
  }

  if (loading) return <div className="py-20 text-center text-gray-500">Đang tải giỏ hàng...</div>;

  if (items.length === 0) {
    return (
      <div className="max-w-3xl mx-auto px-4 py-24 text-center">
        <div className="w-24 h-24 mx-auto rounded-full bg-[var(--color-bg-soft)] flex items-center justify-center mb-6">
          <ShoppingBag className="w-10 h-10 text-gray-300" />
        </div>
        <h2 className="font-display text-3xl mb-3">Giỏ hàng trống</h2>
        <p className="text-gray-500 mb-8">Hãy chọn sản phẩm yêu thích để thêm vào giỏ hàng.</p>
        <Link to="/" className="btn-primary inline-flex items-center gap-2">
          <ArrowLeft className="w-4 h-4" /> Tiếp tục mua sắm
        </Link>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <Link to="/" className="inline-flex items-center gap-1 text-sm text-gray-600 hover:text-black mb-6">
        <ArrowLeft className="w-4 h-4" /> Tiếp tục mua sắm
      </Link>
      <h1 className="font-display text-2xl md:text-3xl lg:text-4xl mb-6 md:mb-8">
        Giỏ hàng ({items.reduce((s, i) => s + i.quantity, 0)} sản phẩm)
      </h1>
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2 space-y-4">
          {items.map((item) => {
            const unitPrice = +item.variant.product.price + +item.variant.priceDelta;
            const img = item.variant.product.images?.[0]?.url;
            return (
              <div key={item.id} className="card p-4 flex flex-col sm:flex-row gap-4">
                <div className="w-full sm:w-28 h-32 sm:h-28 rounded-xl overflow-hidden bg-[var(--color-bg-soft)] shrink-0">
                  {img && <img src={img} alt={item.variant.product.name} className="w-full h-full object-cover" />}
                </div>
                <div className="flex-1 min-w-0">
                  <Link to={`/product/${item.variant.product.slug}`} className="font-medium hover:underline">
                    {item.variant.product.name}
                  </Link>
                  <div className="flex flex-wrap items-center gap-2 mt-1 text-sm text-gray-500">
                    {item.variant.size && <span className="badge-soft">Size: {item.variant.size}</span>}
                    {item.variant.color && <span className="badge-soft">Màu: {item.variant.color}</span>}
                  </div>
                  <p className="text-lg font-semibold mt-2">{fmt(unitPrice)}</p>
                </div>
                <div className="flex sm:flex-col items-center sm:items-end justify-between gap-2">
                  <div className="inline-flex items-center border border-[var(--color-line)] rounded-full">
                    <button
                      onClick={() => onUpdateQty(item, item.quantity - 1)}
                      className="w-9 h-9 flex items-center justify-center hover:bg-[var(--color-bg-soft)] rounded-l-full"
                    >
                      <Minus className="w-3.5 h-3.5" />
                    </button>
                    <span className="w-10 text-center text-sm font-medium">{item.quantity}</span>
                    <button
                      onClick={() => onUpdateQty(item, item.quantity + 1)}
                      className="w-9 h-9 flex items-center justify-center hover:bg-[var(--color-bg-soft)] rounded-r-full"
                    >
                      <Plus className="w-3.5 h-3.5" />
                    </button>
                  </div>
                  <button onClick={() => onRemove(item)} className="p-2 text-gray-500 hover:text-red-600">
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>
            );
          })}
        </div>
        <aside className="lg:col-span-1">
          <div className="card p-6 sticky top-24">
            <h3 className="font-display text-xl mb-5">Tóm tắt đơn hàng</h3>
            <div className="space-y-3 pb-4 border-b border-[var(--color-line)]">
              <div className="flex justify-between text-sm">
                <span className="text-gray-500">Tạm tính</span>
                <span className="font-medium">{fmt(totalPrice)}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-gray-500">Phí vận chuyển</span>
                <span className="font-medium text-green-600">{shipping === 0 ? 'Miễn phí' : fmt(shipping)}</span>
              </div>
              {totalPrice < 500_000 && (
                <p className="text-xs text-gray-500 bg-[var(--color-bg-warm)] p-2 rounded-lg">
                  Mua thêm <strong>{fmt(500_000 - totalPrice)}</strong> để được miễn phí vận chuyển.
                </p>
              )}
            </div>
            <div className="flex justify-between items-baseline py-4">
              <span className="text-sm font-medium">Tổng cộng</span>
              <span className="text-2xl font-semibold">{fmt(total)}</span>
            </div>
            <button onClick={() => nav('/checkout')} className="btn-primary w-full">
              Thanh toán
            </button>
            <Link to="/try-on" className="btn-secondary w-full mt-3 inline-flex items-center justify-center gap-2">
              <Sparkles className="w-4 h-4" /> Thử đồ ảo trước
            </Link>
            <div className="mt-6 pt-6 border-t border-[var(--color-line)] flex items-center gap-2 text-xs text-gray-500">
              <ShieldCheck className="w-4 h-4 text-green-600" />
              Thanh toán bảo mật · Đổi trả 30 ngày
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
}
