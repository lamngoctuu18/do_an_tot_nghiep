import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Heart, ShoppingBag, Trash2 } from 'lucide-react';
import { wishlistApi, cartApi, ApiError } from '../../lib/api';
import { useToast } from '../../components/Toast';

const fmt = (v: any) => new Intl.NumberFormat('vi-VN').format(+v) + 'đ';

export default function Wishlist() {
  const toast = useToast();
  const [items, setItems] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  const load = async () => {
    setLoading(true);
    try {
      setItems(await wishlistApi.list());
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const remove = async (productId: number, name: string) => {
    try {
      await wishlistApi.remove(productId);
      setItems((l) => l.filter((w) => w.product.id !== productId));
      toast.show(`Đã xoá "${name}" khỏi yêu thích`, 'info');
    } catch (e: any) {
      toast.show(e instanceof ApiError ? e.message : 'Lỗi xoá', 'error');
    }
  };

  const addToCart = async (product: any) => {
    try {
      const variantId = product.variants?.[0]?.id;
      if (!variantId) {
        toast.show('Sản phẩm không có biến thể.', 'error');
        return;
      }
      await cartApi.add(variantId, 1);
      toast.show(`Đã thêm "${product.name}" vào giỏ`, 'success');
    } catch (e: any) {
      toast.show(e instanceof ApiError ? e.message : 'Lỗi thêm giỏ', 'error');
    }
  };

  if (loading) return <div className="text-gray-500">Đang tải...</div>;

  return (
    <div>
      <h2 className="text-xl font-semibold mb-4">Sản phẩm yêu thích</h2>
      {items.length === 0 ? (
        <div className="text-gray-500 border border-dashed border-gray-200 rounded-md p-8 text-center">
          <Heart className="w-10 h-10 mx-auto mb-3 text-gray-300" />
          Chưa có sản phẩm nào trong danh sách yêu thích.
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {items.map((w) => {
            const p = w.product;
            const img = p.images?.[0]?.url;
            return (
              <div key={w.id} className="border border-gray-200 rounded-md overflow-hidden flex flex-col">
                <Link to={`/product/${p.slug}`} className="aspect-square bg-[var(--color-bg-soft)]">
                  {img && <img src={img} alt={p.name} className="w-full h-full object-cover" />}
                </Link>
                <div className="p-3 flex-1 flex flex-col">
                  <Link to={`/product/${p.slug}`} className="text-sm font-medium hover:underline line-clamp-2">
                    {p.name}
                  </Link>
                  <div className="font-semibold mt-1">{fmt(p.price)}</div>
                  <div className="flex gap-2 mt-3">
                    <button
                      onClick={() => addToCart(p)}
                      className="flex-1 bg-[var(--color-ink)] text-white rounded-md py-1.5 text-xs inline-flex items-center justify-center gap-1"
                    >
                      <ShoppingBag className="w-3.5 h-3.5" /> Thêm giỏ
                    </button>
                    <button
                      onClick={() => remove(p.id, p.name)}
                      className="p-1.5 border border-gray-200 rounded-md text-red-600"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
