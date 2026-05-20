import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Plus, Pencil, Trash2, Send } from 'lucide-react';
import { sellerApi, ApiError } from '../../lib/api';

const STATUS_LABEL: Record<string, string> = {
  DRAFT: 'Nháp',
  PENDING: 'Chờ duyệt',
  ACTIVE: 'Đang bán',
  REJECTED: 'Bị từ chối',
  HIDDEN: 'Đã ẩn',
};
const STATUS_COLOR: Record<string, string> = {
  DRAFT: 'bg-gray-100 text-gray-700',
  PENDING: 'bg-amber-100 text-amber-700',
  ACTIVE: 'bg-green-100 text-green-700',
  REJECTED: 'bg-red-100 text-red-700',
  HIDDEN: 'bg-slate-100 text-slate-700',
};

const fmt = (v: any) => new Intl.NumberFormat('vi-VN').format(+v) + 'đ';

export default function SellerProducts() {
  const [items, setItems] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  const load = async () => {
    setLoading(true);
    try {
      setItems(await sellerApi.myProducts());
    } catch (e: any) {
      alert(e instanceof ApiError ? e.message : 'Lỗi tải');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const submit = async (id: number) => {
    if (!confirm('Gửi sản phẩm này để chờ duyệt?')) return;
    try {
      await sellerApi.submitProduct(id);
      await load();
    } catch (e: any) {
      alert(e instanceof ApiError ? e.message : 'Lỗi');
    }
  };

  const del = async (id: number) => {
    if (!confirm('Xoá sản phẩm này?')) return;
    try {
      await sellerApi.deleteProduct(id);
      await load();
    } catch (e: any) {
      alert(e instanceof ApiError ? e.message : 'Lỗi');
    }
  };

  if (loading) return <div className="text-gray-500">Đang tải...</div>;

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold">Sản phẩm của shop</h2>
        <Link
          to="/seller/products/new"
          className="inline-flex items-center gap-1.5 px-3 py-2 bg-[var(--color-ink)] text-white text-sm rounded-md hover:opacity-90"
        >
          <Plus className="w-4 h-4" /> Thêm sản phẩm
        </Link>
      </div>

      {items.length === 0 ? (
        <div className="text-gray-500 border border-dashed border-gray-200 rounded-md p-10 text-center">
          Chưa có sản phẩm nào. Bấm "Thêm sản phẩm" để tạo mới.
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-gray-50 text-left">
              <tr>
                <th className="p-2.5">Ảnh</th>
                <th className="p-2.5">Tên</th>
                <th className="p-2.5">Giá</th>
                <th className="p-2.5">Biến thể</th>
                <th className="p-2.5">Trạng thái</th>
                <th className="p-2.5">Hành động</th>
              </tr>
            </thead>
            <tbody>
              {items.map((p) => (
                <tr key={p.id} className="border-t border-gray-100 align-middle">
                  <td className="p-2.5">
                    {p.images?.[0]?.url ? (
                      <img
                        src={p.images[0].url}
                        alt={p.name}
                        className="w-12 h-12 rounded object-cover"
                      />
                    ) : (
                      <div className="w-12 h-12 rounded bg-gray-100" />
                    )}
                  </td>
                  <td className="p-2.5">
                    <div className="font-medium">{p.name}</div>
                    <div className="text-xs text-gray-500">{p.slug}</div>
                  </td>
                  <td className="p-2.5">{fmt(p.price)}</td>
                  <td className="p-2.5">{p.variants?.length ?? 0}</td>
                  <td className="p-2.5">
                    <span
                      className={`text-xs px-2 py-0.5 rounded ${STATUS_COLOR[p.status] ?? 'bg-gray-100'}`}
                    >
                      {STATUS_LABEL[p.status] ?? p.status}
                    </span>
                    {p.rejectReason && (
                      <div className="text-[11px] text-red-600 mt-1">{p.rejectReason}</div>
                    )}
                  </td>
                  <td className="p-2.5">
                    <div className="flex flex-wrap gap-1.5">
                      <Link
                        to={`/seller/products/${p.id}`}
                        className="inline-flex items-center gap-1 text-xs px-2 py-1 border border-gray-200 rounded hover:bg-gray-50"
                      >
                        <Pencil className="w-3.5 h-3.5" /> Sửa
                      </Link>
                      {(p.status === 'DRAFT' || p.status === 'REJECTED') && (
                        <button
                          onClick={() => submit(p.id)}
                          className="inline-flex items-center gap-1 text-xs px-2 py-1 bg-amber-500 text-white rounded hover:opacity-90"
                        >
                          <Send className="w-3.5 h-3.5" /> Gửi duyệt
                        </button>
                      )}
                      <button
                        onClick={() => del(p.id)}
                        className="inline-flex items-center gap-1 text-xs px-2 py-1 bg-red-600 text-white rounded hover:opacity-90"
                      >
                        <Trash2 className="w-3.5 h-3.5" /> Xoá
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
