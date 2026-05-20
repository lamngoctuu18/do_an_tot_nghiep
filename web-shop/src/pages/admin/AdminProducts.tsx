import { useEffect, useState } from 'react';
import { adminApi, ApiError } from '../../lib/api';
import { useToast } from '../../components/Toast';
import ReasonModal from '../../components/ReasonModal';

const fmt = (v: any) => new Intl.NumberFormat('vi-VN').format(+v) + 'đ';

export default function AdminProducts() {
  const toast = useToast();
  const [items, setItems] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [rejectTarget, setRejectTarget] = useState<any | null>(null);

  const load = async () => {
    setLoading(true);
    try {
      const data = await adminApi.productsPending();
      setItems(Array.isArray(data) ? data : (data as any).items ?? []);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const approve = async (p: any) => {
    if (!confirm(`Duyệt sản phẩm "${p.name}"?`)) return;
    try {
      await adminApi.approveProduct(p.id);
      toast.show(`Đã duyệt "${p.name}"`, 'success');
      await load();
    } catch (e: any) {
      toast.show(e instanceof ApiError ? e.message : 'Lỗi', 'error');
    }
  };

  const handleReject = async (reason: string) => {
    if (!rejectTarget) return;
    try {
      await adminApi.rejectProduct(rejectTarget.id, reason);
      toast.show(`Đã từ chối "${rejectTarget.name}"`, 'success');
      setRejectTarget(null);
      await load();
    } catch (e: any) {
      toast.show(e instanceof ApiError ? e.message : 'Lỗi', 'error');
    }
  };

  if (loading) return <div className="text-gray-500">Đang tải...</div>;

  return (
    <div>
      <h2 className="text-xl font-semibold mb-4">Sản phẩm chờ duyệt</h2>
      {items.length === 0 ? (
        <div className="text-gray-500 border border-dashed border-gray-200 rounded-md p-8 text-center">
          Không có sản phẩm nào đang chờ duyệt.
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-gray-50 text-left">
              <tr>
                <th className="p-2.5">ID</th>
                <th className="p-2.5">Sản phẩm</th>
                <th className="p-2.5">Giá</th>
                <th className="p-2.5">Shop</th>
                <th className="p-2.5">Hành động</th>
              </tr>
            </thead>
            <tbody>
              {items.map((p) => (
                <tr key={p.id} className="border-t border-gray-100">
                  <td className="p-2.5">{p.id}</td>
                  <td className="p-2.5">
                    <div className="flex items-center gap-2">
                      {p.images?.[0]?.url && (
                        <img
                          src={p.images[0].url}
                          alt=""
                          className="w-10 h-10 rounded object-cover"
                        />
                      )}
                      <span className="font-medium">{p.name}</span>
                    </div>
                  </td>
                  <td className="p-2.5">{fmt(p.price)}</td>
                  <td className="p-2.5">{p.seller?.shopName ?? '-'}</td>
                  <td className="p-2.5">
                    <div className="flex gap-1.5">
                      <button
                        onClick={() => approve(p)}
                        className="px-2 py-1 bg-green-600 text-white rounded text-xs"
                      >
                        Duyệt
                      </button>
                      <button
                        onClick={() => setRejectTarget(p)}
                        className="px-2 py-1 bg-red-600 text-white rounded text-xs"
                      >
                        Từ chối
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <ReasonModal
        open={!!rejectTarget}
        title={`Từ chối "${rejectTarget?.name ?? ''}"`}
        placeholder="Lý do từ chối (tối thiểu 3 ký tự)..."
        confirmLabel="Từ chối"
        onClose={() => setRejectTarget(null)}
        onConfirm={handleReject}
      />
    </div>
  );
}
