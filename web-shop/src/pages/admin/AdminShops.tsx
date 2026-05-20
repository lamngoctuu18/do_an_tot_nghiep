import { useEffect, useState } from 'react';
import { adminApi, ApiError } from '../../lib/api';
import { useToast } from '../../components/Toast';
import ReasonModal from '../../components/ReasonModal';

export default function AdminShops() {
  const toast = useToast();
  const [items, setItems] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [rejectTarget, setRejectTarget] = useState<any | null>(null);

  const load = async () => {
    setLoading(true);
    try {
      setItems(await adminApi.shops());
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const approve = async (s: any) => {
    if (!confirm(`Duyệt shop "${s.shopName ?? s.name}"?`)) return;
    try {
      await adminApi.approveShop(s.id);
      toast.show(`Đã duyệt shop ${s.shopName ?? s.name}`, 'success');
      await load();
    } catch (e: any) {
      toast.show(e instanceof ApiError ? e.message : 'Lỗi', 'error');
    }
  };

  const handleReject = async (reason: string) => {
    if (!rejectTarget) return;
    try {
      await adminApi.rejectShop(rejectTarget.id, reason);
      toast.show(`Đã từ chối shop`, 'success');
      setRejectTarget(null);
      await load();
    } catch (e: any) {
      toast.show(e instanceof ApiError ? e.message : 'Lỗi', 'error');
    }
  };

  if (loading) return <div className="text-gray-500">Đang tải...</div>;

  return (
    <div>
      <h2 className="text-xl font-semibold mb-4">Cửa hàng</h2>
      {items.length === 0 ? (
        <div className="text-gray-500 border border-dashed border-gray-200 rounded-md p-8 text-center">
          Chưa có shop nào.
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-gray-50 text-left">
              <tr>
                <th className="p-2.5">ID</th>
                <th className="p-2.5">Tên shop</th>
                <th className="p-2.5">Chủ shop</th>
                <th className="p-2.5">Trạng thái</th>
                <th className="p-2.5">Hành động</th>
              </tr>
            </thead>
            <tbody>
              {items.map((s) => (
                <tr key={s.id} className="border-t border-gray-100">
                  <td className="p-2.5">{s.id}</td>
                  <td className="p-2.5 font-medium">{s.shopName ?? s.name}</td>
                  <td className="p-2.5">
                    <div>{s.user?.fullName ?? '-'}</div>
                    <div className="text-xs text-gray-500">{s.user?.email}</div>
                  </td>
                  <td className="p-2.5">
                    <span
                      className={`text-xs px-2 py-0.5 rounded ${
                        s.status === 'APPROVED'
                          ? 'bg-green-100 text-green-700'
                          : s.status === 'REJECTED'
                            ? 'bg-red-100 text-red-700'
                            : 'bg-amber-100 text-amber-700'
                      }`}
                    >
                      {s.status}
                    </span>
                    {s.rejectReason && (
                      <div className="text-xs text-red-600 mt-1">Lý do: {s.rejectReason}</div>
                    )}
                  </td>
                  <td className="p-2.5">
                    {s.status === 'PENDING' && (
                      <div className="flex gap-1.5">
                        <button
                          onClick={() => approve(s)}
                          className="px-2 py-1 bg-green-600 text-white rounded text-xs"
                        >
                          Duyệt
                        </button>
                        <button
                          onClick={() => setRejectTarget(s)}
                          className="px-2 py-1 bg-red-600 text-white rounded text-xs"
                        >
                          Từ chối
                        </button>
                      </div>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <ReasonModal
        open={!!rejectTarget}
        title={`Từ chối shop "${rejectTarget?.shopName ?? ''}"`}
        placeholder="Lý do từ chối (tối thiểu 3 ký tự)..."
        confirmLabel="Từ chối"
        onClose={() => setRejectTarget(null)}
        onConfirm={handleReject}
      />
    </div>
  );
}
