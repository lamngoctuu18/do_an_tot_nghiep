import { useEffect, useState } from 'react';
import { adminApi, ApiError } from '../../lib/api';
import { Lock, Unlock } from 'lucide-react';
import { useToast } from '../../components/Toast';

export default function AdminUsers() {
  const toast = useToast();
  const [items, setItems] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState<number | null>(null);

  const load = async () => {
    setLoading(true);
    try {
      const data = await adminApi.users();
      setItems(Array.isArray(data) ? data : data.items ?? []);
    } catch (e: any) {
      setErr(e instanceof ApiError ? e.message : 'Lỗi tải');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const toggle = async (u: any) => {
    const locking = u.status !== 'LOCKED';
    if (locking && !confirm(`Khoá tài khoản "${u.email}"?`)) return;
    setBusy(u.id);
    try {
      if (u.status === 'LOCKED') await adminApi.unlockUser(u.id);
      else await adminApi.lockUser(u.id);
      toast.show(locking ? `Đã khoá ${u.email}` : `Đã mở khoá ${u.email}`, 'success');
      await load();
    } catch (e: any) {
      toast.show(e instanceof ApiError ? e.message : 'Lỗi', 'error');
    } finally {
      setBusy(null);
    }
  };

  if (loading) return <div className="text-gray-500">Đang tải...</div>;
  if (err) return <div className="text-red-600 text-sm">{err}</div>;

  return (
    <div>
      <h2 className="text-xl font-semibold mb-4">Người dùng</h2>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="bg-gray-50 text-left">
            <tr>
              <th className="p-2.5">ID</th>
              <th className="p-2.5">Họ tên</th>
              <th className="p-2.5">Email</th>
              <th className="p-2.5">Vai trò</th>
              <th className="p-2.5">Trạng thái</th>
              <th className="p-2.5"></th>
            </tr>
          </thead>
          <tbody>
            {items.map((u) => (
              <tr key={u.id} className="border-t border-gray-100">
                <td className="p-2.5">{u.id}</td>
                <td className="p-2.5">{u.fullName}</td>
                <td className="p-2.5">{u.email}</td>
                <td className="p-2.5">
                  <span className="text-xs px-1.5 py-0.5 rounded bg-gray-100">{u.role}</span>
                </td>
                <td className="p-2.5">
                  <span
                    className={`text-xs px-2 py-0.5 rounded ${
                      u.status === 'LOCKED' ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'
                    }`}
                  >
                    {u.status}
                  </span>
                </td>
                <td className="p-2.5">
                  <button
                    onClick={() => toggle(u)}
                    disabled={busy === u.id || u.role === 'ADMIN'}
                    className="inline-flex items-center gap-1 text-xs px-2 py-1 border border-gray-200 rounded hover:bg-gray-50 disabled:opacity-40 disabled:cursor-not-allowed"
                    title={u.role === 'ADMIN' ? 'Không thể khoá admin' : ''}
                  >
                    {u.status === 'LOCKED' ? (
                      <>
                        <Unlock className="w-3.5 h-3.5" /> Mở khoá
                      </>
                    ) : (
                      <>
                        <Lock className="w-3.5 h-3.5" /> Khoá
                      </>
                    )}
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
