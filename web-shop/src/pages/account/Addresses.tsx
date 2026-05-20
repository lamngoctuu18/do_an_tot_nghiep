import { useEffect, useState } from 'react';
import { addressesApi, ApiError } from '../../lib/api';
import { Pencil, Trash2, Plus, Check } from 'lucide-react';
import { useToast } from '../../components/Toast';

type Address = {
  id: number;
  recipient: string;
  phone: string;
  line1: string;
  ward: string;
  district: string;
  city: string;
  isDefault: boolean;
};

const empty = { recipient: '', phone: '', line1: '', ward: '', district: '', city: '', isDefault: false };

const PHONE_RE = /^(0|\+84)(3|5|7|8|9)\d{8}$/;

function validate(f: any): string | null {
  if (!f.recipient || f.recipient.trim().length < 2) return 'Tên người nhận phải có ít nhất 2 ký tự.';
  if (!PHONE_RE.test(f.phone?.replace(/\s/g, ''))) return 'Số điện thoại không hợp lệ (VD: 09xxxxxxxx).';
  if (!f.line1 || f.line1.trim().length < 3) return 'Địa chỉ (số nhà, đường) phải có ít nhất 3 ký tự.';
  if (!f.ward?.trim()) return 'Vui lòng nhập Phường/Xã.';
  if (!f.district?.trim()) return 'Vui lòng nhập Quận/Huyện.';
  if (!f.city?.trim()) return 'Vui lòng nhập Tỉnh/Thành.';
  return null;
}

export default function Addresses() {
  const toast = useToast();
  const [list, setList] = useState<Address[]>([]);
  const [loading, setLoading] = useState(true);
  const [editing, setEditing] = useState<Address | null>(null);
  const [creating, setCreating] = useState(false);
  const [form, setForm] = useState<any>(empty);
  const [err, setErr] = useState<string | null>(null);

  const load = async () => {
    setLoading(true);
    try {
      setList(await addressesApi.list());
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const openCreate = () => {
    setEditing(null);
    setForm(empty);
    setCreating(true);
    setErr(null);
  };

  const openEdit = (a: Address) => {
    setEditing(a);
    setForm({ ...a });
    setCreating(true);
    setErr(null);
  };

  const save = async (e: React.FormEvent) => {
    e.preventDefault();
    setErr(null);
    const v = validate(form);
    if (v) {
      setErr(v);
      return;
    }
    try {
      if (editing) await addressesApi.update(editing.id, form);
      else await addressesApi.create(form);
      setCreating(false);
      toast.show(editing ? 'Đã cập nhật địa chỉ' : 'Đã thêm địa chỉ', 'success');
      await load();
    } catch (e: any) {
      setErr(e instanceof ApiError ? e.message : 'Lỗi lưu địa chỉ');
    }
  };

  const remove = async (id: number) => {
    if (!confirm('Xoá địa chỉ này?')) return;
    try {
      await addressesApi.remove(id);
      toast.show('Đã xoá địa chỉ', 'info');
      await load();
    } catch (e: any) {
      toast.show(e instanceof ApiError ? e.message : 'Lỗi xoá', 'error');
    }
  };

  const setDefault = async (id: number) => {
    try {
      await addressesApi.setDefault(id);
      toast.show('Đã đặt làm mặc định', 'success');
      await load();
    } catch (e: any) {
      toast.show(e instanceof ApiError ? e.message : 'Lỗi', 'error');
    }
  };

  if (loading) return <div className="text-gray-500">Đang tải...</div>;

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold">Sổ địa chỉ</h2>
        <button
          onClick={openCreate}
          className="inline-flex items-center gap-1.5 bg-[var(--color-ink)] text-white px-4 py-2 rounded-md text-sm"
        >
          <Plus className="w-4 h-4" /> Thêm địa chỉ
        </button>
      </div>

      {list.length === 0 && (
        <div className="text-gray-500 border border-dashed border-gray-200 rounded-md p-8 text-center">
          Chưa có địa chỉ nào.
        </div>
      )}

      <div className="space-y-3">
        {list.map((a) => (
          <div key={a.id} className="border border-gray-200 rounded-md p-4">
            <div className="flex justify-between items-start gap-4">
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <span className="font-semibold">{a.recipient}</span>
                  <span className="text-gray-400">|</span>
                  <span className="text-gray-600">{a.phone}</span>
                  {a.isDefault && (
                    <span className="text-xs bg-[var(--color-accent)]/20 text-[var(--color-ink)] px-2 py-0.5 rounded">
                      Mặc định
                    </span>
                  )}
                </div>
                <div className="text-sm text-gray-600">
                  {a.line1}, {a.ward}, {a.district}, {a.city}
                </div>
              </div>
              <div className="flex items-center gap-2 shrink-0">
                {!a.isDefault && (
                  <button
                    onClick={() => setDefault(a.id)}
                    className="text-xs text-gray-600 hover:text-[var(--color-ink)] inline-flex items-center gap-1"
                  >
                    <Check className="w-3.5 h-3.5" /> Mặc định
                  </button>
                )}
                <button onClick={() => openEdit(a)} className="p-1.5 hover:bg-gray-50 rounded">
                  <Pencil className="w-4 h-4" />
                </button>
                <button onClick={() => remove(a.id)} className="p-1.5 hover:bg-red-50 rounded text-red-600">
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>

      {creating && (
        <div className="fixed inset-0 bg-black/40 z-50 flex items-center justify-center p-4">
          <form
            onSubmit={save}
            className="bg-white rounded-lg p-6 w-full max-w-lg space-y-3 max-h-[90vh] overflow-y-auto"
          >
            <h3 className="text-lg font-semibold">{editing ? 'Sửa địa chỉ' : 'Thêm địa chỉ'}</h3>
            {[
              ['recipient', 'Người nhận'],
              ['phone', 'Số điện thoại'],
              ['line1', 'Địa chỉ (số nhà, đường)'],
              ['ward', 'Phường/Xã'],
              ['district', 'Quận/Huyện'],
              ['city', 'Tỉnh/Thành'],
            ].map(([k, l]) => (
              <div key={k}>
                <label className="block text-sm mb-1">{l}</label>
                <input
                  required
                  value={form[k] ?? ''}
                  onChange={(e) => setForm({ ...form, [k]: e.target.value })}
                  className="w-full border border-gray-200 rounded-md px-3 py-2"
                />
              </div>
            ))}
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={!!form.isDefault}
                onChange={(e) => setForm({ ...form, isDefault: e.target.checked })}
              />
              Đặt làm địa chỉ mặc định
            </label>
            {err && <div className="text-red-600 text-sm">{err}</div>}
            <div className="flex justify-end gap-2 pt-2">
              <button
                type="button"
                onClick={() => setCreating(false)}
                className="px-4 py-2 border border-gray-200 rounded-md"
              >
                Hủy
              </button>
              <button type="submit" className="bg-[var(--color-ink)] text-white px-4 py-2 rounded-md">
                Lưu
              </button>
            </div>
          </form>
        </div>
      )}
    </div>
  );
}
