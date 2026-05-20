import { useEffect, useState } from 'react';

interface Props {
  open: boolean;
  title: string;
  placeholder?: string;
  confirmLabel?: string;
  confirmClass?: string;
  minLength?: number;
  onClose: () => void;
  onConfirm: (reason: string) => Promise<void> | void;
}

export default function ReasonModal({
  open,
  title,
  placeholder = 'Nhập lý do...',
  confirmLabel = 'Xác nhận',
  confirmClass = 'bg-red-600 text-white',
  minLength = 3,
  onClose,
  onConfirm,
}: Props) {
  const [reason, setReason] = useState('');
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (open) {
      setReason('');
      setErr(null);
    }
  }, [open]);

  if (!open) return null;

  const submit = async () => {
    const r = reason.trim();
    if (r.length < minLength) {
      setErr(`Lý do tối thiểu ${minLength} ký tự`);
      return;
    }
    setBusy(true);
    setErr(null);
    try {
      await onConfirm(r);
    } catch (e: any) {
      setErr(e?.message ?? 'Lỗi');
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/40 z-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-lg p-4 sm:p-6 w-full max-w-md">
        <h3 className="text-lg font-semibold mb-3">{title}</h3>
        <textarea
          value={reason}
          onChange={(e) => setReason(e.target.value)}
          rows={4}
          className="w-full border border-gray-200 rounded-md p-2.5 text-sm"
          placeholder={placeholder}
          autoFocus
        />
        {err && <div className="text-red-600 text-sm mt-2">{err}</div>}
        <div className="flex justify-end gap-2 mt-4">
          <button
            onClick={onClose}
            disabled={busy}
            className="px-4 py-2 border border-gray-200 rounded-md text-sm"
          >
            Đóng
          </button>
          <button
            onClick={submit}
            disabled={busy}
            className={`px-4 py-2 rounded-md text-sm disabled:opacity-50 ${confirmClass}`}
          >
            {busy ? 'Đang xử lý...' : confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
