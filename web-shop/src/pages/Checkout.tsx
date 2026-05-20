import { useEffect, useMemo, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { ordersApi, addressesApi, ApiError } from '../lib/api';
import { useCart } from '../context/CartContext';
import { useToast } from '../components/Toast';

const fmt = (v: number) => new Intl.NumberFormat('vi-VN').format(v) + 'đ';

export default function Checkout() {
  const { items, totalPrice, refresh } = useCart();
  const toast = useToast();
  const [addresses, setAddresses] = useState<any[]>([]);
  const [addressId, setAddressId] = useState<number | null>(null);
  const [paymentMethod, setPaymentMethod] = useState<'COD' | 'VNPAY'>('COD');
  const [note, setNote] = useState('');
  const [err, setErr] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const nav = useNavigate();

  useEffect(() => {
    addressesApi.list().then((list) => {
      setAddresses(list);
      const def = list.find((a: any) => a.isDefault) ?? list[0];
      if (def) setAddressId(def.id);
    });
  }, []);

  const shipping = useMemo(
    () => (totalPrice >= 500_000 ? 0 : totalPrice > 0 ? 30_000 : 0),
    [totalPrice],
  );
  const total = totalPrice + shipping;

  const submit = async () => {
    if (!addressId) {
      setErr('Vui lòng chọn địa chỉ giao hàng.');
      toast.show('Vui lòng chọn địa chỉ giao hàng.', 'error');
      return;
    }
    setSubmitting(true);
    setErr(null);
    try {
      const result: any = await ordersApi.checkout({ addressId, paymentMethod, note: note || undefined });
      await refresh();
      if (paymentMethod === 'VNPAY' && result.paymentUrl) {
        window.location.href = result.paymentUrl;
        return;
      }
      const code = result.code ?? result.order?.code;
      toast.show(`Đặt hàng thành công! Mã đơn: ${code}`, 'success');
      nav(`/account/orders/${code}`);
    } catch (e: any) {
      const msg = e instanceof ApiError ? e.message : 'Lỗi tạo đơn hàng';
      setErr(msg);
      toast.show(msg, 'error');
    } finally {
      setSubmitting(false);
    }
  };

  if (items.length === 0) {
    return (
      <div className="max-w-3xl mx-auto px-4 py-16 text-center">
        <h2 className="text-2xl font-semibold mb-3">Giỏ hàng trống</h2>
        <Link to="/" className="text-[var(--color-accent)]">
          Tiếp tục mua sắm
        </Link>
      </div>
    );
  }

  return (
    <div className="max-w-5xl mx-auto px-4 py-10">
      <h1 className="text-2xl font-semibold mb-6">Thanh toán</h1>
      <div className="grid grid-cols-1 lg:grid-cols-[1fr_360px] gap-8">
        <div className="space-y-6">
          <section className="border border-gray-200 rounded-md p-5">
            <div className="flex justify-between items-center mb-3">
              <h2 className="font-semibold">Địa chỉ giao hàng</h2>
              <Link to="/account/addresses" className="text-sm text-[var(--color-accent)]">
                + Thêm địa chỉ
              </Link>
            </div>
            {addresses.length === 0 ? (
              <p className="text-sm text-gray-500">
                Bạn chưa có địa chỉ. <Link to="/account/addresses" className="underline">Thêm ngay</Link>.
              </p>
            ) : (
              <div className="space-y-2">
                {addresses.map((a) => (
                  <label
                    key={a.id}
                    className={`flex gap-3 p-3 border rounded-md cursor-pointer ${
                      addressId === a.id ? 'border-[var(--color-accent)] bg-amber-50/30' : 'border-gray-200'
                    }`}
                  >
                    <input
                      type="radio"
                      name="addr"
                      checked={addressId === a.id}
                      onChange={() => setAddressId(a.id)}
                      className="mt-1"
                    />
                    <div className="text-sm">
                      <div className="font-medium">
                        {a.recipient} · {a.phone}
                      </div>
                      <div className="text-gray-600">
                        {a.line1}, {a.ward}, {a.district}, {a.city}
                      </div>
                    </div>
                  </label>
                ))}
              </div>
            )}
          </section>

          <section className="border border-gray-200 rounded-md p-5">
            <h2 className="font-semibold mb-3">Phương thức thanh toán</h2>
            {(['COD', 'VNPAY'] as const).map((m) => (
              <label key={m} className="flex items-center gap-3 py-2 cursor-pointer">
                <input
                  type="radio"
                  checked={paymentMethod === m}
                  onChange={() => setPaymentMethod(m)}
                />
                <span className="text-sm">
                  {m === 'COD' ? 'Thanh toán khi nhận hàng (COD)' : 'VNPAY (chuyển hướng cổng thanh toán)'}
                </span>
              </label>
            ))}
          </section>

          <section className="border border-gray-200 rounded-md p-5">
            <h2 className="font-semibold mb-3">Ghi chú</h2>
            <textarea
              rows={3}
              value={note}
              onChange={(e) => setNote(e.target.value)}
              placeholder="Ghi chú cho shop..."
              className="w-full border border-gray-200 rounded-md p-2.5 text-sm"
            />
          </section>
        </div>

        <aside className="border border-gray-200 rounded-md p-5 h-fit sticky top-24">
          <h2 className="font-semibold mb-3">Đơn hàng</h2>
          <div className="space-y-2 text-sm max-h-60 overflow-y-auto pr-1 mb-3">
            {items.map((it) => (
              <div key={it.id} className="flex justify-between gap-2">
                <span className="line-clamp-1">
                  {it.variant.product.name} × {it.quantity}
                </span>
                <span>{fmt((+it.variant.product.price + +it.variant.priceDelta) * it.quantity)}</span>
              </div>
            ))}
          </div>
          <div className="border-t pt-3 space-y-1 text-sm">
            <div className="flex justify-between">
              <span>Tạm tính</span>
              <span>{fmt(totalPrice)}</span>
            </div>
            <div className="flex justify-between">
              <span>Phí vận chuyển</span>
              <span>{shipping === 0 ? 'Miễn phí' : fmt(shipping)}</span>
            </div>
            <div className="flex justify-between font-semibold text-base pt-2 border-t mt-2">
              <span>Tổng</span>
              <span>{fmt(total)}</span>
            </div>
          </div>
          {err && <div className="text-red-600 text-sm mt-3">{err}</div>}
          <button
            onClick={submit}
            disabled={submitting || !addressId}
            className="w-full bg-[var(--color-ink)] text-white py-3 rounded-md font-medium mt-4 disabled:opacity-50"
          >
            {submitting ? 'Đang xử lý...' : 'Đặt hàng'}
          </button>
        </aside>
      </div>
    </div>
  );
}
