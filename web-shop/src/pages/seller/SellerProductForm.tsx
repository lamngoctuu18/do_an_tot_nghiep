import { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { Plus, Trash2, Upload, ArrowLeft } from 'lucide-react';
import { sellerApi, productsApi, ApiError } from '../../lib/api';

type Variant = {
  color?: string;
  size?: string;
  sku: string;
  stock: number;
  priceDelta?: number;
};
type TryonAsset = { clothImageUrl: string; clothType?: string };

const slugify = (s: string) =>
  s
    .toLowerCase()
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/đ/g, 'd')
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');

export default function SellerProductForm() {
  const { id } = useParams();
  const isEdit = !!id;
  const navigate = useNavigate();

  const [loading, setLoading] = useState(isEdit);
  const [saving, setSaving] = useState(false);
  const [categories, setCategories] = useState<any[]>([]);
  const [err, setErr] = useState<string | null>(null);

  const [name, setName] = useState('');
  const [slug, setSlug] = useState('');
  const [description, setDescription] = useState('');
  const [price, setPrice] = useState<number | ''>('');
  const [originalPrice, setOriginalPrice] = useState<number | ''>('');
  const [categoryId, setCategoryId] = useState<number | ''>('');
  const [badge, setBadge] = useState('');
  const [tryOnEnabled, setTryOnEnabled] = useState(false);
  const [images, setImages] = useState<string[]>([]);
  const [variants, setVariants] = useState<Variant[]>([
    { color: '', size: '', sku: '', stock: 0 },
  ]);
  const [tryonAssets, setTryonAssets] = useState<TryonAsset[]>([]);

  useEffect(() => {
    productsApi.categories().then(setCategories).catch(() => {});
    if (isEdit) {
      sellerApi
        .myProduct(+id!)
        .then((p) => {
          setName(p.name);
          setSlug(p.slug);
          setDescription(p.description ?? '');
          setPrice(+p.price);
          setOriginalPrice(p.originalPrice ? +p.originalPrice : '');
          setCategoryId(p.category?.id ?? '');
          setBadge(p.badge ?? '');
          setTryOnEnabled(!!p.tryOnEnabled);
          setImages(p.images?.map((i: any) => i.url) ?? []);
          setVariants(
            p.variants?.length
              ? p.variants.map((v: any) => ({
                  color: v.color ?? '',
                  size: v.size ?? '',
                  sku: v.sku,
                  stock: v.stock,
                  priceDelta: +v.priceDelta || 0,
                }))
              : [{ color: '', size: '', sku: '', stock: 0 }],
          );
          setTryonAssets(
            p.tryonAssets?.map((t: any) => ({
              clothImageUrl: t.clothImageUrl,
              clothType: t.clothType,
            })) ?? [],
          );
        })
        .catch((e) => setErr(e instanceof ApiError ? e.message : 'Lỗi tải'))
        .finally(() => setLoading(false));
    }
  }, [id]);

  const upload = async (file: File): Promise<string> => {
    const r = await sellerApi.uploadImage(file);
    return r.url;
  };

  const onUploadImage = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const url = await upload(file);
      setImages((prev) => [...prev, url]);
    } catch (err: any) {
      alert(err instanceof ApiError ? err.message : 'Lỗi upload');
    } finally {
      e.target.value = '';
    }
  };

  const onUploadTryon = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const url = await upload(file);
      setTryonAssets((prev) => [...prev, { clothImageUrl: url, clothType: 'AUTO' }]);
    } catch (err: any) {
      alert(err instanceof ApiError ? err.message : 'Lỗi upload');
    } finally {
      e.target.value = '';
    }
  };

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setErr(null);
    if (!name || !slug || price === '') {
      setErr('Vui lòng điền tên, slug và giá');
      return;
    }
    setSaving(true);
    try {
      const body: any = {
        name,
        slug,
        description: description || undefined,
        price: +price,
        originalPrice: originalPrice === '' ? undefined : +originalPrice,
        categoryId: categoryId === '' ? undefined : +categoryId,
        badge: badge || undefined,
        tryOnEnabled,
        images,
        variants: variants
          .filter((v) => v.sku.trim())
          .map((v) => ({
            color: v.color || undefined,
            size: v.size || undefined,
            sku: v.sku,
            stock: +v.stock || 0,
            priceDelta: +(v.priceDelta ?? 0),
          })),
        tryonAssets,
      };
      if (isEdit) {
        await sellerApi.updateProduct(+id!, body);
      } else {
        await sellerApi.createProduct(body);
      }
      navigate('/seller/products');
    } catch (e: any) {
      setErr(e instanceof ApiError ? e.message : 'Lỗi lưu');
    } finally {
      setSaving(false);
    }
  };

  if (loading) return <div className="text-gray-500">Đang tải...</div>;

  return (
    <form onSubmit={submit} className="space-y-6 max-w-3xl">
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={() => navigate('/seller/products')}
          className="inline-flex items-center gap-1 text-sm text-gray-600 hover:text-black"
        >
          <ArrowLeft className="w-4 h-4" /> Quay lại
        </button>
      </div>
      <h2 className="text-xl font-semibold">
        {isEdit ? 'Chỉnh sửa sản phẩm' : 'Thêm sản phẩm mới'}
      </h2>

      {err && <div className="text-sm text-red-600 bg-red-50 p-2.5 rounded">{err}</div>}

      <section className="space-y-3">
        <div>
          <label className="block text-sm font-medium mb-1">Tên sản phẩm *</label>
          <input
            value={name}
            onChange={(e) => {
              setName(e.target.value);
              if (!isEdit) setSlug(slugify(e.target.value));
            }}
            className="input"
            required
          />
        </div>
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="block text-sm font-medium mb-1">Slug *</label>
            <input value={slug} onChange={(e) => setSlug(e.target.value)} className="input" required />
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">Danh mục</label>
            <select
              value={categoryId}
              onChange={(e) => setCategoryId(e.target.value === '' ? '' : +e.target.value)}
              className="select"
            >
              <option value="">— Không —</option>
              {categories.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.name}
                </option>
              ))}
            </select>
          </div>
        </div>
        <div>
          <label className="block text-sm font-medium mb-1">Mô tả</label>
          <textarea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            rows={4}
            className="input"
          />
        </div>
        <div className="grid grid-cols-3 gap-3">
          <div>
            <label className="block text-sm font-medium mb-1">Giá (VND) *</label>
            <input
              type="number"
              value={price}
              onChange={(e) => setPrice(e.target.value === '' ? '' : +e.target.value)}
              className="input"
              required
              min={0}
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">Giá gốc</label>
            <input
              type="number"
              value={originalPrice}
              onChange={(e) =>
                setOriginalPrice(e.target.value === '' ? '' : +e.target.value)
              }
              className="input"
              min={0}
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">Nhãn (Hot/Mới)</label>
            <input value={badge} onChange={(e) => setBadge(e.target.value)} className="input" />
          </div>
        </div>
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={tryOnEnabled}
            onChange={(e) => setTryOnEnabled(e.target.checked)}
          />
          Bật thử đồ ảo (cần ảnh CLOTH bên dưới)
        </label>
      </section>

      <section>
        <div className="flex items-center justify-between mb-2">
          <h3 className="font-semibold">Ảnh sản phẩm</h3>
          <label className="inline-flex items-center gap-1.5 text-sm px-3 py-1.5 border border-gray-200 rounded cursor-pointer hover:bg-gray-50">
            <Upload className="w-3.5 h-3.5" /> Tải ảnh
            <input type="file" accept="image/*" onChange={onUploadImage} className="hidden" />
          </label>
        </div>
        <div className="grid grid-cols-4 gap-2">
          {images.map((url, i) => (
            <div key={i} className="relative group">
              <img src={url} alt="" className="w-full aspect-square object-cover rounded border" />
              <button
                type="button"
                onClick={() => setImages((p) => p.filter((_, j) => j !== i))}
                className="absolute top-1 right-1 p-1 bg-red-600 text-white rounded opacity-0 group-hover:opacity-100"
              >
                <Trash2 className="w-3.5 h-3.5" />
              </button>
            </div>
          ))}
          {images.length === 0 && (
            <div className="col-span-4 text-sm text-gray-500 border border-dashed rounded p-4 text-center">
              Chưa có ảnh
            </div>
          )}
        </div>
      </section>

      <section>
        <div className="flex items-center justify-between mb-2">
          <h3 className="font-semibold">Biến thể (size/màu/SKU)</h3>
          <button
            type="button"
            onClick={() =>
              setVariants((p) => [...p, { color: '', size: '', sku: '', stock: 0 }])
            }
            className="inline-flex items-center gap-1 text-sm px-2 py-1 border border-gray-200 rounded hover:bg-gray-50"
          >
            <Plus className="w-3.5 h-3.5" /> Thêm
          </button>
        </div>
        <div className="space-y-2">
          {variants.map((v, i) => (
            <div key={i} className="grid grid-cols-[1fr_1fr_1.4fr_0.8fr_0.8fr_auto] gap-2 items-center">
              <input
                placeholder="Màu"
                value={v.color ?? ''}
                onChange={(e) => {
                  const c = [...variants];
                  c[i] = { ...v, color: e.target.value };
                  setVariants(c);
                }}
                className="input"
              />
              <input
                placeholder="Size"
                value={v.size ?? ''}
                onChange={(e) => {
                  const c = [...variants];
                  c[i] = { ...v, size: e.target.value };
                  setVariants(c);
                }}
                className="input"
              />
              <input
                placeholder="SKU"
                value={v.sku}
                onChange={(e) => {
                  const c = [...variants];
                  c[i] = { ...v, sku: e.target.value };
                  setVariants(c);
                }}
                className="input"
              />
              <input
                type="number"
                placeholder="Tồn"
                value={v.stock}
                onChange={(e) => {
                  const c = [...variants];
                  c[i] = { ...v, stock: +e.target.value };
                  setVariants(c);
                }}
                className="input"
              />
              <input
                type="number"
                placeholder="+/- giá"
                value={v.priceDelta ?? 0}
                onChange={(e) => {
                  const c = [...variants];
                  c[i] = { ...v, priceDelta: +e.target.value };
                  setVariants(c);
                }}
                className="input"
              />
              <button
                type="button"
                onClick={() => setVariants((p) => p.filter((_, j) => j !== i))}
                className="p-2 text-red-600 hover:bg-red-50 rounded"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            </div>
          ))}
        </div>
      </section>

      {tryOnEnabled && (
        <section>
          <div className="flex items-center justify-between mb-2">
            <h3 className="font-semibold">Ảnh CLOTH (cho thử đồ ảo)</h3>
            <label className="inline-flex items-center gap-1.5 text-sm px-3 py-1.5 border border-gray-200 rounded cursor-pointer hover:bg-gray-50">
              <Upload className="w-3.5 h-3.5" /> Tải ảnh CLOTH
              <input type="file" accept="image/*" onChange={onUploadTryon} className="hidden" />
            </label>
          </div>
          <div className="grid grid-cols-4 gap-2">
            {tryonAssets.map((t, i) => (
              <div key={i} className="relative group">
                <img
                  src={t.clothImageUrl}
                  alt=""
                  className="w-full aspect-square object-cover rounded border"
                />
                <button
                  type="button"
                  onClick={() => setTryonAssets((p) => p.filter((_, j) => j !== i))}
                  className="absolute top-1 right-1 p-1 bg-red-600 text-white rounded opacity-0 group-hover:opacity-100"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>
            ))}
            {tryonAssets.length === 0 && (
              <div className="col-span-4 text-sm text-gray-500 border border-dashed rounded p-4 text-center">
                Chưa có ảnh CLOTH
              </div>
            )}
          </div>
        </section>
      )}

      <div className="flex justify-end gap-2 pt-2 border-t">
        <button
          type="button"
          onClick={() => navigate('/seller/products')}
          className="px-4 py-2 border border-gray-200 rounded text-sm hover:bg-gray-50"
        >
          Huỷ
        </button>
        <button
          type="submit"
          disabled={saving}
          className="px-4 py-2 bg-[var(--color-ink)] text-white rounded text-sm hover:opacity-90 disabled:opacity-50"
        >
          {saving ? 'Đang lưu...' : isEdit ? 'Cập nhật' : 'Tạo sản phẩm'}
        </button>
      </div>
    </form>
  );
}
