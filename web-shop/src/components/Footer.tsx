import { Shirt } from "lucide-react";

export default function Footer() {
  return (
    <footer className="bg-surface border-t border-white/10 mt-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div>
            <div className="flex items-center gap-2 mb-4">
              <Shirt className="w-6 h-6 text-primary" />
              <span className="text-lg font-bold">VTON Shop</span>
            </div>
            <p className="text-sm text-gray-400">
              Cửa hàng thời trang trực tuyến tích hợp công nghệ AI thử đồ ảo.
              Trải nghiệm mua sắm thông minh, tiện lợi.
            </p>
          </div>
          <div>
            <h3 className="font-semibold mb-4">Liên kết</h3>
            <ul className="space-y-2 text-sm text-gray-400">
              <li><a href="/" className="hover:text-white transition-colors">Trang chủ</a></li>
              <li><a href="/try-on" className="hover:text-white transition-colors">Thử đồ AI</a></li>
              <li><a href="/cart" className="hover:text-white transition-colors">Giỏ hàng</a></li>
            </ul>
          </div>
          <div>
            <h3 className="font-semibold mb-4">Công nghệ</h3>
            <ul className="space-y-2 text-sm text-gray-400">
              <li>TPS Warping + Affine Alignment</li>
              <li>U2Net Cloth Segmentation</li>
              <li>SegFormer Human Parsing</li>
              <li>Masked Diffusion Refinement</li>
            </ul>
          </div>
        </div>
        <div className="border-t border-white/10 mt-8 pt-8 text-center text-sm text-gray-500">
          © 2026 VTON Shop — Đồ án tốt nghiệp. All rights reserved.
        </div>
      </div>
    </footer>
  );
}
