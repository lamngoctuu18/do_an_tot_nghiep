export interface Product {
  id: number;
  name: string;
  price: number;
  originalPrice?: number;
  image: string;
  category: string;
  description: string;
  sizes: string[];
  colors: string[];
  rating: number;
  reviews: number;
  badge?: string;
}

export const products: Product[] = [
  {
    id: 1,
    name: "Áo Thun Trắng Basic",
    price: 199000,
    originalPrice: 299000,
    image: "https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?w=500",
    category: "Áo thun",
    description: "Áo thun trắng basic chất liệu cotton 100%, phom dáng regular fit thoải mái. Phù hợp mặc hàng ngày hoặc layer cùng áo khoác.",
    sizes: ["S", "M", "L", "XL"],
    colors: ["Trắng", "Đen", "Xám"],
    rating: 4.8,
    reviews: 234,
    badge: "Bán chạy",
  },
  {
    id: 2,
    name: "Áo Sơ Mi Oxford",
    price: 450000,
    image: "https://images.unsplash.com/photo-1596755094514-f87e34085b2c?w=500",
    category: "Áo sơ mi",
    description: "Áo sơ mi Oxford vải dày dặn, mềm mịn. Thiết kế cổ button-down lịch sự, phù hợp đi làm hoặc dạo phố.",
    sizes: ["S", "M", "L", "XL", "XXL"],
    colors: ["Xanh nhạt", "Trắng", "Hồng nhạt"],
    rating: 4.6,
    reviews: 128,
  },
  {
    id: 3,
    name: "Áo Hoodie Oversize",
    price: 550000,
    originalPrice: 750000,
    image: "https://images.unsplash.com/photo-1556821840-3a63f95609a7?w=500",
    category: "Áo hoodie",
    description: "Hoodie oversize chất nỉ bông dày dặn, giữ ấm tốt. Có mũ trùm đầu và túi kangaroo phía trước.",
    sizes: ["M", "L", "XL"],
    colors: ["Đen", "Xám", "Be"],
    rating: 4.9,
    reviews: 567,
    badge: "Mới",
  },
  {
    id: 4,
    name: "Áo Polo Classic",
    price: 350000,
    image: "https://images.unsplash.com/photo-1625910513413-5fc69d80b841?w=500",
    category: "Áo polo",
    description: "Áo polo cổ bẻ classic với chất liệu cotton pique co giãn nhẹ. Phom smart casual lịch lãm.",
    sizes: ["S", "M", "L", "XL"],
    colors: ["Navy", "Trắng", "Đỏ đô"],
    rating: 4.5,
    reviews: 89,
  },
  {
    id: 5,
    name: "Áo Khoác Bomber",
    price: 890000,
    originalPrice: 1200000,
    image: "https://images.unsplash.com/photo-1591047139829-d91aecb6caea?w=500",
    category: "Áo khoác",
    description: "Áo khoác bomber phong cách streetwear. Chất liệu polyester chống gió nhẹ, lót lưới thoáng mát.",
    sizes: ["M", "L", "XL"],
    colors: ["Đen", "Xanh rêu", "Navy"],
    rating: 4.7,
    reviews: 312,
    badge: "Hot",
  },
  {
    id: 6,
    name: "Áo Len Cổ Lọ",
    price: 480000,
    image: "https://images.unsplash.com/photo-1576566588028-4147f3842f27?w=500",
    category: "Áo len",
    description: "Áo len cổ lọ chất len mịn cao cấp, giữ ấm tốt. Thiết kế tối giản, dễ phối đồ cho mùa đông.",
    sizes: ["S", "M", "L"],
    colors: ["Đen", "Nâu", "Kem"],
    rating: 4.4,
    reviews: 76,
  },
  {
    id: 7,
    name: "Áo Dài Tay Stripe",
    price: 280000,
    image: "https://images.unsplash.com/photo-1618354691373-d851c5c3a990?w=500",
    category: "Áo thun",
    description: "Áo thun dài tay họa tiết sọc ngang, chất cotton pha spandex co giãn thoải mái. Layer hoàn hảo cho mùa thu.",
    sizes: ["S", "M", "L", "XL"],
    colors: ["Đen/Trắng", "Navy/Trắng"],
    rating: 4.3,
    reviews: 145,
  },
  {
    id: 8,
    name: "Áo Blazer Slim Fit",
    price: 1290000,
    originalPrice: 1590000,
    image: "https://images.unsplash.com/photo-1507679799987-c73779587ccf?w=500",
    category: "Áo khoác",
    description: "Blazer slim fit lịch lãm, vải tweed cao cấp. Phù hợp cho các sự kiện trang trọng hoặc phong cách smart casual.",
    sizes: ["S", "M", "L", "XL"],
    colors: ["Đen", "Xám đậm", "Navy"],
    rating: 4.8,
    reviews: 203,
    badge: "Premium",
  },
];

export const categories = [
  "Tất cả",
  "Áo thun",
  "Áo sơ mi",
  "Áo hoodie",
  "Áo polo",
  "Áo khoác",
  "Áo len",
];

export function formatPrice(price: number): string {
  return new Intl.NumberFormat("vi-VN", {
    style: "currency",
    currency: "VND",
  }).format(price);
}
