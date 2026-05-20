import { Injectable, Logger, NotFoundException, UnauthorizedException } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import axios from 'axios';
import { Observable } from 'rxjs';
import { ChatMessage, ChatRole } from './chat-message.entity';
import { Product, ProductStatus } from '../catalog/product.entity';
import { Order } from '../orders/order.entity';

export type BotReply = {
  text?: string;
  products?: any[];
  order?: any;
  tryOnSteps?: { title: string; description: string }[];
  quickReplies?: { label: string; value: string }[];
  requireLogin?: boolean;
};

type Intent =
  | 'greet'
  | 'size'
  | 'suggest'
  | 'tryon'
  | 'order'
  | 'policy'
  | 'fallback';

@Injectable()
export class ChatbotService {
  private readonly logger = new Logger(ChatbotService.name);

  constructor(
    @InjectRepository(ChatMessage)
    private readonly repo: Repository<ChatMessage>,
    @InjectRepository(Product)
    private readonly products: Repository<Product>,
    @InjectRepository(Order)
    private readonly orders: Repository<Order>,
    private readonly config: ConfigService,
  ) {}

  // -------- history --------
  async listHistory(sessionId: string, userId?: number) {
    const qb = this.repo
      .createQueryBuilder('m')
      .where('m.sessionId = :sessionId', { sessionId })
      .orderBy('m.id', 'ASC')
      .limit(200);
    if (userId) qb.andWhere('(m.userId = :uid OR m.userId IS NULL)', { uid: userId });
    return qb.getMany();
  }

  async clearHistory(sessionId: string, userId?: number) {
    const qb = this.repo
      .createQueryBuilder()
      .delete()
      .from(ChatMessage)
      .where('session_id = :sessionId', { sessionId });
    if (userId) qb.andWhere('(user_id = :uid OR user_id IS NULL)', { uid: userId });
    await qb.execute();
    return { success: true };
  }

  // -------- core dispatch --------
  async handleMessage(message: string, sessionId: string, userId?: number): Promise<BotReply> {
    await this.persist(sessionId, userId, ChatRole.USER, message, null);
    const intent = this.detectIntent(message);
    let reply: BotReply;

    switch (intent) {
      case 'greet':
        reply = this.greet();
        break;
      case 'size':
        reply = this.sizeAdvice(message);
        break;
      case 'suggest':
        reply = await this.suggestProducts(message);
        break;
      case 'tryon':
        reply = this.tryOnGuide();
        break;
      case 'order':
        reply = await this.checkOrder(message, userId);
        break;
      case 'policy':
        reply = this.policy();
        break;
      default:
        reply = await this.llmFallback(message, sessionId, userId);
    }

    await this.persist(sessionId, userId, ChatRole.BOT, reply.text || '', {
      products: reply.products,
      order: reply.order,
      tryOnSteps: reply.tryOnSteps,
      quickReplies: reply.quickReplies,
    });
    return reply;
  }

  // -------- streaming (SSE) --------
  streamMessage(message: string, sessionId: string, userId?: number): Observable<MessageEvent> {
    return new Observable<MessageEvent>((subscriber) => {
      let cancelled = false;

      (async () => {
        try {
          await this.persist(sessionId, userId, ChatRole.USER, message, null);
          const intent = this.detectIntent(message);

          // Non-text payloads come whole; only text streams.
          let textPart = '';
          let attachments: BotReply = {};
          if (intent === 'greet') attachments = this.greet();
          else if (intent === 'size') attachments = this.sizeAdvice(message);
          else if (intent === 'suggest') attachments = await this.suggestProducts(message);
          else if (intent === 'tryon') attachments = this.tryOnGuide();
          else if (intent === 'order') attachments = await this.checkOrder(message, userId);
          else if (intent === 'policy') attachments = this.policy();
          else attachments = await this.llmFallback(message, sessionId, userId);

          textPart = attachments.text || '';
          const baseAttachments = { ...attachments };
          delete baseAttachments.text;

          // stream text character by character (chunks of ~3 chars)
          subscriber.next({ data: { type: 'start' } } as MessageEvent);
          for (let i = 0; i < textPart.length; i += 3) {
            if (cancelled) return;
            const chunk = textPart.slice(i, i + 3);
            subscriber.next({ data: { type: 'delta', content: chunk } } as MessageEvent);
            await new Promise((r) => setTimeout(r, 24));
          }
          subscriber.next({ data: { type: 'attachments', payload: baseAttachments } } as MessageEvent);
          subscriber.next({ data: { type: 'done' } } as MessageEvent);

          await this.persist(sessionId, userId, ChatRole.BOT, textPart, baseAttachments);
          subscriber.complete();
        } catch (e: any) {
          this.logger.error(`stream error: ${e?.message}`);
          subscriber.next({ data: { type: 'error', message: e?.message || 'error' } } as MessageEvent);
          subscriber.complete();
        }
      })();

      return () => {
        cancelled = true;
      };
    });
  }

  // -------- intent detection --------
  private detectIntent(raw: string): Intent {
    const t = raw.toLowerCase().trim();
    if (/^qa:size\b/.test(t) || /\b(size|cỡ|kích thước|cao|nặng|kg|cm)\b/.test(t)) return 'size';
    if (/^qa:suggest\b/.test(t) || /\b(gợi ý|gioi y|outfit|phối|sản phẩm|recommend|đẹp|nên mua)\b/.test(t))
      return 'suggest';
    if (/^qa:tryon\b/.test(t) || /\b(thử đồ|try.?on|virtual|ảo|thử)\b/.test(t)) return 'tryon';
    if (/^qa:order\b/.test(t) || /\b(đơn hàng|order|mã đơn|ord-|tracking|vận chuyển)\b/.test(t))
      return 'order';
    if (/^qa:policy\b/.test(t) || /\b(đổi trả|hoàn tiền|policy|return|refund|chính sách)\b/.test(t))
      return 'policy';
    if (/^(xin chào|chào|hello|hi|hey)\b/.test(t)) return 'greet';
    return 'fallback';
  }

  // -------- handlers --------
  private greet(): BotReply {
    return {
      text: 'Xin chào! Tôi là Trợ lý thời trang AI. Tôi giúp bạn được gì hôm nay?',
      quickReplies: [
        { label: 'Tư vấn size', value: 'qa:size' },
        { label: 'Gợi ý sản phẩm', value: 'qa:suggest' },
        { label: 'Hướng dẫn thử đồ', value: 'qa:tryon' },
        { label: 'Kiểm tra đơn hàng', value: 'qa:order' },
        { label: 'Chính sách đổi trả', value: 'qa:policy' },
      ],
    };
  }

  private sizeAdvice(message: string): BotReply {
    const heightMatch = message.match(/(\d{2,3})\s*(?:cm|m)?/);
    const weightMatch = message.match(/(\d{2,3})\s*kg/);
    if (heightMatch && weightMatch) {
      const h = parseInt(heightMatch[1], 10);
      const w = parseInt(weightMatch[1], 10);
      let size = 'M';
      if (w < 50) size = 'S';
      else if (w < 65) size = 'M';
      else if (w < 75) size = 'L';
      else size = 'XL';
      return {
        text: `Với chiều cao ${h}cm và cân nặng ${w}kg, tôi gợi ý size **${size}**. Bạn có thể dùng tính năng Thử đồ ảo để xem dáng thực tế trước khi mua.`,
        quickReplies: [{ label: 'Hướng dẫn thử đồ', value: 'qa:tryon' }],
      };
    }
    return {
      text: 'Cho tôi biết chiều cao (cm) và cân nặng (kg) của bạn nhé. Ví dụ: 170cm, 65kg.',
    };
  }

  private async suggestProducts(query: string): Promise<BotReply> {
    const q = query.toLowerCase();
    const qb = this.products
      .createQueryBuilder('p')
      .leftJoinAndSelect('p.images', 'img')
      .where('p.status = :st', { st: ProductStatus.ACTIVE })
      .orderBy('p.salesCount', 'DESC')
      .addOrderBy('p.ratingAvg', 'DESC')
      .take(4);

    // keyword filtering — very lightweight
    const keywords = ['áo', 'quần', 'váy', 'shorts', 'jeans', 'sơ mi', 'thun', 'jacket', 'đầm'];
    const matched = keywords.find((k) => q.includes(k));
    if (matched) qb.andWhere('LOWER(p.name) LIKE :kw', { kw: `%${matched}%` });

    const rows = await qb.getMany();
    if (!rows.length) {
      return { text: 'Hiện chưa có sản phẩm phù hợp. Bạn thử mô tả cụ thể hơn nhé.' };
    }

    return {
      text: 'Một vài sản phẩm bạn có thể thích:',
      products: rows.map((p) => ({
        id: p.id,
        name: p.name,
        price: parseFloat(p.price),
        image: p.images?.sort((a, b) => a.position - b.position)?.[0]?.url || '',
        reason: p.badge || (matched ? `Phù hợp với "${matched}"` : 'Bán chạy'),
      })),
    };
  }

  private tryOnGuide(): BotReply {
    return {
      text: 'Hướng dẫn thử đồ ảo trong 3 bước:',
      tryOnSteps: [
        { title: '1. Chọn sản phẩm', description: 'Vào trang sản phẩm và nhấn nút Thử đồ.' },
        { title: '2. Tải ảnh', description: 'Chụp đứng thẳng, ánh sáng đều, mặc đồ ôm.' },
        { title: '3. Xem kết quả', description: 'AI ghép trang phục lên ảnh trong vài giây.' },
      ],
    };
  }

  private async checkOrder(message: string, userId?: number): Promise<BotReply> {
    if (!userId) {
      return {
        text: 'Bạn cần đăng nhập để kiểm tra đơn hàng. Vui lòng đăng nhập rồi quay lại nhé.',
        requireLogin: true,
      };
    }
    const codeMatch = message.match(/(ord[-_]?[a-z0-9-]+)/i);
    if (codeMatch) {
      const code = codeMatch[1].toUpperCase();
      const order = await this.orders.findOne({
        where: { code, user: { id: userId } },
        relations: ['items'],
      });
      if (!order) return { text: `Không tìm thấy đơn hàng ${code} trong tài khoản của bạn.` };
      return {
        text: `Đơn ${order.code} đang ở trạng thái ${order.status}.`,
        order: {
          code: order.code,
          status: order.status,
          total: parseFloat(order.total),
          createdAt: order.placedAt?.toISOString().slice(0, 10),
          itemsCount: order.items?.length || 0,
        },
      };
    }

    const recent = await this.orders.find({
      where: { user: { id: userId } },
      relations: ['items'],
      order: { placedAt: 'DESC' },
      take: 3,
    });
    if (!recent.length) return { text: 'Bạn chưa có đơn hàng nào.' };
    return {
      text: 'Đây là các đơn gần đây của bạn:',
      order: {
        code: recent[0].code,
        status: recent[0].status,
        total: parseFloat(recent[0].total),
        createdAt: recent[0].placedAt?.toISOString().slice(0, 10),
        itemsCount: recent[0].items?.length || 0,
      },
    };
  }

  private policy(): BotReply {
    return {
      text:
        'Chính sách đổi trả: miễn phí trong 7 ngày kể từ khi nhận hàng, sản phẩm còn nguyên tem và chưa qua sử dụng. Hoàn tiền 3-5 ngày làm việc qua phương thức bạn đã thanh toán.',
    };
  }

  // -------- LLM fallback (OpenAI compatible) --------
  private async llmFallback(message: string, sessionId: string, userId?: number): Promise<BotReply> {
    const apiKey = this.config.get<string>('OPENAI_API_KEY');
    const baseUrl =
      this.config.get<string>('OPENAI_BASE_URL') || 'https://api.openai.com/v1';
    const model = this.config.get<string>('OPENAI_MODEL') || 'gpt-4o-mini';

    if (!apiKey) {
      return {
        text:
          'Cảm ơn bạn! Tôi có thể tư vấn nhanh các chủ đề bên dưới. Bạn chọn một mục hoặc mô tả rõ hơn giúp tôi nhé.',
        quickReplies: [
          { label: 'Tư vấn size', value: 'qa:size' },
          { label: 'Gợi ý sản phẩm', value: 'qa:suggest' },
          { label: 'Hướng dẫn thử đồ', value: 'qa:tryon' },
        ],
      };
    }

    try {
      const recent = await this.repo
        .createQueryBuilder('m')
        .where('m.sessionId = :sessionId', { sessionId })
        .orderBy('m.id', 'DESC')
        .limit(10)
        .getMany();
      const history = recent.reverse().map((m) => ({
        role: m.role === ChatRole.USER ? 'user' : 'assistant',
        content: m.content,
      }));

      const sys = {
        role: 'system',
        content:
          'Bạn là trợ lý thời trang AI cho cửa hàng quần áo Việt Nam. Trả lời ngắn gọn (<= 80 từ), thân thiện, bằng tiếng Việt. Chỉ tư vấn về thời trang, size, thử đồ ảo, đơn hàng, đổi trả.',
      };

      const { data } = await axios.post(
        `${baseUrl}/chat/completions`,
        {
          model,
          messages: [sys, ...history, { role: 'user', content: message }],
          temperature: 0.6,
          max_tokens: 220,
        },
        {
          headers: { Authorization: `Bearer ${apiKey}` },
          timeout: 15000,
        },
      );
      const text = data?.choices?.[0]?.message?.content?.trim() || '';
      if (!text) throw new Error('Empty LLM reply');
      return { text };
    } catch (e: any) {
      this.logger.warn(`LLM fallback failed: ${e?.message}`);
      return {
        text: 'Xin lỗi, tôi chưa hiểu rõ câu hỏi. Bạn thử diễn đạt khác hoặc chọn một chủ đề bên dưới nhé.',
        quickReplies: [
          { label: 'Tư vấn size', value: 'qa:size' },
          { label: 'Gợi ý sản phẩm', value: 'qa:suggest' },
        ],
      };
    }
  }

  // -------- persistence --------
  private async persist(
    sessionId: string,
    userId: number | undefined,
    role: ChatRole,
    content: string,
    metadata: any,
  ) {
    const row = this.repo.create({
      sessionId,
      role,
      content,
      metadata: metadata || null,
      user: userId ? ({ id: userId } as any) : null,
    });
    await this.repo.save(row);
  }
}
