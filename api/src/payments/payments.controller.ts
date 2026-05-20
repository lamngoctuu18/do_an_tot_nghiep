import { Controller, Get, Query, NotFoundException, Res } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { Response } from 'express';
import { Order, OrderStatus } from '../orders/order.entity';
import { ApiTags } from '@nestjs/swagger';

@ApiTags('Payments')
@Controller('payments')
export class PaymentsController {
  constructor(@InjectRepository(Order) private orders: Repository<Order>) {}

  @Get('vnpay/redirect')
  redirect(@Query('code') code: string, @Res() res: Response) {
    const html = `<!doctype html><html><body style="font-family:system-ui;padding:40px;text-align:center">
<h2>VNPay (Sandbox stub)</h2>
<p>Mã đơn: <b>${code}</b></p>
<form method="GET" action="/api/payments/vnpay/return">
  <input type="hidden" name="code" value="${code}" />
  <input type="hidden" name="vnp_ResponseCode" value="00" />
  <button style="padding:10px 24px;background:#000;color:#fff;border:0;border-radius:8px;cursor:pointer">
    Thanh toán thành công (mock)
  </button>
</form></body></html>`;
    res.type('html').send(html);
  }

  @Get('vnpay/return')
  async returnUrl(
    @Query('code') code: string,
    @Query('vnp_ResponseCode') rc: string,
  ) {
    const order = await this.orders.findOne({ where: { code } });
    if (!order) throw new NotFoundException('Đơn không tồn tại');
    if (rc === '00') {
      order.status = OrderStatus.PAID;
      await this.orders.save(order);
      return { ok: true, code, status: order.status };
    }
    return { ok: false, code, status: order.status };
  }
}
