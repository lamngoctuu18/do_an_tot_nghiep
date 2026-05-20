import { Controller, Get } from '@nestjs/common';
import { ApiTags } from '@nestjs/swagger';

@ApiTags('Health')
@Controller()
export class HealthController {
  @Get('healthz')
  health() {
    return { ok: true, service: 'vton-shop-api', ts: Date.now() };
  }
}
