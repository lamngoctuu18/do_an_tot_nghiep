import {
  Body,
  Controller,
  Delete,
  Get,
  Post,
  Query,
  Req,
  Sse,
  UseGuards,
} from '@nestjs/common';
import { ApiBearerAuth, ApiTags } from '@nestjs/swagger';
import { Observable } from 'rxjs';
import { OptionalJwtAuthGuard } from '../auth/optional-jwt.guard';
import { ChatbotService } from './chatbot.service';
import { ChatMessageDto } from './dto/chat-message.dto';

@ApiTags('Chatbot')
@ApiBearerAuth('JWT')
@UseGuards(OptionalJwtAuthGuard)
@Controller('chatbot')
export class ChatbotController {
  constructor(private readonly svc: ChatbotService) {}

  @Post('messages')
  async send(@Body() dto: ChatMessageDto, @Req() req: any) {
    const userId = req.user?.id;
    const sessionId = dto.sessionId || this.fallbackSession(req);
    const reply = await this.svc.handleMessage(dto.message, sessionId, userId);
    return { sessionId, reply };
  }

  @Sse('messages/stream')
  stream(
    @Query('message') message: string,
    @Query('sessionId') sessionId: string,
    @Req() req: any,
  ): Observable<MessageEvent> {
    const userId = req.user?.id;
    const sid = sessionId || this.fallbackSession(req);
    return this.svc.streamMessage(message, sid, userId);
  }

  @Get('history')
  async history(@Query('sessionId') sessionId: string, @Req() req: any) {
    const userId = req.user?.id;
    const list = await this.svc.listHistory(sessionId, userId);
    return { messages: list };
  }

  @Delete('history')
  async clear(@Query('sessionId') sessionId: string, @Req() req: any) {
    const userId = req.user?.id;
    return this.svc.clearHistory(sessionId, userId);
  }

  private fallbackSession(req: any): string {
    const ip = req.ip || req.connection?.remoteAddress || 'anon';
    return `anon-${Buffer.from(ip).toString('hex').slice(0, 16)}`;
  }
}
