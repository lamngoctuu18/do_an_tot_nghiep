import {
  Controller,
  Get,
  Param,
  ParseBoolPipe,
  ParseIntPipe,
  Patch,
  Query,
  UseGuards,
} from '@nestjs/common';
import { ApiBearerAuth, ApiTags } from '@nestjs/swagger';
import { JwtAuthGuard } from '../auth/jwt-auth.guard';
import { CurrentUser } from '../common/decorators/current-user.decorator';
import { NotificationsService } from './notifications.service';

@ApiTags('Notifications')
@ApiBearerAuth('JWT')
@UseGuards(JwtAuthGuard)
@Controller('notifications')
export class NotificationsController {
  constructor(private svc: NotificationsService) {}

  @Get()
  list(
    @CurrentUser() u: any,
    @Query('is_read', new ParseBoolPipe({ optional: true } as any)) isRead?: boolean,
  ) {
    return this.svc.list(u.id, isRead);
  }

  @Patch(':id/read')
  read(@CurrentUser() u: any, @Param('id', ParseIntPipe) id: number) {
    return this.svc.markRead(u.id, id);
  }

  @Patch('read-all')
  readAll(@CurrentUser() u: any) {
    return this.svc.markAllRead(u.id);
  }
}
