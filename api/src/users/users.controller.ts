import { Body, Controller, Get, Patch, Put, UseGuards } from '@nestjs/common';
import { ApiBearerAuth, ApiTags } from '@nestjs/swagger';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { User, Gender } from './user.entity';
import { JwtAuthGuard } from '../auth/jwt-auth.guard';
import { CurrentUser } from '../common/decorators/current-user.decorator';
import { IsDateString, IsEnum, IsOptional, IsString, MinLength } from 'class-validator';

class UpdateProfileDto {
  @IsOptional() @IsString() @MinLength(2) fullName?: string;
  @IsOptional() @IsString() phone?: string;
  @IsOptional() @IsString() avatarUrl?: string;
  @IsOptional() @IsEnum(Gender) gender?: Gender;
  @IsOptional() @IsDateString() dateOfBirth?: string;
}

@ApiTags('Users')
@ApiBearerAuth('JWT')
@UseGuards(JwtAuthGuard)
@Controller('users')
export class UsersController {
  constructor(@InjectRepository(User) private users: Repository<User>) {}

  @Get('me')
  async me(@CurrentUser() u: any) {
    return this.users.findOne({ where: { id: u.id }, relations: ['addresses'] });
  }

  @Patch('me')
  async patch(@CurrentUser() u: any, @Body() dto: UpdateProfileDto) {
    return this.update(u, dto);
  }

  @Put('me')
  async update(@CurrentUser() u: any, @Body() dto: UpdateProfileDto) {
    const payload: any = { ...dto };
    if (dto.dateOfBirth) payload.dateOfBirth = new Date(dto.dateOfBirth);
    await this.users.update(u.id, payload);
    return this.users.findOne({ where: { id: u.id } });
  }
}
