import { IsOptional, IsString, MaxLength, MinLength } from 'class-validator';

export class ChatMessageDto {
  @IsString()
  @MinLength(1)
  @MaxLength(2000)
  message: string;

  @IsOptional()
  @IsString()
  @MaxLength(64)
  sessionId?: string;
}
