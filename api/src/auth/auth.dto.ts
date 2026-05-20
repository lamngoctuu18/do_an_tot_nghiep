import { IsEmail, IsOptional, IsString, MinLength } from 'class-validator';

export class RegisterDto {
  @IsEmail() email: string;
  @MinLength(6) password: string;
  @IsString() fullName: string;
  @IsOptional() @IsString() phone?: string;
}

export class LoginDto {
  @IsEmail() email: string;
  @IsString() password: string;
}

export class RefreshDto {
  @IsString() refreshToken: string;
}
