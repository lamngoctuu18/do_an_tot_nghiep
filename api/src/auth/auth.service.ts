import {
  ConflictException,
  ForbiddenException,
  Injectable,
  UnauthorizedException,
} from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { JwtService } from '@nestjs/jwt';
import { ConfigService } from '@nestjs/config';
import * as bcrypt from 'bcryptjs';
import { v4 as uuid } from 'uuid';
import { User } from '../users/user.entity';
import { Cart } from '../cart/cart.entity';
import { RefreshToken } from './refresh-token.entity';
import { LoginDto, RegisterDto } from './auth.dto';
import { UserRole, UserStatus } from '../common/enums';

@Injectable()
export class AuthService {
  constructor(
    @InjectRepository(User) private users: Repository<User>,
    @InjectRepository(Cart) private carts: Repository<Cart>,
    @InjectRepository(RefreshToken) private refreshTokens: Repository<RefreshToken>,
    private jwt: JwtService,
    private cs: ConfigService,
  ) {}

  async register(dto: RegisterDto) {
    const exists = await this.users.findOne({ where: { email: dto.email } });
    if (exists) throw new ConflictException('Email đã được sử dụng');
    const user = this.users.create({
      email: dto.email,
      passwordHash: await bcrypt.hash(dto.password, 10),
      fullName: dto.fullName,
      phone: dto.phone,
      role: UserRole.CUSTOMER,
    });
    await this.users.save(user);
    await this.carts.save(this.carts.create({ user }));
    return this.issueTokens(user);
  }

  async login(dto: LoginDto) {
    const user = await this.users
      .createQueryBuilder('u')
      .addSelect('u.passwordHash')
      .where('u.email = :email', { email: dto.email })
      .getOne();
    if (!user || !(await bcrypt.compare(dto.password, user.passwordHash))) {
      throw new UnauthorizedException('Email hoặc mật khẩu không đúng');
    }
    if (user.status === UserStatus.LOCKED) {
      throw new ForbiddenException('Tài khoản đã bị khóa');
    }
    return this.issueTokens(user);
  }

  async logout(userId: number, refreshToken?: string) {
    if (refreshToken) {
      await this.refreshTokens.update(
        { token: refreshToken, user: { id: userId } as any },
        { revoked: true },
      );
    } else {
      await this.refreshTokens.update(
        { user: { id: userId } as any, revoked: false },
        { revoked: true },
      );
    }
    return { success: true };
  }

  async refresh(refreshToken: string) {
    const row = await this.refreshTokens.findOne({
      where: { token: refreshToken, revoked: false },
      relations: ['user'],
    });
    if (!row || row.expiresAt < new Date()) {
      throw new UnauthorizedException('Refresh token không hợp lệ');
    }
    row.revoked = true;
    await this.refreshTokens.save(row);
    return this.issueTokens(row.user);
  }

  private async issueTokens(user: User) {
    const payload = { sub: user.id, email: user.email, role: user.role };
    const accessTtl = +this.cs.get('JWT_ACCESS_TTL', 900);
    const refreshTtl = +this.cs.get('JWT_REFRESH_TTL', 2592000);
    const accessToken = await this.jwt.signAsync(payload, {
      secret: this.cs.get('JWT_SECRET'),
      expiresIn: accessTtl,
    });
    const refreshToken = uuid().replace(/-/g, '') + uuid().replace(/-/g, '');
    await this.refreshTokens.save(
      this.refreshTokens.create({
        user,
        token: refreshToken,
        expiresAt: new Date(Date.now() + refreshTtl * 1000),
      }),
    );
    return {
      accessToken,
      refreshToken,
      user: {
        id: user.id,
        email: user.email,
        fullName: user.fullName,
        role: user.role,
        avatarUrl: user.avatarUrl,
      },
    };
  }
}
