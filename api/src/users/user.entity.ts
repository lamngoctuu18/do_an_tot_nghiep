import {
  Column,
  CreateDateColumn,
  Entity,
  OneToMany,
  OneToOne,
  PrimaryGeneratedColumn,
  UpdateDateColumn,
} from 'typeorm';
import { UserRole, UserStatus } from '../common/enums';
import { Address } from './address.entity';
import { Seller } from '../sellers/seller.entity';

export enum Gender {
  MALE = 'MALE',
  FEMALE = 'FEMALE',
  OTHER = 'OTHER',
}

@Entity('users')
export class User {
  @PrimaryGeneratedColumn() id: number;
  @Column({ unique: true, length: 191 }) email: string;
  @Column({ name: 'password_hash', select: false }) passwordHash: string;
  @Column({ name: 'full_name', length: 120 }) fullName: string;
  @Column({ length: 32, nullable: true }) phone: string;
  @Column({ name: 'avatar_url', length: 500, nullable: true }) avatarUrl: string;
  @Column({ type: 'enum', enum: Gender, nullable: true }) gender: Gender;
  @Column({ name: 'date_of_birth', type: 'date', nullable: true }) dateOfBirth: Date;
  @Column({ type: 'enum', enum: UserRole, default: UserRole.CUSTOMER }) role: UserRole;
  @Column({ type: 'enum', enum: UserStatus, default: UserStatus.ACTIVE })
  status: UserStatus;
  @CreateDateColumn({ name: 'created_at' }) createdAt: Date;
  @UpdateDateColumn({ name: 'updated_at' }) updatedAt: Date;

  @OneToMany(() => Address, (a) => a.user) addresses: Address[];
  @OneToOne(() => Seller, (s) => s.user) seller: Seller;
}
