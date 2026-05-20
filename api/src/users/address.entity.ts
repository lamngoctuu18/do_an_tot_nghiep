import {
  Column,
  Entity,
  ManyToOne,
  PrimaryGeneratedColumn,
} from 'typeorm';
import { User } from './user.entity';

@Entity('addresses')
export class Address {
  @PrimaryGeneratedColumn() id: number;
  @ManyToOne(() => User, (u) => u.addresses, { onDelete: 'CASCADE' })
  user: User;
  @Column({ length: 120 }) recipient: string;
  @Column({ length: 32 }) phone: string;
  @Column({ length: 255 }) line1: string;
  @Column({ length: 80 }) ward: string;
  @Column({ length: 80 }) district: string;
  @Column({ length: 80 }) city: string;
  @Column({ name: 'is_default', default: false }) isDefault: boolean;
}
