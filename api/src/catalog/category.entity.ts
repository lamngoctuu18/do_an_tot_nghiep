import { Column, Entity, Index, ManyToOne, PrimaryGeneratedColumn } from 'typeorm';

@Entity('categories')
@Index(['slug'], { unique: true })
export class Category {
  @PrimaryGeneratedColumn() id: number;
  @ManyToOne(() => Category, { nullable: true, onDelete: 'SET NULL' }) parent: Category;
  @Column({ length: 120 }) name: string;
  @Column({ length: 140 }) slug: string;
  @Column({ name: 'image_url', length: 500, nullable: true }) imageUrl: string;
}
