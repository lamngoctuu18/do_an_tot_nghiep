import { Logger } from '@nestjs/common';
import * as mysql from 'mysql2/promise';

/**
 * Tự động `CREATE DATABASE IF NOT EXISTS` trước khi TypeORM kết nối.
 * Bảng sẽ được tự sinh do `synchronize: true` ở dev.
 */
export async function ensureDatabase(opts: {
  host: string;
  port: number;
  user: string;
  password: string;
  database: string;
}) {
  const log = new Logger('DBBootstrap');
  const conn = await mysql.createConnection({
    host: opts.host,
    port: opts.port,
    user: opts.user,
    password: opts.password,
    multipleStatements: true,
  });
  try {
    await conn.query(
      `CREATE DATABASE IF NOT EXISTS \`${opts.database}\` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;`,
    );
    log.log(`Database \`${opts.database}\` ready @ ${opts.host}:${opts.port}`);
  } finally {
    await conn.end();
  }
}
