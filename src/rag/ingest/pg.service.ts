import { Injectable, OnModuleDestroy, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { Pool } from 'pg';
import { Signer } from '@aws-sdk/rds-signer';
import * as fs from 'fs';
import * as path from 'path';

export type DbName = 'lead' | 'boq';

/**
 * Single connection pool shared by both 'lead' and 'boq' sources.
 * Uses AWS RDS IAM Authentication via @aws-sdk/rds-signer.
 * DB_USERNAME (lead-service) has SELECT on both ls_* and bs_* tables.
 */
@Injectable()
export class PgService implements OnModuleDestroy {
  private readonly logger = new Logger(PgService.name);
  private pool: Pool | null = null;

  constructor(private readonly config: ConfigService) {}

  private getPool(): Pool {
    if (this.pool) return this.pool;

    const host = this.config.get<string>('DB_HOST')!;
    const port = Number(this.config.get('DB_PORT') || 5432);
    const username = this.config.get<string>('DB_USERNAME')!;
    const database = this.config.get<string>('DB_NAME')!;
    const region = this.config.get<string>('DB_REGION') || 'ap-south-1';
    const certPath = this.config.get<string>('DB_CERT');

    const certExists = certPath && fs.existsSync(path.resolve(certPath));
    const ssl = certExists
      ? { ca: fs.readFileSync(path.resolve(certPath!)).toString() }
      : { rejectUnauthorized: false };

    const accessKeyId = this.config.get<string>('AWS_ACCESS_KEY_ID');
    const secretAccessKey = this.config.get<string>('AWS_SECRET_ACCESS_KEY');
    const credentials =
      accessKeyId && secretAccessKey ? { accessKeyId, secretAccessKey } : undefined;

    const signer = new Signer({ hostname: host, port, username, region, credentials });

    this.logger.log(`Building pool for DB user "${username}" @ ${host}:${port}/${database}`);

    this.pool = new Pool({
      host,
      port,
      user: username,
      database,
      max: 5,
      ssl,
      password: async () => {
        try {
          const token = await signer.getAuthToken();
          this.logger.log(`IAM token minted for "${username}"`);
          return token;
        } catch (err) {
          this.logger.error(`IAM token failed for "${username}": ${err}`);
          throw err;
        }
      },
    });

    return this.pool;
  }

  async query<T = any>(db: DbName, text: string, params: any[]): Promise<T[]> {
    const normalizedQuery = text.trim().toUpperCase();
    if (
      !normalizedQuery.startsWith('SELECT') ||
      /\b(DELETE|DROP|TRUNCATE|UPDATE|INSERT|ALTER|CREATE)\b/.test(normalizedQuery)
    ) {
      throw new Error(
        'SECURITY BLOCK: Only pure SELECT queries are allowed. Destructive operations are forbidden.',
      );
    }
    const res = await this.getPool().query(text, params);
    return res.rows as T[];
  }

  async onModuleDestroy() {
    await this.pool?.end();
  }
}
