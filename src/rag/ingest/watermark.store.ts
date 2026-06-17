import { Injectable } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import * as fs from 'fs';
import * as path from 'path';

/**
 * Tracks the last-synced `updated_at` per source so incremental syncs only
 * re-read changed rows. File-backed for simplicity (single instance);
 * swap for Redis/a DB table if you scale to multiple instances.
 */
@Injectable()
export class WatermarkStore {
  private readonly file: string;

  constructor(config: ConfigService) {
    const dir =
      config.get<string>('INGEST_STATE_DIR') || path.join(process.cwd(), '.ingest-state');
    fs.mkdirSync(dir, { recursive: true });
    this.file = path.join(dir, 'watermarks.json');
  }

  private read(): Record<string, string> {
    try {
      return JSON.parse(fs.readFileSync(this.file, 'utf8'));
    } catch {
      return {};
    }
  }

  get(key: string): string | undefined {
    return this.read()[key];
  }

  set(key: string, value: string) {
    const all = this.read();
    all[key] = value;
    fs.writeFileSync(this.file, JSON.stringify(all, null, 2));
  }
}
