import { Injectable } from '@nestjs/common';
import Redis, { RedisOptions } from 'ioredis';

@Injectable()
export class RedisService extends Redis {
  constructor(options: RedisOptions) {
    super(options);
  }
}
