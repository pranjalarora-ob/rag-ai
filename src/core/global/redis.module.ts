import { Global, Module } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { RedisService } from './redis.service';

@Global()
@Module({
  exports: [RedisService],
  providers: [
    {
      useFactory: (config: ConfigService) => {
        return new RedisService({
          host: config.get('REDIS_HOST'),
          port: config.get('REDIS_PORT'),
        });
      },
      provide: RedisService,
      inject: [ConfigService],
    },
  ],
})
export class RedisModule {}
