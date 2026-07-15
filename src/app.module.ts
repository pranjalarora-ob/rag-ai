import { Module } from '@nestjs/common';
import { ConfigModule, ConfigService } from '@nestjs/config';
import { MongooseModule } from '@nestjs/mongoose';
import { RagModule } from './rag/rag.module';
import { ChatModule } from './chat/chat.module';
import { HttpModule } from '@nestjs/axios';
import { ThrottlerGuard, ThrottlerModule } from '@nestjs/throttler';
import { CacheModule } from '@nestjs/cache-manager';
import { AppService } from './app.service';

@Module({
  imports: [
    ConfigModule.forRoot({ isGlobal: true }),
    MongooseModule.forRootAsync({
      imports: [ConfigModule],
      useFactory: (config: ConfigService) => ({
        uri: config.get<string>('MONGO_URL'),
      }),
      inject: [ConfigService],
    }),
    HttpModule.register({
      timeout: 1000 * 60,
    }),
    ThrottlerModule.forRoot([
      {
        ttl: 60000,
        limit: 1000,
      },
    ]),
    CacheModule.register({
      isGlobal: true,
    }),
    RagModule,
    ChatModule,
  ],
  exports: [HttpModule],
  providers: [
    AppService,
    {
      provide: "APP_GUARD",
      useClass: ThrottlerGuard,
    },
  ],
})
export class AppModule {}
