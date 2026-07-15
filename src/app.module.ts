import { Module } from '@nestjs/common';
import { ConfigModule, ConfigService } from '@nestjs/config';
import { MongooseModule } from '@nestjs/mongoose';
import { RagModule } from './rag/rag.module';
import { ChatModule } from './chat/chat.module';
import { HttpModule } from '@nestjs/axios';

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
    RagModule,
    ChatModule,
  ],
  exports: [HttpModule],
})
export class AppModule {}
