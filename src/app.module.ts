import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { RagModule } from './rag/rag.module';

@Module({
  imports: [ConfigModule.forRoot({ isGlobal: true }), RagModule],
})
export class AppModule {}
