import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { RagModule } from './rag/rag.module';
import { CdcModule } from './cdc/cdc.module';

@Module({
  imports: [ConfigModule.forRoot({ isGlobal: true }), RagModule, CdcModule],
})
export class AppModule {}
