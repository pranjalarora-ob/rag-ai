import { Module } from '@nestjs/common';
import { CdcController } from './cdc.controller';
import { CdcService } from './cdc.service';
import { RagModule } from '../rag/rag.module';

@Module({
  imports: [RagModule],
  controllers: [CdcController],
  providers: [CdcService],
})
export class CdcModule {}