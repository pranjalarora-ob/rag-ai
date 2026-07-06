import { Module } from '@nestjs/common';
import { CdcController } from './cdc.controller';
import { CdcService } from './cdc.service';

@Module({
  controllers: [CdcController],
  providers: [CdcService],
})
export class CdcModule {}