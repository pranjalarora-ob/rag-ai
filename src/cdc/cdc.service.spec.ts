import { Test, TestingModule } from '@nestjs/testing';
import { CdcService } from './cdc.service';

describe('CdcService', () => {
  let service: CdcService;

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [CdcService],
    }).compile();

    service = module.get<CdcService>(CdcService);
  });

  it('should be defined', () => {
    expect(service).toBeDefined();
  });
});
