import { Module } from '@nestjs/common';
import { HttpModule } from '@nestjs/axios';
import { ApiUserService } from './user.service';
import { ApiController } from './api.controller';

@Module({
  imports: [HttpModule],
  providers: [ApiUserService],
  controllers: [ApiController],
  exports: [ApiUserService],
})
export class ApiUserModule {}