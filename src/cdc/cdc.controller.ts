import {
  Controller,
} from '@nestjs/common';

import {
  Ctx,
  EventPattern,
  KafkaContext,
  Payload,
} from '@nestjs/microservices';

import { CdcService } from './cdc.service';

@Controller()
export class CdcController {
  constructor(
    private readonly cdcService: CdcService,
  ) {}

  @EventPattern('officebanao.public.projects')
  async handleProjects(
    @Payload() message: any,
    @Ctx() context: KafkaContext,
  ) {
    const topic = context.getTopic();

    const event =
      typeof message.value === 'string'
        ? JSON.parse(message.value)
        : message.value;

    await this.cdcService.process(
      topic,
      event,
    );
  }

  @EventPattern('officebanao.public.quotations')
  async handleQuotation(
    @Payload() message: any,
    @Ctx() context: KafkaContext,
  ) {
    const topic = context.getTopic();

    const event =
      typeof message.value === 'string'
        ? JSON.parse(message.value)
        : message.value;

    await this.cdcService.process(
      topic,
      event,
    );
  }
}