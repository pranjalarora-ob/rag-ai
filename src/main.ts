import { NestFactory } from '@nestjs/core';
import { ValidationPipe } from '@nestjs/common';
import { json } from 'express';
import { SwaggerModule, DocumentBuilder } from '@nestjs/swagger';
import { AppModule } from './app.module';

async function bootstrap() {
  const app = await NestFactory.create(AppModule);

  app.useGlobalPipes(new ValidationPipe({ transform: true, whitelist: true }));
  app.use(json({ limit: '50mb' }));

  const config = new DocumentBuilder()
    .setTitle('RAG AI')
    .setDescription('RAG + Planner agent over project data')
    .setVersion('1.0')
    .build();
  SwaggerModule.setup('docs', app, SwaggerModule.createDocument(app, config));

  const port = Number(process.env.PORT) || 3000;
  await app.listen(port);
  console.log(`RAG AI running on http://localhost:${port}  (Swagger: /docs)`);
}
bootstrap();
