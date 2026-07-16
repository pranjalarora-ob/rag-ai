import { Module } from '@nestjs/common';
import { ConfigModule, ConfigService } from '@nestjs/config';
import { MongooseModule } from '@nestjs/mongoose';
import { RagModule } from './rag/rag.module';
import { ChatModule } from './chat/chat.module';
import { HttpModule } from '@nestjs/axios';
import { ThrottlerGuard, ThrottlerModule } from '@nestjs/throttler';
import { CacheModule } from '@nestjs/cache-manager';
import { AppService } from './app.service';
import { RedisModule } from './core/global/redis.module';
import * as dotenv from 'dotenv';
import * as Joi from 'joi';
import { GetSecretValueCommand, SecretsManagerClient } from '@aws-sdk/client-secrets-manager';

@Module({
  imports: [
    ConfigModule.forRoot({ 
      isGlobal: true,
      load: [
        async () => {
          const validate = (keys: NodeJS.ProcessEnv | dotenv.DotenvConfigOutput) => {
            const validationSchema = Joi.object({
              REDIS_HOST: Joi.string().optional(),
              REDIS_PORT: Joi.string().optional(),
              REDIS_CONFIG_DISABLE: Joi.string().optional(),
            }).unknown();
            const { value: envVars, error } = validationSchema.prefs({ errors: { label: 'key' } }).validate(keys);

            if (error) {
              throw new Error(`Config validation error: ${error.message}`);
            }

            return envVars;
          };
          if (process.env.AWS_SECRETSMANAGER_ENTRY) {
            const command = new GetSecretValueCommand({
              SecretId: process.env.AWS_SECRETSMANAGER_ENTRY,
            });

            const secretsManagerClient = new SecretsManagerClient({
              region: process.env.AWS_REGION,
            });

            const keys = (await secretsManagerClient.send(command)).SecretString || '';
            const envs = dotenv.parse(Buffer.from(keys));

            return validate(envs);
          }
          return validate(process.env);
      }],
     }),
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
    RedisModule,
  ],
  exports: [HttpModule, RedisModule],
  providers: [
    AppService,
    {
      provide: "APP_GUARD",
      useClass: ThrottlerGuard,
    },
  ],
})
export class AppModule {}
