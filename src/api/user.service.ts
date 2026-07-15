import { HttpService } from '@nestjs/axios';
import { ForbiddenException, HttpStatus, Injectable } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
// import { SendOtp, SendUser, VerifyOtp } from './dto/user.dto';
import { AxiosInstance } from 'axios';
import { createInternalServiceClient, isPayloadEncryptionEnabledEnv } from 'src/core/lib/internal-http-client';

@Injectable()
export class ApiUserService {
  private readonly API_URL = '';
  private readonly INTERNAL_API_URL = '';
  private readonly OB_PROCUREMENT_ACCOUNT_ID = '';
  private readonly USER_SERVICE_TOKEN = process.env.USER_SERVICE_TOKEN || '';
  private userServiceInternal: AxiosInstance | null = null;

  constructor(private readonly httpService: HttpService, private readonly configService: ConfigService) {
    this.API_URL = this.configService.get('API_URL') || '';
    this.INTERNAL_API_URL = this.configService.get('INTERNAL_API_URL') || '';
    this.OB_PROCUREMENT_ACCOUNT_ID = this.configService.get('OB_PROCUREMENT_ACCOUNT_ID') || '';
    this.USER_SERVICE_TOKEN = this.configService.get('USER_SERVICE_TOKEN') || '';
  }

  private getUserServiceInternal(): AxiosInstance {
    if (!this.userServiceInternal) {
      const base = (this.API_URL || '').trim().replace(/\/+$/, '');
      if (!base) {
        throw new Error('product-service: API_URL is missing. Ensure env config is loaded before handling requests.');
      }
      this.userServiceInternal = createInternalServiceClient({
        baseURL: base,
        hmacSecret: this.configService.get<string>('INTERNAL_SVC_HMAC_SECRET'),
        cryptoSecret: this.configService.get<string>('CRYPTO_SECRET'),
        cryptoAlgorithm: this.configService.get<string>('CRYPTO_ALGORITHM'),
        payloadEncryptionEnabled: isPayloadEncryptionEnabledEnv(this.configService.get('PAYLOAD_ENCRYPTION_ENABLED')),
      });
    }
    return this.userServiceInternal;
  }

  private getUserServiceExternal(): AxiosInstance {
    const base = (this.INTERNAL_API_URL || '').trim().replace(/\/+$/, '');
    if (!base) {
      throw new Error('product-service: INTERNAL_API_URL is missing. Ensure env config is loaded before handling requests.');
    }
    this.userServiceInternal = createInternalServiceClient({
      baseURL: base,
      hmacSecret: this.configService.get<string>('INTERNAL_SVC_HMAC_SECRET'),
      cryptoSecret: this.configService.get<string>('CRYPTO_SECRET'),
      cryptoAlgorithm: this.configService.get<string>('CRYPTO_ALGORITHM'),
      payloadEncryptionEnabled: isPayloadEncryptionEnabledEnv(this.configService.get('PAYLOAD_ENCRYPTION_ENABLED')),
    });
    return this.userServiceInternal;
  }


  async getUserList(token?: string) {
    const result = await this.getUserServiceInternal().get(`/usvc/oms/v1/userlist`, { headers: { authorization: token } });
    return result.data;
  }

  async getUserListInternal(token?: string) {
    const result = await this.getUserServiceInternal().get(`/usvc/oms/v1/userlist/internal`, { headers: { authorization: token } });
    return result.data;
  }

  async getPolicyList(token?: string) {
    const result = await this.getUserServiceInternal().get(`/usvc/oms/v1/policylist`, { headers: { authorization: token } });
    return result.data;
  }

  async verifyOmsToken(token?: string) {
    const result = await this.getUserServiceInternal().get(`/auth/v1/oms/verifytoken`, { headers: { authorization: token } });

    return result.data;
  }

  async verifyEpToken(token?: string) {
    const result = await this.getUserServiceInternal().get(`/auth/v1/ep/verifytoken`, { headers: { authorization: token } });

    return result.data;
  }

  async verifyCpToken(token?: string) {
    const result = await this.getUserServiceInternal().get(`/auth/v1/cp/verifytoken`, { headers: { authorization: token } });

    return result.data;
  }

  async verifyWbToken(token?: string) {
    const result = await this.getUserServiceInternal().get(`/auth/v1/wb/verifytoken`, { headers: { authorization: token } });

    return result.data;
  }

  async verifyTempToken(token?: string) {
    const result = await this.getUserServiceInternal().get(`/auth/v1/token/verify`, { headers: { authorization: token } });

    return result.data;
  }

  async getShortConfig() {
    const result = await this.getUserServiceExternal().get(`/usvc/v1/config/short/internal`);

    return result.data;
  }

  async getUsersByIds(userIds: string[]) {
    const result = await this.getUserServiceExternal().post(`/usvc/oms/v1/users/getUsersByIds`, { userIds });

    return result.data?.data;
  }
  async getUsersByEmailIds(emailIds: string[]) {
    const result = await this.getUserServiceExternal().post(`/usvc/oms/v1/usersByEmailIds`, { emailIds });

    return result.data?.data;
  }

  async getConfigUsers() {
    const result = await this.getUserServiceExternal().get(`/usvc/oms/v1/config/short/internal`);

    return result.data;
  }

  async getUsersByPolicyByCode(policyCode: string) {
    const result = await this.getUserServiceExternal().get(`/usvc/oms/v1/policy/${policyCode}/users`);
    return result.data;
  }

  async getKamAgentResponse(token?: string) {
    const result = await this.getUserServiceExternal().get(`/usvc/ext/v1/cp/${this.OB_PROCUREMENT_ACCOUNT_ID}/users`, {
      headers: { authorization: token },
    });
    return result.data;
  }


  async createCronProcAccount(body: any) {
    try {
      const result = await this.getUserServiceExternal().post(`/usvc/ext/v1/oms/cron/createAccount`, body, {
        headers: { internalSecret: this.USER_SERVICE_TOKEN },
      });
      return result.data;
    } catch (error) {
      console.error('Error creating procurement account:', error?.response?.data || error.message);
      throw new ForbiddenException(error?.response?.data?.message || error?.message);
    }
  }
}
