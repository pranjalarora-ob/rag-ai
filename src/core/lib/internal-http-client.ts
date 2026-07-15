import axios, { AxiosInstance, InternalAxiosRequestConfig, AxiosHeaders } from 'axios';
import { Readable } from 'stream';
import { getInternalServiceHeaders } from '../../utils/internal-request-sign';
import { buildInternalServiceJsonRequest, parseServiceResponseBody, isEnvelope } from '../../utils/internal-service-wire-format';
// import { getRequestId } from '../database/sequelize-cls';

function resolveSignPathWithClient(client: AxiosInstance, cfg: InternalAxiosRequestConfig): string {
  try {
    const uri = client.getUri({
      baseURL: cfg.baseURL,
      url: cfg.url,
      params: cfg.params,
      paramsSerializer: cfg.paramsSerializer,
    } as any);
    const u = new URL(uri);
    return `${u.pathname}${u.search}`;
  } catch {
    const rel = cfg.url == null ? '/' : String(cfg.url);
    return rel.startsWith('/') ? rel : `/${rel}`;
  }
}

function isEnvelopeString(s: string): boolean {
  try {
    const o = JSON.parse(s);
    return isEnvelope(o);
  } catch {
    return false;
  }
}

function applyTracingInterceptors(client: AxiosInstance): AxiosInstance {
  client.interceptors.request.use((config: InternalAxiosRequestConfig) => {
    const headers = AxiosHeaders.from(config.headers || {});

    // headers.set('x-request-id', getRequestId() ?? '');

    headers.set('x-service-name', process.env.SERVICE_NAME || 'order-service');

    headers.set('x-system-request', 'true');

    config.headers = headers;

    return config;
  });

  return client;
}

export interface InternalServiceClientOptions extends Record<string, any> {
  hmacSecret?: string;
  cryptoSecret?: string;
  cryptoAlgorithm?: string;
  payloadEncryptionEnabled?: boolean;
}

export function isPayloadEncryptionEnabledEnv(value: unknown): boolean {
  return String(value ?? 'true').toLowerCase() !== 'false';
}

export function createInternalServiceClient(options: InternalServiceClientOptions = {}): AxiosInstance {
  const { hmacSecret, cryptoSecret, cryptoAlgorithm, payloadEncryptionEnabled = true, ...axiosOptions } = options;
  const client = axios.create({ timeout: 60000, ...axiosOptions });
  applyTracingInterceptors(client);

  const resolveSecret = (): string => {
    const secret = (hmacSecret || cryptoSecret || '').trim();
    if (!secret) {
      throw new Error('INTERNAL_SVC_HMAC_SECRET is not configured');
    }
    return secret;
  };

  const applyInternalRequest = (cfg: InternalAxiosRequestConfig): InternalAxiosRequestConfig => {
    const secret = resolveSecret();
    const method = (cfg.method || 'get').toUpperCase();
    const path = resolveSignPathWithClient(client, cfg);

    if (['GET', 'HEAD', 'OPTIONS', 'DELETE'].includes(method)) {
      const h = getInternalServiceHeaders({ secret, method, path, rawBody: '' });
      const headers = AxiosHeaders.from(cfg.headers || {});
      Object.entries(h).forEach(([k, v]) => headers.set(k, v));
      cfg.headers = headers as any;
      return cfg;
    }

    if (cfg.data instanceof Readable) {
      return cfg;
    }

    if (typeof FormData !== 'undefined' && cfg.data instanceof FormData) {
      return cfg;
    }

    if (typeof cfg.data === 'string') {
      const rawBody = cfg.data;
      const h = getInternalServiceHeaders({ secret, method, path, rawBody });
      const headers = AxiosHeaders.from(cfg.headers || {});
      Object.entries(h).forEach(([k, v]) => headers.set(k, v));
      headers.set('Content-Type', 'application/json', false);
      cfg.headers = headers as any;
      return cfg;
    }

    if (cfg.data != null && typeof cfg.data === 'object' && !Buffer.isBuffer(cfg.data)) {
      if (isEnvelopeString(JSON.stringify(cfg.data))) {
        const rawBody = JSON.stringify(cfg.data);
        const h = getInternalServiceHeaders({ secret, method, path, rawBody });
        const headers = AxiosHeaders.from(cfg.headers || {});
        Object.entries(h).forEach(([k, v]) => headers.set(k, v));
        headers.set('Content-Type', 'application/json', false);
        cfg.headers = headers as any;
        cfg.data = rawBody;
        return cfg;
      }
      if (!payloadEncryptionEnabled) {
        const rawBody = JSON.stringify(cfg.data);
        const h = getInternalServiceHeaders({ secret, method, path, rawBody });
        const headers = AxiosHeaders.from(cfg.headers || {});
        Object.entries(h).forEach(([k, v]) => headers.set(k, v));
        headers.set('Content-Type', 'application/json', false);
        cfg.headers = headers as any;
        cfg.data = rawBody;
        return cfg;
      }
      const { headers: signHeaders, body } = buildInternalServiceJsonRequest({
        method,
        path,
        payloadObject: cfg.data,
        secret,
        algorithm: cryptoAlgorithm,
      });
      const merged = AxiosHeaders.from(cfg.headers || {});
      Object.entries(signHeaders).forEach(([k, v]) => merged.set(k, v));
      cfg.headers = merged as any;
      cfg.data = body;
      return cfg;
    }

    const rawBody = cfg.data == null ? '' : String(cfg.data);
    const h = getInternalServiceHeaders({ secret, method, path, rawBody });
    const headers = AxiosHeaders.from(cfg.headers || {});
    Object.entries(h).forEach(([k, v]) => headers.set(k, v));
    cfg.headers = headers as any;
    return cfg;
  };

  const unwrapResponse = (res: any) => {
    if (res && res.data !== undefined) {
      res.data = parseServiceResponseBody(res.data, cryptoSecret || hmacSecret, cryptoAlgorithm);
    }
    return res;
  };

  client.interceptors.request.use((cfg) => applyInternalRequest(cfg));
  client.interceptors.response.use(unwrapResponse, (err: any) => {
    if (err.response && err.response.data !== undefined) {
      try {
        err.response.data = parseServiceResponseBody(err.response.data, cryptoSecret || hmacSecret, cryptoAlgorithm);
      } catch {
        /* leave as-is */
      }
    }
    return Promise.reject(err);
  });

  return client;
}
