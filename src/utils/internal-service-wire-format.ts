import { encrypt, decrypt } from './encryption';
import { getInternalServiceHeaders } from './internal-request-sign';

export function isEnvelope(body: any): boolean {
  if (!body || typeof body !== 'object' || Array.isArray(body)) return false;
  const keys = Object.keys(body);
  if (keys.length !== 2 || !keys.includes('iv') || !keys.includes('encryptedData')) return false;
  return typeof body.iv === 'string' && typeof body.encryptedData === 'string';
}

export function buildInternalServiceJsonRequest(opts: { method: string; path: string; payloadObject: any; secret: string; algorithm?: string }): {
  headers: Record<string, string>;
  body: string;
} {
  if (!opts.secret) {
    throw new Error('INTERNAL_SVC_HMAC_SECRET is not configured');
  }
  const innerJson = JSON.stringify(opts.payloadObject);
  const enc = encrypt(innerJson, opts.secret, opts.algorithm);
  const rawBody = JSON.stringify(enc);
  return {
    headers: {
      'Content-Type': 'application/json',
      ...getInternalServiceHeaders({ secret: opts.secret, method: opts.method, path: opts.path, rawBody }),
    },
    body: rawBody,
  };
}

export function parseServiceResponseBody(body: any, secret?: string, algorithm?: string): any {
  if (!isEnvelope(body)) return body;
  return JSON.parse(decrypt(body.encryptedData, body.iv, secret || '', algorithm));
}
