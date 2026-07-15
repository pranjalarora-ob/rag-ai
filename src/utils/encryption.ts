import * as crypto from 'crypto';

let cachedKey: Buffer | null = null;
let cachedSecret = '';

function getSecretKey(secret: string): Buffer {
  if (cachedKey && cachedSecret === secret) return cachedKey;
  cachedSecret = secret;
  cachedKey = crypto.createHash('sha256').update(String(secret)).digest();
  return cachedKey;
}

export function encrypt(data: string, secret: string, algorithm = 'aes-256-cbc'): { iv: string; encryptedData: string } {
  const iv = crypto.randomBytes(16);
  const cipher = crypto.createCipheriv(algorithm, getSecretKey(secret), iv);
  let encrypted = cipher.update(data, 'utf8', 'hex');
  encrypted += cipher.final('hex');
  return { iv: iv.toString('hex'), encryptedData: encrypted };
}

export function decrypt(encryptedData: string, ivHex: string, secret: string, algorithm = 'aes-256-cbc'): string {
  const decipher = crypto.createDecipheriv(algorithm, getSecretKey(secret), Buffer.from(ivHex, 'hex'));
  let decrypted = decipher.update(encryptedData, 'hex', 'utf8');
  decrypted += decipher.final('utf8');
  return decrypted;
}
